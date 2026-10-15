"""Build URL explanations from occlusion facts."""
import argparse
import itertools
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from model import Scorer
from preprocess import BRANDS, SUSPICIOUS, STRUCTURE_NAMES, advanced_features, clean_url

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
FEATURE_START = 81920
GROUPS = {"host_patterns": (0, 32768), "path_patterns": (32768, 65536), "url_words": (65536, 81920)}
LABELS = (
    "URL length|host length|path length|host labels|hyphens in the host|digits in the host|"
    "path slashes|path equals signs|percent-encoded characters|dots in the URL|"
    "longest host label|suspicious words|brand words|an IP address as host|an @ sign|"
    "punycode in the host|HTTPS|a leading www.|a query string|a file extension|"
    "digits in the URL|a random-looking host"
).split("|")
NAMES = dict(zip(STRUCTURE_NAMES, LABELS), host_patterns="host character patterns",
             path_patterns="path/query character patterns", url_words="words in the URL")
UNITS = [*GROUPS, *STRUCTURE_NAMES]
PHRASES = dict(zip(UNITS, (
    "the host name '{value}'", "{value}", "the words in the URL ({value})",
    "the URL length ({value})", "the host length ({value})", "{value}",
    "a host made of {value}", "{value} in the host", "{value} in the host",
    "{value} in the path", "{value} in the path", "{value}", "{value} in the URL",
    "a longest host part of {value}", "{value}", "{value}", "{value}", "{value}",
    "{value}", "{value}", "{value}", "{value}", "{value}", "the share of digits in the URL ({value})",
    "how random the host name looks ({value} bits per character)",
)))
COUNTS = dict(zip(
    ("log_len_url", "log_len_host", "log_host_labels", "log_host_hyphens",
     "log_host_digits", "log_path_slashes", "log_path_equals", "log_url_pct",
     "log_url_dots", "log_longest_label"),
    (("character", "characters"), ("character", "characters"),
     ("dot-separated part", "dot-separated parts"), ("hyphen", "hyphens"),
     ("digit", "digits"), ("slash", "slashes"), ("'=' sign", "'=' signs"),
     ("percent-encoded character", "percent-encoded characters"),
     ("dot", "dots"), ("character", "characters")),
))
FLAGS = {
    "is_ip_host": ("an IP address instead of a host name", "a normal host name rather than an IP address"),
    "has_at": ("an '@' sign in the URL", "no '@' sign"),
    "has_punycode": ("punycode ('xn--') in the host", "no punycode in the host"),
    "is_https": ("HTTPS", "plain HTTP instead of HTTPS"),
    "starts_www": ("a leading 'www.'", "no leading 'www.'"),
    "has_query": ("a query string", "no query string"),
    "has_file_ext": ("a page file ending such as .php or .html", "no page file ending"),
}
WORD_LISTS = {
    "log_suspicious_words": (SUSPICIOUS, "account-related word", "account-related words", " such as 'login' or 'verify'"),
    "log_brand_words": (BRANDS, "brand name", "brand names", " such as 'paypal' or 'apple'"),
}
SYSTEM_PROMPT = (
    "You explain a phishing detector's decision to a non-technical reader. "
    "Use only the facts given and keep each fact's direction (toward phishing or toward clean). "
    "Do not add new reasons, numbers, advice or opinions about the website. "
    "Write 2 or 3 short sentences in plain English and start with the first reason."
)

def save_reference(df):
    values = advanced_features(df.loc[df.label == 0, "url"]).tocsr()[:, FEATURE_START:].toarray()
    path = ROOT / "models" / "explain_reference.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(np.median(values, axis=0).astype(float).tolist(), indent=2))

@lru_cache(maxsize=1)
def _reference():
    values = json.loads((ROOT / "models" / "explain_reference.json").read_text())
    if len(values) != len(STRUCTURE_NAMES):
        raise ValueError("Feature reference has the wrong width")
    return np.asarray(values, dtype=np.float32)

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def _neutralize(x, keys, reference):
    row = x.copy()
    for key in keys:
        if key in GROUPS:
            row[:, slice(*GROUPS[key])] = 0
        else:
            i = STRUCTURE_NAMES.index(key)
            row[:, FEATURE_START + i] = reference[i]
    return row

def _value(key, x, url):
    parsed = urlsplit(url)
    groups = {
        "host_patterns": parsed.hostname or "",
        "path_patterns": (parsed.path + (f"?{parsed.query}" if parsed.query else "")),
        "url_words": " ".join(re.findall(r"[a-z0-9]{2,}", url)[:4]),
    }
    if key in groups:
        value = groups[key]
        return value[:39] + "…" if key == "path_patterns" and len(value) > 40 else value
    i = STRUCTURE_NAMES.index(key)
    value = float(x[0, FEATURE_START + i])
    if key.startswith("log_"):
        return str(round(math.expm1(value)))
    if 13 <= i <= 19:
        return "yes" if value else "no"
    return {"host_entropy": f"{5 * value:.1f}",
            "url_digit_ratio": f"{100 * value:.1f}%"}.get(key, f"{value:.1f}")

def _count(n, singular, plural):
    return f"no {plural}" if n == 0 else f"{n} {singular if n == 1 else plural}"

def _join_quoted(words):
    quoted = [f"'{word}'" for word in words]
    return (f"{', '.join(quoted[:-1])} and {quoted[-1]}" if len(quoted) > 1
            else "".join(quoted))

VALUE_FORMATTERS = {
    "path_patterns": lambda value, _: f"the path '{value}'" if value else "an empty path",
    "log_len_path": lambda value, _: (
        "an empty path" if int(value) == 0 else
        f"the path length ({_count(int(value), 'character', 'characters')})"),
    "url_words": lambda _, url: " ".join(
        [w for w in re.findall(r"[a-z0-9]{2,}", url)
         if w not in {"http", "https", "www"}][:4]),
}

def _phrase(key, x, url):
    value = _value(key, x, url)
    if key in FLAGS:
        value = FLAGS[key][value == "no"]
    elif key in COUNTS:
        value = _count(int(value), *COUNTS[key])
    elif key in WORD_LISTS:
        words, singular, plural, examples = WORD_LISTS[key]
        n = int(value)
        value = (f"{_count(n, singular, plural)}{examples}" if n == 0 else
                 f"the {singular if n == 1 else plural} " +
                 _join_quoted([w for w in words if w in clean_url(url)]))
    else:
        value = VALUE_FORMATTERS.get(key, lambda item, _: item)(value, url)
    return PHRASES[key].format(value=value)

def _strength(effect):
    return "strongly" if abs(effect) >= 3 else "clearly" if abs(effect) >= 1 else "slightly"

def _fmt(score, threshold):
    if abs(score - threshold) < 0.01:
        return f"{score:.3f}"
    return ("above 0.99" if score < 1 and f"{score:.2f}" == "1.00" else
            "below 0.01" if score > 0 and f"{score:.2f}" == "0.00" else
            f"{score:.2f}")

def _narrative(label, p, threshold, x, url, selected, parts, combo, combo_keys):
    fmt_p, t = _fmt(p, threshold), f"{threshold:.3f}"
    verdict = (
        f"This URL is flagged as phishing with a very high score ({fmt_p}; anything from {t} up is flagged)."
        if label == "phishing" and p >= .99 else
        f"This URL is flagged as phishing, but only just: its score of {fmt_p} is close to the {t} cut-off."
        if label == "phishing" else
        f"This URL is treated as clean, but it is a borderline case: its score of {fmt_p} is just under the {t} cut-off."
        if p >= .5 else
        f"This URL looks clean: its score of {fmt_p} is far below the {t} cut-off.")
    first = selected[0]["effect"]
    sentences = [verdict, f"The strongest signal is {_phrase(selected[0]['key'], x, url)}. "
                 f"This {_strength(first)} points toward "
                 f"{'phishing' if first > 0 else 'a legitimate site'}."]
    sign = 1 if label == "phishing" else -1
    secondary = [(_phrase(r["key"], x, url), r["effect"], r["effect"] * sign > 0)
                 for r in selected[1:]]
    if len(secondary) == 2 and all(item[2] for item in secondary):
        a, b = (item[0] for item in secondary)
        sentences.append(f"{a[0].upper() + a[1:]} and {b} add to this.")
    else:
        sentences.extend(
            f"{phrase[0].upper() + phrase[1:]} adds to this." if supports else
            f"{phrase[0].upper() + phrase[1:]} pulls {_strength(effect)} the other way."
            for phrase, effect, supports in secondary)
    if abs(combo["joint"]) >= 1:
        q, fmt_q = combo["p_without_both"], _fmt(combo["p_without_both"], threshold)
        a, b = (_phrase(key, x, url) for key in combo_keys)
        if abs(p - q) < 0.02:
            sentences.append(f"Even without {a} and {b} the score would stay at {fmt_q}, "
                             "so many signals agree.")
        else:
            ending = ", which would change the verdict." if (q >= threshold) != (p >= threshold) else "."
            sentences.append(f"Together, {a} and {b} have the largest joint effect: without "
                             f"both the score would {'fall' if q < p else 'rise'} from "
                             f"{fmt_p} to {fmt_q}{ending}")
    else:
        sentences.append("No single signal or pair dominates; the score comes from many small signals together.")
    if parts:
        names = " and ".join(f"'{part['part']}'" for part in parts)
        sentences.append(f"In the URL itself, {names} matter{'s' if len(parts) == 1 else ''} most.")
    return " ".join(sentences)

def _part_urls(url):
    try:
        parsed = urlsplit(url)
    except ValueError:
        return []
    segments = {"host": (parsed.hostname or "").split(".")[:-1],
                "path": parsed.path.split("/"), "query": parsed.query.split("&")}
    parts = [(kind, i, value) for kind, values in segments.items()
             for i, value in enumerate(values) if value][:20]
    result = []
    for kind, index, part in parts:
        changed = list(parsed)
        if kind == "host":
            try:
                port = parsed.port
            except ValueError:
                continue
            host = parsed.hostname
            if not host or ":" in host:
                continue
            labels = host.split(".")
            labels.pop(index)
            user, separator, _ = changed[1].rpartition("@")
            changed[1] = (f"{user}@" if separator else "") + ".".join(labels)
            if port is not None:
                changed[1] += f":{port}"
        else:
            i, separator = (2, "/") if kind == "path" else (3, "&")
            changed[i] = separator.join(
                value for j, value in enumerate(changed[i].split(separator)) if j != index)
        result.append((part, urlunsplit(changed)))
    return result

def _facts(p, reasons, parts, combo):
    sentence = lambda name, e: f"{name} {_strength(e)} pushed the score toward {'phishing' if e > 0 else 'clean'}"
    reason_facts = [sentence(f"{r['name']} ({r['value']})", r["effect"]) for r in reasons]
    reason_facts += ["no additional feature unit was isolated"] * (3 - len(reason_facts))
    part_facts = [sentence(f"the URL part '{part['part']}'", part["effect"]) for part in parts]
    part_facts += ["no additional URL part was isolated"] * (2 - len(part_facts))
    pair = (f"{combo['a']} and {combo['b']} together: removing both would move the score "
            f"from {p:.2f} to {combo['p_without_both']:.2f}")
    return reason_facts, part_facts, pair

@lru_cache(maxsize=1)
def _load_llm():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(min(8, os.cpu_count() or 1))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model.eval()
        return torch, tokenizer, model
    except Exception:
        return None

def _valid_text(text, prompt, label, reason):
    allowed = set(re.findall(r"\d+(?:\.\d+)?", prompt))
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    verdict = r"\bphishing\b" if label == "phishing" else r"\b(clean|legitimate)\b"
    return (len(text) >= 20 and bool(re.search(verdict, text, re.IGNORECASE))
            and reason in text.casefold() and all(number in allowed for number in numbers))

def _llm_text(url, label, p, threshold, reasons, parts, combo):
    reason_facts, part_facts, pair = _facts(p, reasons, parts, combo)
    prompt = "\n".join((
        f"URL: {url}",
        f"Verdict: {label} (score {p:.2f}; URLs scoring {threshold:.2f} or more are "
        "flagged as phishing)",
        "Main reasons, strongest first:", *(f"- {fact}" for fact in reason_facts),
        "URL parts that matter most:", *(f"- {fact}" for fact in part_facts),
        f"Strongest combination: {pair}", "Explain why the detector gave this verdict.",
    ))
    loaded = _load_llm()
    if not loaded:
        return None
    try:
        torch, tokenizer, model = loaded
        tokens = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(tokens, attention_mask=tokens.new_ones(tokens.shape),
                                    do_sample=False, max_new_tokens=120)
        text = tokenizer.decode(output[0, tokens.shape[-1]:], skip_special_tokens=True).strip()
        reason = " ".join(reasons[0]["name"].split()[:2]).casefold()
        return text if _valid_text(text, prompt, label, reason) else None
    except Exception:
        return None

def _empty(text="The input is empty or not a usable URL, so there is nothing to explain."):
    return dict(label="clean", probability=0.5, text=text, facts=text, reasons=[],
                parts=[], combination={}, generated_by="template")

def explain(url, scorer=None, use_llm=True) -> dict:
    try:
        url = clean_url(url)
        if not url:
            return _empty()
        scorer = scorer or Scorer("advanced")
        p = float(scorer.predict_proba([url])[0])
        threshold = scorer.threshold
        label = "phishing" if p >= threshold else "clean"
        x, reference = advanced_features([url]).toarray(), _reference()
        full = float(_logit(p))
        effects = [{"key": k, "effect": full - float(_logit(s)), "p_without": float(s)}
                   for k, s in zip(UNITS, scorer.score_matrix(
                       np.vstack([_neutralize(x, [k], reference) for k in UNITS])))]
        sign = 1 if label == "phishing" else -1
        selected = sorted(effects, key=lambda r: (r["effect"] * sign <= 0, -abs(r["effect"])))[:3]
        reasons = [{"name": NAMES[r["key"]], "value": _value(r["key"], x, url),
                    "effect": round(r["effect"], 4), "p_without": round(r["p_without"], 4)}
                   for r in selected]
        pairs = list(itertools.combinations(
            sorted(effects, key=lambda r: abs(r["effect"]), reverse=True)[:4], 2))
        a, b, joint, p_without = max(
            ((a, b, full - float(_logit(score)), float(score)) for (a, b), score in zip(
                pairs, scorer.score_matrix(np.vstack([
                    _neutralize(x, [a["key"], b["key"]], reference) for a, b in pairs])))),
            key=lambda row: abs(row[2]))
        combo_keys = (a["key"], b["key"])
        combo = {"a": NAMES[a["key"]], "b": NAMES[b["key"]], "joint": round(joint, 4),
                 "synergy": round(joint - a["effect"] - b["effect"], 4),
                 "p_without_both": round(p_without, 4)}
        candidates = _part_urls(url)
        part_scores = scorer.predict_proba([v for _, v in candidates]) if candidates else []
        parts = [{"part": r["part"], "effect": round(r["effect"], 4),
                  "p_without": round(r["p_without"], 4)}
                 for r in sorted(
                     [{"part": part, "effect": full - float(_logit(score)), "p_without": float(score)}
                      for (part, _), score in zip(candidates, part_scores)],
                     key=lambda r: -abs(r["effect"]))
                 if r["effect"] * sign > 0][:2]
        facts = _narrative(label, p, threshold, x, url, selected, parts, combo, combo_keys)
        text = _llm_text(url, label, p, threshold, reasons, parts, combo) if use_llm else None
        return dict(label=label, probability=round(p, 4), text=text or facts, facts=facts,
                    reasons=reasons, parts=parts, combination=combo,
                    generated_by=MODEL_ID if text else "template")
    except Exception:
        return _empty("The explanation could not be calculated for this URL.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    print(json.dumps(explain(args.url, use_llm=not args.no_llm), indent=2))

if __name__ == "__main__":
    main()
