"""URL cleaning and the two feature sets. Shared by training and predict()."""
import html
import math
import re

import numpy as np
import tldextract
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer

MAX_CHARS = 2048  # Bound hashing work for unusually long or malicious inputs.
SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://")
SUSPICIOUS = ("login", "signin", "verify", "secure", "account", "update",
              "webscr", "banking", "confirm", "password")
BRANDS = ("paypal", "apple", "microsoft", "office", "outlook", "amazon",
          "netflix", "facebook", "chase", "wells")
# Use the bundled suffix list so parsing never waits on a network lookup.
_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None,
                                 include_psl_private_domains=True)


def _hasher(ngrams, n_features, analyzer="char"):
    # Fixed-size hashing avoids storing a vocabulary and keeps inference lightweight.
    return HashingVectorizer(analyzer=analyzer, ngram_range=ngrams,
                             n_features=n_features, alternate_sign=False,
                             norm="l2", lowercase=False, dtype=np.float32,
                             token_pattern=r"[a-z0-9]{2,}")


BASELINE_HASH = _hasher((3, 5), 2**16)
HOST_HASH = _hasher((2, 5), 2**15)
PATH_HASH = _hasher((3, 5), 2**15)
WORD_HASH = _hasher((1, 1), 2**14, analyzer="word")

# Fixed order keeps the dense structure block readable in the EDA.
STRUCTURE_NAMES = [
    "log_len_url", "log_len_host", "log_len_path", "log_host_labels",
    "log_host_hyphens", "log_host_digits", "log_path_slashes",
    "log_path_equals", "log_url_pct", "log_url_dots",
    "log_longest_label", "log_suspicious_words", "log_brand_words",
    "is_ip_host", "has_at", "has_punycode", "is_https", "starts_www",
    "has_query", "has_file_ext", "url_digit_ratio", "host_entropy",
]


def clean_url(raw) -> str:
    """Lowercase, HTML-unescape, drop control chars, cap length. Never raises."""
    if raw is None:
        return ""
    text = html.unescape(raw if isinstance(raw, str) else str(raw))
    text = re.sub(r"[\x00-\x1f\x7f]", "", text).strip().lower()
    return re.sub(r"\s+", "%20", text)[:MAX_CHARS]  # Keep spaces as their URL-encoded form.


def split_url(url: str) -> tuple[str, str]:
    """Return (host, path+query+fragment) of a cleaned URL without parsing errors."""
    rest = SCHEME.sub("", url)
    authority, path = re.match(r"([^/?#]*)(.*)", rest, re.S).groups()
    host = authority.rsplit("@", 1)[-1].split(":")[0].strip(".")  # Ignore userinfo and port.
    return host, path


def eval_key(raw) -> str:
    """The organisers' normalisation: case, scheme, leading www., trailing slashes."""
    return re.sub(r"^www\.", "", SCHEME.sub("", clean_url(raw))).rstrip("/")


def registered_domain(raw) -> str:
    """eTLD+1 (private suffixes count, so x.blogspot.com is its own site)."""
    host = re.sub(r"^www\.", "", split_url(clean_url(raw))[0])
    parts = _extract(host)  # Local suffix data avoids network calls during scoring.
    return f"{parts.domain}.{parts.suffix}" if parts.domain and parts.suffix else host


def baseline_features(urls) -> csr_matrix:
    """Hashed character 3-5-grams of the whole cleaned URL, L2-normalised."""
    # Hashing needs no stored vocabulary.
    return BASELINE_HASH.transform([clean_url(u) for u in urls])


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = np.unique(list(text), return_counts=True)[1] / len(text)
    return float(-(counts * np.log2(counts)).sum())


def structure(url: str, host: str, path: str) -> list[float]:
    """Hand-made URL structure features; counts are log-scaled so no scaler is needed."""
    labels = host.split(".") if host else []
    counts = [len(url), len(host), len(path), len(labels), host.count("-"),
              sum(c.isdigit() for c in host), path.count("/"), path.count("="),
              url.count("%"), url.count("."), max(map(len, labels), default=0),
              sum(w in url for w in SUSPICIOUS), sum(b in url for b in BRANDS)]
    flags = [bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host)), "@" in url,
             "xn--" in host, url.startswith("https://"), host.startswith("www."),
             "?" in path, bool(re.search(r"\.(php|html?|asp)", path))]
    ratios = [sum(c.isdigit() for c in url) / max(len(url), 1), _entropy(host) / 5]
    return [math.log1p(c) for c in counts] + [float(f) for f in flags] + ratios  # Tame long-URL count tails.


def advanced_features(urls) -> csr_matrix:
    """Host and path n-grams kept apart, word tokens, plus structure features."""
    cleaned = [clean_url(u) for u in urls]
    hosts, paths = zip(*map(split_url, cleaned)) if cleaned else ((), ())
    dense = np.array([structure(u, h, p) for u, h, p in zip(cleaned, hosts, paths)],
                     dtype=np.float32).reshape(len(cleaned), -1)
    # Separate blocks let the model distinguish host clues from path clues.
    return hstack([HOST_HASH.transform(hosts), PATH_HASH.transform(paths),
                   WORD_HASH.transform(cleaned), csr_matrix(dense)],
                  format="csr", dtype=np.float32)


FEATURES = {"baseline": baseline_features, "advanced": advanced_features}
