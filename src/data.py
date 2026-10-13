"""Load the CSVs, clean train.csv and write the data audit (reports/audit.json)."""
import json
from pathlib import Path

import pandas as pd

from preprocess import BRANDS, SUSPICIOUS, clean_url, eval_key, registered_domain, split_url

ROOT = Path(__file__).resolve().parents[1]
HTML_ENTITY = r"&(?:amp|gt|lt|quot);"
BARE_HOMEPAGE = r"https?://(www\.)?[^/?#]+/?"


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / f"{name}.csv", dtype={"url": str}, keep_default_na=False)
    df["key"] = df["url"].map(eval_key)  # One stable key for duplicate and overlap checks.
    df["host"] = df["url"].map(lambda u: split_url(clean_url(u))[0].removeprefix("www."))  # Ignore www. for host overlap.
    df["domain"] = df["url"].map(registered_domain)  # Compare sites at the registrable-domain level.
    return df


def clean_train(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop every row of a normalised URL with conflicting labels, then keep one row per key."""
    # Drop conflicting URLs, not hosts: one site can legitimately serve both labels.
    conflict = df.groupby("key")["label"].transform("nunique") > 1
    kept = df[~conflict]
    duplicate = kept.duplicated("key")  # Keep one normalized URL after conflicts are gone.
    out = kept[~duplicate].reset_index(drop=True)
    log = {"rows_in": len(df), "conflicting_keys": int(df.loc[conflict, "key"].nunique()),
           "dropped_conflict_rows": int(conflict.sum()),
           "dropped_duplicate_rows": int(duplicate.sum()), "rows_out": len(out),
           "phishing_out": int(out.label.sum()), "legit_out": int((out.label == 0).sum())}
    return out, log


def characteristics(df: pd.DataFrame) -> dict:
    """Share of legitimate / phishing URLs with each property."""
    url = df["url"].str.lower()
    flags = {
        "has_path_or_query": df["key"].str.contains(r"[/?#]"),  # URL-specific, unlike host features.
        "starts_www": url.str.contains(r"^[a-z]+://www\.|^www\."),
        "digit_in_host": df["host"].str.contains(r"\d"),
        "3plus_host_labels": df["host"].str.count(r"\.") >= 2,
        "hyphen_in_host": df["host"].str.contains("-", regex=False),
        "credential_word": url.str.contains("|".join(SUSPICIOUS)),
        "brand_word": url.str.contains("|".join(BRANDS)),
        "html_entity": df["url"].str.contains(HTML_ENTITY),
    }
    return {name: {"legit": round(float(f[df.label == 0].mean()), 3),
                   "phish": round(float(f[df.label == 1].mean()), 3)} for name, f in flags.items()}


def overlap(ev: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Exact URL, normalized URL, host, then registered-domain overlap."""
    # The ladder shows whether scores can benefit from seeing the same site in training.
    levels = {"exact_url": ev["url"].isin(set(train["url"])),
              "normalised_url": ev["key"].isin(set(train["key"])),
              "host": ev["host"].isin(set(train["host"])),
              "registered_domain": ev["domain"].isin(set(train["domain"]))}
    return {k: {"rows": int(m.sum()), "share": round(float(m.mean()), 3),
                "phish_rows": int((m & (ev.label == 1)).sum())} for k, m in levels.items()}


def audit() -> dict:
    raw = load("train")
    cleaned, log = clean_train(raw)
    test, external = load("test"), load("external_test")
    issues = {
        "exact_duplicate_rows": int(raw.duplicated(["url", "label"]).sum()),
        "urls_with_both_labels": int((raw.groupby("url")["label"].nunique() > 1).sum()),
        "normalised_duplicate_rows": int(raw["key"].duplicated().sum()),
        "html_entity_rows": int(raw["url"].str.contains(HTML_ENTITY).sum()),
        "html_entity_rows_phishing": int(raw.loc[raw["url"].str.contains(HTML_ENTITY), "label"].sum()),
        "non_ascii_rows": int(raw["url"].str.contains(r"[^\x00-\x7f]").sum()),
        "uppercase_rows": int((raw["url"] != raw["url"].str.lower()).sum()),
        "over_500_chars": int((raw["url"].str.len() > 500).sum()),
        "hosts_with_both_labels": int((raw.groupby("host")["label"].nunique() > 1).sum()),
        "top_domains": {d: {"rows": int(n), "phish": int(raw.loc[raw.domain == d, "label"].sum())}
                        for d, n in raw["domain"].value_counts().head(8).items()},
    }
    # This isolates the external set's bare-homepage construction.
    external_legit_bare = external[external.label == 0]["url"].str.fullmatch(BARE_HOMEPAGE)
    result = {
        "train_issues": issues, "cleaning": log,
        "characteristics": {"train": characteristics(cleaned), "test": characteristics(test),
                            "external": characteristics(external)},
        "overlap_with_train": {"test": overlap(test, raw), "external": overlap(external, raw),
                               "test_vs_external": overlap(external, test)},
        "external_legit_bare_homepage": round(float(external_legit_bare.mean()), 3),
        "train_bare_homepage": {
            lbl: int(cleaned[cleaned.label == lbl]["url"].str.fullmatch(BARE_HOMEPAGE).sum())
            for lbl in (0, 1)},
    }
    (ROOT / "reports" / "audit.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
