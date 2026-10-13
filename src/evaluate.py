"""Score both models on test.csv and external_test.csv at the validation threshold.
Writes reports/evaluation.json and predictions.csv (advanced model, one row per test URL)."""
import json

import numpy as np

from data import BARE_HOMEPAGE, ROOT, clean_train, load
from metrics import at_threshold, summary
from model import Scorer


def main():
    train, _ = clean_train(load("train"))  # Seen-domain checks use the same training cleanup.
    sets = {"test": load("test"), "external": load("external_test")}
    result = {}
    # This is the only stage that scores the held-out files.
    for name in ("baseline", "advanced", "xgboost"):
        scorer = Scorer(name)
        result[name] = {"threshold": round(scorer.threshold, 4)}
        for set_name, df in sets.items():
            y, p = df.label.to_numpy(), scorer.predict_proba(df.url)
            report = summary(y, p, scorer.threshold, df.domain.to_numpy())
            seen = df.domain.isin(set(train.domain)).to_numpy()  # Split whole sites, not URL rows.
            report["domain_in_train"] = at_threshold(y[seen], p[seen], scorer.threshold) | {"n": int(seen.sum())}
            # Unseen domains show how well the score travels to genuinely new sites.
            report["domain_not_in_train"] = summary(y[~seen], p[~seen], scorer.threshold, df.domain[~seen].to_numpy())
            # Bare homepages are a useful check against easy path-based phishing cues.
            bare = df.url.str.fullmatch(BARE_HOMEPAGE).to_numpy()
            phish = y == 1
            report["recall_phish_bare_domain"] = {"n": int((bare & phish).sum()),
                                                  "recall": round(float((p[bare & phish] >= scorer.threshold).mean()), 4)}
            report["recall_phish_with_path"] = {"n": int((~bare & phish).sum()),
                                                "recall": round(float((p[~bare & phish] >= scorer.threshold).mean()), 4)}
            result[name][set_name] = report
            if name == "advanced" and set_name == "test":
                df.assign(probability=np.round(p, 6))[["url", "probability"]].to_csv(
                    ROOT / "predictions.csv", index=False)
    (ROOT / "reports" / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
