"""Small auto-ML sweep: model families x hyper-parameters (+ a blend), compared in
5-fold cross-validation grouped by registered domain. Writes reports/experiments.json."""
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from data import ROOT, clean_train, load
from metrics import recall_at_fpr
from model import make
from preprocess import FEATURES

BASELINE = ("baseline", "logreg", {"C": 4.0})
CANDIDATES = [BASELINE]
CANDIDATES += [("advanced", "logreg", {"C": c}) for c in (1.0, 4.0, 16.0)]
CANDIDATES += [("advanced", "lightgbm", {"n_estimators": n, "learning_rate": 0.05, "num_leaves": leaves,
                                         "colsample_bytree": 0.3, "min_child_samples": 10})
               for n, leaves in itertools.product((300, 800), (15, 63))]
CANDIDATES += [("advanced", "xgboost", {"n_estimators": n, "learning_rate": 0.1, "max_depth": depth,
                                       "colsample_bytree": 0.3, "tree_method": "hist"})
               for n, depth in itertools.product((300, 600), (4, 8))]
STEP = 0.25


def oof(df, features, kind, params, grouped=True):
    """Out-of-fold phishing probabilities; grouped folds keep each registered domain on one side."""
    y, X = df.label.to_numpy(), FEATURES[features](df.url)
    # Each URL gets a score from a model that never trained on its domain.
    folds = (StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, df.domain) if grouped
             else StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    p = np.zeros(len(y))
    for train, val in folds:
        p[val] = make(kind, params).fit(X[train], y[train]).predict_proba(X[val])[:, 1]  # Fill held-out rows only.
    return p


def score(y, p):
    # Report both operating points so a tight review queue is visible too.
    return {"roc_auc": round(roc_auc_score(y, p), 4),
            "recall_at_1pct_fpr": round(recall_at_fpr(y, p, 0.01), 4),
            "recall_at_0.1pct_fpr": round(recall_at_fpr(y, p, 0.001), 4)}


def main():
    raw = load("train")
    df, _ = clean_train(raw)
    y = df.label.to_numpy()
    rows, scores = [], {}
    for i, (features, kind, params) in enumerate(CANDIDATES):
        start = time.time()
        scores[i] = oof(df, features, kind, params)
        rows.append({"id": i, "features": features, "model": kind, "params": params,
                     **score(y, scores[i]), "seconds": round(time.time() - start, 1)})
        print(rows[-1], flush=True)
    adv = [r for r in rows if r["features"] == "advanced"]
    families = ("logreg", "lightgbm", "xgboost")
    # Fix the best single model in each family before comparing combinations.
    family_best = {
        family: max((r for r in adv if r["model"] == family),
                    key=lambda r: r["recall_at_1pct_fpr"])
        for family in families
    }
    family_scores = {family: scores[row["id"]]
                     for family, row in family_best.items()}
    correlations = pd.DataFrame(family_scores).corr(method="spearman").round(3).to_dict()  # Compare OOF rankings.
    # Keep the best model in each family fixed and only vary blend weights.
    blend_grid = tuple(i * STEP for i in range(5))
    for weights in itertools.product(blend_grid, repeat=3):
        if sum(weights) != 1 or sum(weight > 0 for weight in weights) < 2:  # Skip singles and invalid mixes.
            continue
        components = [[family_best[family]["id"], weight]
                      for family, weight in zip(families, weights) if weight]
        p = sum(weight * scores[i] for i, weight in components)
        rows.append({"id": len(rows), "features": "advanced", "model": "blend",
                     "params": {}, "components": components, **score(y, p)})
    for row in rows:
        row.setdefault("components", [[row["id"], 1.0]])
    best = max(rows[1:], key=lambda r: (r["recall_at_1pct_fpr"], r["roc_auc"]))  # Recall first, AUC breaks ties.
    checks = {  # Show how random folds and skipped cleaning change the baseline score.
        "baseline_random_folds": score(y, oof(df, *BASELINE, grouped=False)),
        "baseline_uncleaned_train": score(raw.label.to_numpy(), oof(raw, *BASELINE)),
    }
    blend = sum(w * scores[i] for i, w in best["components"])  # Rebuild selected OOF scores for thresholding.
    oof_columns = {"baseline": scores[0], "advanced": blend}
    oof_columns.update({family: values for family, values in family_scores.items()})
    pd.DataFrame(oof_columns).round(6).to_csv(  # rows follow cleaned train
        ROOT / "reports" / "oof_scores.csv", index=False)
    result = {"selection_rule": "highest grouped-CV recall at 1% FPR among advanced candidates and blends, AUC tie-break",
              "candidates": rows, "selected": best, "checks": checks,
              "family_best_correlation": correlations}
    (ROOT / "reports" / "experiments.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"selected": best, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
