"""Fit the baseline and the selected advanced model on all cleaned training rows, pick
thresholds from the grouped-CV out-of-fold scores, export ONNX. Test files are not read."""
import json

import numpy as np
import pandas as pd

from data import ROOT, clean_train, load
from explain import save_reference
from metrics import deployed_precision, threshold_for_fpr
from model import MODELS, Scorer, export, make
from preprocess import FEATURES

TARGET_FPR = 0.01
BASELINE_ID = 0  # experiments.py lists the baseline first.


def threshold_table(y, p):
    """Validation recall, FPR and implied precision at 1-in-1,000 for several FPR targets."""
    rows = []
    for target in (0.05, 0.02, 0.01, 0.005, 0.001):
        t = threshold_for_fpr(y, p, target)  # Each row shows the recall cost of a tighter queue.
        tpr, fpr = (p[y == 1] >= t).mean(), (p[y == 0] >= t).mean()
        rows.append({"target_fpr": target, "threshold": round(t, 4), "val_recall": round(tpr, 4),
                     "val_fpr": round(fpr, 5), "val_false_positives": int((p[y == 0] >= t).sum()),
                     "deployed_precision": round(deployed_precision(tpr, fpr), 4)})
    return rows


def main():
    df, cleaning = clean_train(load("train"))
    save_reference(df)
    y = df.label.to_numpy()
    experiments = json.loads((ROOT / "reports" / "experiments.json").read_text())
    candidates = {r["id"]: r for r in experiments["candidates"]}
    oof = pd.read_csv(ROOT / "reports" / "oof_scores.csv")
    assert len(oof) == len(df)  # OOF rows must still line up with cleaned labels.
    xgboost_id = max((r for r in experiments["candidates"] if r["model"] == "xgboost"),
                     key=lambda r: r["recall_at_1pct_fpr"])["id"]
    MODELS.mkdir(exist_ok=True)
    summary = {"cleaning": cleaning, "target_val_fpr": TARGET_FPR}
    for name, components in (("baseline", [[BASELINE_ID, 1.0]]),
                             ("advanced", experiments["selected"]["components"]),
                             ("xgboost", [[xgboost_id, 1.0]])):
        # Out-of-fold negatives set the cutoff before either test set is read.
        threshold = threshold_for_fpr(y, oof[name].to_numpy(), TARGET_FPR)
        for path in MODELS.glob(f"{name}*.onnx"):
            path.unlink()
        fitted = []  # Refit on every labeled URL after using OOF scores for the cutoff.
        for i, weight in components:
            c = candidates[i]
            model = make(c["model"], c["params"]).fit(FEATURES[c["features"]](df.url), y)
            suffix = "" if len(components) == 1 else f"_{c['model']}"
            export(model, c["features"], weight, threshold, MODELS / f"{name}{suffix}.onnx")
            fitted.append((model, c["features"], weight))
        sample = df.url.sample(500, random_state=0).tolist()
        sklearn_p = sum(w * m.predict_proba(FEATURES[f](sample))[:, 1] for m, f, w in fitted)
        # This compares the exported graphs with their Python models, not held-out quality.
        summary[name] = {"components": [[candidates[i]["model"], candidates[i]["params"], w] for i, w in components],
                         "threshold": round(threshold, 4),
                         "onnx_vs_python_max_abs_diff": float(np.abs(Scorer(name).predict_proba(sample) - sklearn_p).max()),
                         "validation_thresholds": threshold_table(y, oof[name].to_numpy())}
    (ROOT / "reports" / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
