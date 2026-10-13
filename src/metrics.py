"""Threshold, interval and deployed-precision helpers."""
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve

PREVALENCE = 0.001


def recall_at_fpr(y, p, max_fpr):
    # Use the best point on the ROC curve that stays under the review limit.
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= max_fpr].max())


def threshold_for_fpr(y, p, target_fpr):
    """Lowest threshold whose false-positive rate on (validation) negatives is <= target."""
    # Pick the cutoff from OOF negatives, never from held-out rows.
    neg = np.sort(p[y == 0])[::-1]
    return float(np.nextafter(neg[int(target_fpr * len(neg))], 1.0))  # Step above the boundary to exclude ties.


def clopper_pearson(k, n):
    # Exact binomial bounds behave better than normal intervals for rare errors.
    low = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    high = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(low), float(high)


def deployed_precision(tpr, fpr, prevalence=PREVALENCE):
    # Bayes rule: expected true alerts divided by all alerts at deployment prevalence.
    return tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence))


def domain_bootstrap_auc(y, p, groups, n=500, seed=0):
    """95% AUC interval resampling whole registered domains, not rows."""
    rng = np.random.default_rng(seed)
    _, idx = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(idx == g) for g in range(idx.max() + 1)]
    aucs = []
    for _ in range(n):  # Resample sites because URLs from one domain are not independent.
        rows = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        aucs.append(roc_auc_score(y[rows], p[rows]))
    return [round(float(v), 4) for v in np.percentile(aucs, [2.5, 97.5])]


def at_threshold(y, p, threshold):
    """Confusion counts, Clopper-Pearson rate intervals and precision at 1-in-1,000."""
    hit = p >= threshold
    tp, fn = int((hit & (y == 1)).sum()), int((~hit & (y == 1)).sum())
    fp, tn = int((hit & (y == 0)).sum()), int((~hit & (y == 0)).sum())
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)  # Measure the chosen cutoff on this set.
    tpr_ci, fpr_ci = clopper_pearson(tp, tp + fn), clopper_pearson(fp, fp + tn)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": round(tpr, 4), "recall_ci": [round(v, 4) for v in tpr_ci],
        "fpr": round(fpr, 5), "fpr_ci": [round(v, 5) for v in fpr_ci],
        "precision": round(tp / max(tp + fp, 1), 4),
        "deployed_precision": round(deployed_precision(tpr, fpr), 4) if tp + fp else None,
        "deployed_precision_range": [round(deployed_precision(tpr_ci[0], fpr_ci[1]), 4),
                                     round(deployed_precision(tpr_ci[1], max(fpr_ci[0], 1e-9)), 4)],
    }


def summary(y, p, threshold, groups):
    return {"n": int(len(y)), "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "roc_auc_ci": domain_bootstrap_auc(y, p, groups),
            "recall_at_1pct_fpr": round(recall_at_fpr(y, p, 0.01), 4),
            "mean_p_legit": round(float(p[y == 0].mean()), 3),
            "mean_p_phish": round(float(p[y == 1].mean()), 3),
            "at_threshold": at_threshold(y, p, threshold)}
