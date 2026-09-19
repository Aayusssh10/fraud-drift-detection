"""Cost matrix and threshold optimization for the fraud-drift-detection project."""
import numpy as np
import pandas as pd

# Illustrative cost matrix from Phase 4 (04_cost_threshold_and_shap.ipynb) — not derived
# from real business data. Ratio (100:1) drives the threshold choice more than the
# absolute units, which are deliberately left abstract since the dataset itself doesn't
# specify a currency.
COST_FN = 100  # missed fraud
COST_FP = 1    # false alarm

# Cost-minimizing threshold found by sweeping COST_FN/COST_FP over the Phase 2 static
# model's day-2 scores in Phase 4. Re-derive with find_optimal_threshold if the model,
# data, or cost ratio changes.
THRESHOLD = 0.19


def expected_cost(y_true, scores, threshold, cost_fn, cost_fp):
    """Total cost at a given threshold: missed frauds cost `cost_fn` each,
    false alarms cost `cost_fp` each. Correctly-handled transactions cost nothing
    (a simplification — real systems may also cost some amount per true positive
    for investigation, but the brief's cost matrix only specifies these two).
    """
    preds = (scores >= threshold).astype(int)
    fn = int(((preds == 0) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    return fn * cost_fn + fp * cost_fp


def sweep_thresholds(y_true, scores, cost_fn, cost_fp, thresholds=None):
    """Compute expected cost, and raw FN/FP counts, across a range of thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        fn = int(((preds == 0) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tp = int(((preds == 1) & (y_true == 1)).sum())
        cost = fn * cost_fn + fp * cost_fp
        rows.append({"threshold": t, "cost": cost, "fn": fn, "fp": fp, "tp": tp})

    return pd.DataFrame(rows)


def find_optimal_threshold(y_true, scores, cost_fn, cost_fp, thresholds=None):
    """Return the threshold (and its row) that minimizes expected cost."""
    sweep = sweep_thresholds(y_true, scores, cost_fn, cost_fp, thresholds)
    best = sweep.loc[sweep["cost"].idxmin()]
    return best["threshold"], sweep
