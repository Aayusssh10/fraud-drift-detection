"""Drift detector logic and controlled drift simulation for the fraud-drift-detection project."""
import numpy as np

# Parameters from Phase 3 (03_drift_analysis.ipynb): starting at SHIFT_HOUR, fraud rows are
# blended ALPHA of the way toward the legitimate-transaction centroid, simulating fraudsters
# adapting to evade detection. Deliberately injected for demonstration, not observed drift.
SHIFT_HOUR = 36
ALPHA = 0.7


def inject_adversarial_evasion_shift(df, feature_cols, shift_hour, legit_centroid, alpha=0.7,
                                      hour_col="Hour", class_col="Class"):
    """Simulate fraudsters adapting to evade detection.

    For every fraud row at or after `shift_hour`, blend its feature vector toward the
    legitimate-transaction centroid by `alpha` (0 = no change, 1 = fully replaced by the
    centroid). This is a deliberately injected, clearly-labeled synthetic shift, not
    something observed in the real data. Returns a copy; does not mutate the input.
    """
    df = df.copy()
    mask = (df[hour_col] >= shift_hour) & (df[class_col] == 1)
    original = df.loc[mask, feature_cols].to_numpy()
    centroid = legit_centroid[feature_cols].to_numpy()
    df.loc[mask, feature_cols] = (1 - alpha) * original + alpha * centroid
    return df
