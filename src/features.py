"""Feature engineering utilities for the fraud-drift-detection project."""

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def add_hour(df):
    """Add an Hour column derived from the raw Time column (seconds since first transaction)."""
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600).astype(int)
    return df
