import pandas as pd

from src.features import FEATURE_COLS, add_hour


def test_add_hour_converts_seconds_to_hour_buckets():
    df = pd.DataFrame({"Time": [0, 3599, 3600, 7200]})
    result = add_hour(df)
    assert result["Hour"].tolist() == [0, 0, 1, 2]


def test_add_hour_does_not_mutate_input():
    df = pd.DataFrame({"Time": [0, 3600]})
    add_hour(df)
    assert "Hour" not in df.columns


def test_feature_cols_has_28_v_columns_plus_amount():
    assert len(FEATURE_COLS) == 29
    assert FEATURE_COLS[:5] == ["V1", "V2", "V3", "V4", "V5"]
    assert FEATURE_COLS[-1] == "Amount"
