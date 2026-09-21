import pandas as pd

from src.drift import inject_adversarial_evasion_shift


def _sample_df():
    return pd.DataFrame({
        "Hour": [30, 30, 40, 40],
        "Class": [1, 0, 1, 0],
        "f1": [10.0, 10.0, 10.0, 10.0],
        "f2": [20.0, 20.0, 20.0, 20.0],
    })


def _zero_centroid():
    return pd.Series({"f1": 0.0, "f2": 0.0})


def test_rows_before_shift_hour_are_unchanged():
    df = _sample_df()
    result = inject_adversarial_evasion_shift(df, ["f1", "f2"], shift_hour=36, legit_centroid=_zero_centroid(), alpha=1.0)
    before = result[result["Hour"] < 36]
    assert (before["f1"] == 10.0).all()
    assert (before["f2"] == 20.0).all()


def test_legit_rows_are_never_shifted_even_after_shift_hour():
    df = _sample_df()
    result = inject_adversarial_evasion_shift(df, ["f1", "f2"], shift_hour=36, legit_centroid=_zero_centroid(), alpha=1.0)
    legit_after = result[(result["Hour"] >= 36) & (result["Class"] == 0)]
    assert (legit_after["f1"] == 10.0).all()
    assert (legit_after["f2"] == 20.0).all()


def test_fraud_rows_after_shift_hour_are_fully_replaced_by_centroid_when_alpha_is_1():
    df = _sample_df()
    result = inject_adversarial_evasion_shift(df, ["f1", "f2"], shift_hour=36, legit_centroid=_zero_centroid(), alpha=1.0)
    fraud_after = result[(result["Hour"] >= 36) & (result["Class"] == 1)]
    assert (fraud_after["f1"] == 0.0).all()
    assert (fraud_after["f2"] == 0.0).all()


def test_does_not_mutate_input_dataframe():
    df = _sample_df()
    inject_adversarial_evasion_shift(df, ["f1", "f2"], shift_hour=36, legit_centroid=_zero_centroid(), alpha=1.0)
    assert (df["f1"] == 10.0).all()
