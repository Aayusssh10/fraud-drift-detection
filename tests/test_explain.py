from src.explain import top_contributions


class FakeExplanationRow:
    """Stands in for one row of a shap.Explanation without needing to train a model."""

    def __init__(self, values, data):
        self.values = values
        self.data = data


def test_top_contributions_returns_n_items_sorted_by_absolute_value():
    row = FakeExplanationRow(values=[0.1, -5.0, 2.0, -0.5], data=[1, 2, 3, 4])
    result = top_contributions(row, feature_cols=["a", "b", "c", "d"], n=2)
    assert len(result) == 2
    assert result[0][0] == "b"  # |-5.0| is the largest contribution
    assert result[1][0] == "c"  # |2.0| is next largest


def test_top_contributions_returns_feature_raw_value_and_shap_value_tuples():
    row = FakeExplanationRow(values=[3.0], data=[42])
    result = top_contributions(row, feature_cols=["only_feature"], n=5)
    assert result == [("only_feature", 3.0, 42)]
