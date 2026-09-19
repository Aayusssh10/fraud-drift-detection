"""SHAP explanation wrappers shared by notebooks and the Streamlit app."""
import shap


def build_explainer(model):
    return shap.TreeExplainer(model)


def explain(explainer, X):
    """Return a shap.Explanation for the given rows."""
    return explainer(X)


def top_contributions(explanation_row, feature_cols, n=5):
    """Return the top-n (feature, raw_value, shap_contribution) tuples by |contribution|,
    for one row of a shap.Explanation (e.g. explanation[i])."""
    pairs = list(zip(feature_cols, explanation_row.values, explanation_row.data))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:n]
