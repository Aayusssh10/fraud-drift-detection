"""Baseline model wrappers shared by notebooks and the Streamlit app."""
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier


def fit_isolation_forest(X_train, random_state=42):
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state, n_jobs=-1)
    model.fit(X_train)
    return model


def isolation_forest_scores(model, X):
    """Higher score = more suspicious (fraud-like)."""
    return -model.score_samples(X)


def fit_xgboost(X_train, y_train, random_state=42):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def xgboost_scores(model, X):
    return model.predict_proba(X)[:, 1]
