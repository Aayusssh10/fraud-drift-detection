# Adaptive Fraud Detection with Concept Drift + Cost-Aware Explainable Alerts

A fraud detection system that keeps working as fraud patterns change over time, tunes its
alert threshold to real business cost instead of accuracy, and explains every flag it raises.

## Why this project

Most "fraud detection" portfolio projects generate synthetic transaction data and run an
Isolation Forest + IQR check once, on a random train/test split. That doesn't demonstrate
anything hard, and it hides the two problems that actually make fraud detection difficult in
practice: extreme class imbalance (accuracy is meaningless) and concept drift (a model trained
once decays as fraud patterns change). This project uses real data, a time-based evaluation
split, cost-based thresholding, and an explicit drift-detection + retraining comparison instead.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle, ULB/Worldline) — ~285,299 European card transactions over 2 days, 492 fraudulent
(~0.172%). Features `V1`-`V28` are PCA-anonymized; `Time` (seconds since first transaction) and
`Amount` are raw.

Not committed to this repo (see `.gitignore`) — fetch it yourself:

```bash
python src/data.py --download
```

This requires a Kaggle API token at `~/.kaggle/kaggle.json` (see
[Kaggle API docs](https://www.kaggle.com/docs/api)) and that you've accepted the dataset's
terms on its Kaggle page at least once.

## Project status

All phases complete:

- [x] Phase 0 — Setup
- [x] Phase 1 — EDA
- [x] Phase 2 — Baseline models (time-based split)
- [x] Phase 3 — Concept drift analysis + drift-triggered retraining
- [x] Phase 4 — Cost-sensitive thresholding + SHAP explanations
- [x] Phase 5 — Streamlit demo dashboard
- [x] Phase 6 — Final writeup

See [reports/final_report.md](reports/final_report.md) for the full writeup, with charts, findings, and limitations.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the sanity tests for `src/` with:

```bash
pytest tests/
```

## Structure

- `src/` — reusable pipeline code (data loading, features, models, drift detection, cost
  thresholding, SHAP wrappers), imported by both notebooks and the Streamlit app.
- `notebooks/` — phase-by-phase analysis and experimentation.
- `app/streamlit_app.py` — live demo dashboard.
- `reports/final_report.md` — final writeup with charts, findings, and limitations.
- `tests/` — sanity tests for `src/`.
