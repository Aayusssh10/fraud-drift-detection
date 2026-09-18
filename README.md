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

Work in progress, built phase by phase:

- [x] Phase 0 — Setup
- [ ] Phase 1 — EDA
- [ ] Phase 2 — Baseline models (time-based split)
- [ ] Phase 3 — Concept drift analysis + drift-triggered retraining
- [ ] Phase 4 — Cost-sensitive thresholding + SHAP explanations
- [ ] Phase 5 — Streamlit demo dashboard
- [ ] Phase 6 — Final writeup

See [reports/final_report.md](reports/final_report.md) for the full writeup once later phases land.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

- `src/` — reusable pipeline code (data loading, features, models, drift detection, cost
  thresholding, SHAP wrappers), imported by both notebooks and the Streamlit app.
- `notebooks/` — phase-by-phase analysis and experimentation.
- `app/streamlit_app.py` — live demo dashboard.
- `reports/final_report.md` — final writeup with charts, findings, and limitations.
- `tests/` — sanity tests for `src/`.
