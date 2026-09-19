"""Live demo: replays day-2 transactions, flags fraud, explains alerts, and reacts to drift.

Uses the same injected adversarial-evasion drift scenario from Phase 3 (03_drift_analysis.ipynb)
so the drift alarm and auto-retraining have something real to react to during a demo. This is a
simulated scenario for demonstration purposes, not organic drift observed in the real data.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from river import drift as river_drift

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.cost import COST_FN, COST_FP, THRESHOLD
from src.data import load_raw
from src.drift import SHIFT_HOUR, ALPHA, inject_adversarial_evasion_shift
from src.explain import build_explainer, explain, top_contributions
from src.features import FEATURE_COLS, add_hour
from src.models import fit_xgboost, xgboost_scores

BATCH_SIZE = 300
MAX_ALERTS_SHOWN = 15

st.set_page_config(page_title="Fraud Drift Detection - Live Demo", layout="wide")


@st.cache_resource
def load_pipeline():
    df = add_hour(load_raw())
    train_df = df[df["Hour"] < 24].reset_index(drop=True)
    test_df = df[df["Hour"] >= 24].sort_values("Time").reset_index(drop=True)

    X_train, y_train = train_df[FEATURE_COLS], train_df["Class"]
    legit_centroid = X_train[y_train == 0].mean()

    stream_df = inject_adversarial_evasion_shift(
        test_df, FEATURE_COLS, shift_hour=SHIFT_HOUR, legit_centroid=legit_centroid, alpha=ALPHA
    )

    base_model = fit_xgboost(X_train, y_train)
    explainer = build_explainer(base_model)
    return train_df, stream_df, base_model, explainer


def cumulative_train_data(train_df, stream_df, up_to_idx):
    seen = stream_df.iloc[:up_to_idx]
    combined = pd.concat([train_df, seen], ignore_index=True)
    return combined[FEATURE_COLS], combined["Class"]


def init_state():
    train_df, stream_df, base_model, explainer = load_pipeline()
    st.session_state.train_df = train_df
    st.session_state.stream_df = stream_df
    st.session_state.model = base_model
    st.session_state.explainer = explainer
    st.session_state.position = 0
    st.session_state.adwin = river_drift.ADWIN()
    st.session_state.playing = False
    st.session_state.alerts = []
    st.session_state.retrain_log = []
    st.session_state.alarm_ticks_left = 0
    st.session_state.history = []
    st.session_state.tp = 0
    st.session_state.fp = 0
    st.session_state.fn = 0
    st.session_state.tn = 0


def process_tick():
    stream_df = st.session_state.stream_df
    train_df = st.session_state.train_df
    start = st.session_state.position
    end = min(start + BATCH_SIZE, len(stream_df))
    if start >= end:
        st.session_state.playing = False
        return

    batch = stream_df.iloc[start:end]
    X_batch = batch[FEATURE_COLS]
    scores = xgboost_scores(st.session_state.model, X_batch)

    fired = False
    for i, (_, row) in enumerate(batch.iterrows()):
        score = scores[i]
        flagged = score >= THRESHOLD
        is_fraud = row["Class"] == 1

        if is_fraud:
            st.session_state.tp += int(flagged)
            st.session_state.fn += int(not flagged)
            st.session_state.adwin.update(int(not flagged))
            if st.session_state.adwin.drift_detected:
                fired = True
        else:
            st.session_state.fp += int(flagged)
            st.session_state.tn += int(not flagged)

        if flagged:
            expl = explain(st.session_state.explainer, X_batch.iloc[[i]])
            top = top_contributions(expl[0], FEATURE_COLS, n=3)
            reason = ", ".join(f"{f} ({v:+.2f})" for f, v, _ in top)
            st.session_state.alerts.insert(0, {
                "hour": int(row["Hour"]),
                "amount": round(float(row["Amount"]), 2),
                "score": round(float(score), 3),
                "actual": "FRAUD" if is_fraud else "legit",
                "reason": reason,
            })
    st.session_state.alerts = st.session_state.alerts[:MAX_ALERTS_SHOWN]
    st.session_state.position = end

    cost = st.session_state.fn * COST_FN + st.session_state.fp * COST_FP
    recall = st.session_state.tp / max(st.session_state.tp + st.session_state.fn, 1)
    precision = st.session_state.tp / max(st.session_state.tp + st.session_state.fp, 1)
    st.session_state.history.append({"position": end, "recall": recall, "precision": precision, "cost": cost})

    if fired:
        st.session_state.alarm_ticks_left = 5
        Xc, yc = cumulative_train_data(train_df, stream_df, end)
        st.session_state.model = fit_xgboost(Xc, yc)
        st.session_state.explainer = build_explainer(st.session_state.model)
        st.session_state.adwin = river_drift.ADWIN()
        st.session_state.retrain_log.insert(
            0, f"Retrained at transaction {end} (hour {int(batch.iloc[-1]['Hour'])}) after drift detected"
        )
    elif st.session_state.alarm_ticks_left > 0:
        st.session_state.alarm_ticks_left -= 1


if "position" not in st.session_state:
    init_state()

st.title("Adaptive Fraud Detection — Live Demo")
st.caption(
    "Replays day-2 transactions with the Phase 3 injected adversarial-evasion drift (starting "
    f"hour {SHIFT_HOUR}), so the drift alarm and auto-retraining have something to react to. "
    "This is a simulated scenario for demonstration, not organic drift in the real data. "
    f"Alerts use the cost-minimizing threshold ({THRESHOLD}) from Phase 4."
)

with st.sidebar:
    st.header("Playback controls")
    speed_ms = st.slider("Delay between ticks (ms)", min_value=0, max_value=1000, value=200, step=50)
    st.caption(f"{BATCH_SIZE} transactions replayed per tick")
    col1, col2 = st.columns(2)
    if col1.button("Pause" if st.session_state.playing else "Play"):
        st.session_state.playing = not st.session_state.playing
        st.rerun()  # re-render immediately so the button label reflects the new state
    if col2.button("Reset"):
        init_state()
        st.rerun()

total = len(st.session_state.stream_df)
st.progress(
    st.session_state.position / total,
    text=f"{st.session_state.position}/{total} transactions replayed",
)

alarm_on = st.session_state.alarm_ticks_left > 0
cols = st.columns(5)
cols[0].metric("Transactions seen", st.session_state.position)
recall_so_far = st.session_state.tp / max(st.session_state.tp + st.session_state.fn, 1)
precision_so_far = st.session_state.tp / max(st.session_state.tp + st.session_state.fp, 1)
cols[1].metric("Recall so far", f"{recall_so_far:.1%}")
cols[2].metric("Precision so far", f"{precision_so_far:.1%}")
cols[3].metric("Expected cost so far", st.session_state.fn * COST_FN + st.session_state.fp * COST_FP)
cols[4].metric("Drift alarm", "ACTIVE" if alarm_on else "clear")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history).set_index("position")
    st.line_chart(hist_df[["recall", "precision"]])

st.subheader("Recent alerts")
if st.session_state.alerts:
    st.dataframe(pd.DataFrame(st.session_state.alerts), width="stretch")
else:
    st.write("No alerts yet — press Play to start the replay.")

with st.expander(f"Retrain log ({len(st.session_state.retrain_log)} events)"):
    if st.session_state.retrain_log:
        for entry in st.session_state.retrain_log:
            st.write("- " + entry)
    else:
        st.write("No retrains yet.")

if st.session_state.playing:
    process_tick()
    time.sleep(speed_ms / 1000)
    st.rerun()
