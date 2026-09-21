import numpy as np

from src.cost import expected_cost, find_optimal_threshold, sweep_thresholds


def test_expected_cost_counts_fn_and_fp_correctly():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.9, 0.1])  # one fraud caught, one fraud missed, one false alarm, one correct
    cost = expected_cost(y_true, scores, threshold=0.5, cost_fn=100, cost_fp=1)
    assert cost == 101  # 1 missed fraud (100) + 1 false alarm (1)


def test_find_optimal_threshold_prefers_catching_fraud_when_fn_is_expensive():
    # only a threshold <= 0.3 catches the one fraud case (score 0.31)
    y_true = np.array([1, 0, 0, 0, 0])
    scores = np.array([0.31, 0.05, 0.05, 0.05, 0.05])
    best_threshold, _ = find_optimal_threshold(
        y_true, scores, cost_fn=1000, cost_fp=1, thresholds=np.array([0.1, 0.3, 0.5, 0.9])
    )
    assert best_threshold <= 0.3


def test_sweep_thresholds_cost_rises_when_threshold_moves_above_fraud_scores():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    sweep = sweep_thresholds(y_true, scores, cost_fn=100, cost_fp=1, thresholds=np.array([0.5, 0.95]))
    cost_at_050 = sweep.loc[sweep["threshold"] == 0.5, "cost"].iloc[0]
    cost_at_095 = sweep.loc[sweep["threshold"] == 0.95, "cost"].iloc[0]
    assert cost_at_095 > cost_at_050  # threshold 0.95 misses both frauds; 0.5 catches both
