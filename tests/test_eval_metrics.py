import pytest
from esc.eval.decay import check_hypothesis_h1, fit_horizon_decay
from esc.eval.gepa_metric import epistemic_gepa_metric
from esc.eval.metrics import compute_eval_vector, compute_pass_at_k, compute_pass_pow_k
from esc.systems.base import SystemResult


def test_pass_at_k_and_pass_pow_k():
    runs = [
        [True, True, True],
        [True, False, True],
        [False, False, False],
    ]
    p_k = compute_pass_at_k(runs, k=2)
    p_pow = compute_pass_pow_k(runs, k=2)
    assert 0.0 <= p_k <= 1.0
    assert 0.0 <= p_pow <= 1.0
    assert p_pow <= p_k  # pass^k is strictly more conservative than pass@k


def test_horizon_decay_fit():
    accuracies = {2: 0.95, 4: 0.85, 8: 0.70, 16: 0.50}
    decay = fit_horizon_decay(accuracies, "TestSystem")
    assert decay.beta > 0.0  # Decays with depth
    assert decay.r_squared > 0.8

    accuracies_steep = {2: 0.90, 4: 0.50, 8: 0.20, 16: 0.05}
    decay_steep = fit_horizon_decay(accuracies_steep, "SteepSystem")

    h1 = check_hypothesis_h1(decay_c=decay, decay_a=decay_steep)
    assert h1["h1_supported"] is None
    assert h1["descriptive_beta_c_lower"] is True
    assert h1["beta_C"] < h1["beta_A"]


@pytest.mark.parametrize('accuracies', [{2: 0.5}, {2: 0.5, 4: 0.5}, {2: 0.2, 4: 0.8}])
def test_descriptive_comparison_does_not_invent_ratio(accuracies):
    decay = fit_horizon_decay(accuracies, 'C')
    comparison = check_hypothesis_h1(decay, fit_horizon_decay({2: 0.8, 4: 0.2}, 'A'))
    assert comparison['h1_supported'] is None
    assert comparison['beta_ratio_A_over_C'] is None


def test_gepa_metric_feedback():
    class DummyEx:
        target_answer = "42"
        token_budget = 1000

    class DummyPred:
        final_answer = "42"
        false_promotions = 0
        contract_violations = 0
        tokens_used = 200

    pred_res = epistemic_gepa_metric(DummyEx(), DummyPred())
    assert pred_res.score == 1.0

    class BadPred:
        final_answer = "99"
        false_promotions = 1
        contract_violations = 2
        tokens_used = 1500

    bad_res = epistemic_gepa_metric(DummyEx(), BadPred())
    assert bad_res.score == 0.0
    assert "incorrect claims crossed" in bad_res.feedback
    assert "contract violations occurred" in bad_res.feedback
