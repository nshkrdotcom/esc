"""Evaluation metrics, horizon decay analysis, and GEPA feedback metric."""

from esc.eval.decay import HorizonDecayResult, check_hypothesis_h1, fit_horizon_decay
from esc.eval.gepa_metric import epistemic_gepa_metric
from esc.eval.metrics import EvalVector, compute_eval_vector, compute_pass_at_k, compute_pass_pow_k

__all__ = [
    "EvalVector",
    "HorizonDecayResult",
    "check_hypothesis_h1",
    "compute_eval_vector",
    "compute_pass_at_k",
    "compute_pass_pow_k",
    "epistemic_gepa_metric",
    "fit_horizon_decay",
]
