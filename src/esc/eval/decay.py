"""Horizon scaling decay analysis: Fitting log P(success) = alpha - beta * d."""

from __future__ import annotations

from typing import Any
import numpy as np
from pydantic import BaseModel
from scipy import stats


class HorizonDecayResult(BaseModel):
    """Fitted horizon decay model for a given architecture."""

    system_name: str
    alpha: float
    beta: float
    r_squared: float
    std_err: float
    accuracies_by_depth: dict[int, float]
    fitted_probs: dict[int, float]


def fit_horizon_decay(
    accuracies_by_depth: dict[int, float],
    system_name: str,
) -> HorizonDecayResult:
    """Fit exponential horizon decay log P(success) = alpha - beta * depth.

    Lower beta indicates superior horizon scaling (slower degradation with depth).
    """
    depths = sorted(accuracies_by_depth.keys())
    probs = np.array([max(1e-4, min(1.0 - 1e-4, accuracies_by_depth[d])) for d in depths])
    log_probs = np.log(probs)
    x = np.array(depths, dtype=float)

    if len(depths) < 2:
        return HorizonDecayResult(
            system_name=system_name,
            alpha=float(log_probs[0]) if len(log_probs) > 0 else 0.0,
            beta=0.0,
            r_squared=1.0,
            std_err=0.0,
            accuracies_by_depth=accuracies_by_depth,
            fitted_probs={d: accuracies_by_depth[d] for d in depths},
        )

    if np.all(log_probs == log_probs[0]):
        slope, intercept, r_value, std_err = 0.0, float(log_probs[0]), 0.0, 0.0
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_probs)

    beta = -float(slope)
    alpha = float(intercept)
    r_sq = float(r_value**2)

    fitted = {int(d): float(np.exp(alpha - beta * d)) for d in depths}

    return HorizonDecayResult(
        system_name=system_name,
        alpha=alpha,
        beta=beta,
        r_squared=r_sq,
        std_err=float(std_err),
        accuracies_by_depth=accuracies_by_depth,
        fitted_probs=fitted,
    )


def check_hypothesis_h1(
    decay_c: HorizonDecayResult,
    decay_a: HorizonDecayResult,
) -> dict[str, Any]:
    """Legacy helper name; compare descriptive slopes without testing H1.

    These fits contain neither matched-cost evidence nor paired uncertainty.
    A point-estimate ordering cannot establish support for the hypothesis.
    """
    beta_c = decay_c.beta
    beta_a = decay_a.beta
    comparable = (len(decay_c.accuracies_by_depth) >= 2
                  and decay_c.accuracies_by_depth.keys() == decay_a.accuracies_by_depth.keys())
    ratio = beta_a / beta_c if comparable and beta_a > 0 and beta_c > 0 else None

    return {
        "h1_supported": None,
        "descriptive_beta_c_lower": beta_c < beta_a if comparable else None,
        "beta_C": beta_c,
        "beta_A": beta_a,
        "beta_ratio_A_over_C": ratio,
        "interpretation": "Not tested: descriptive slopes do not establish matched compute or statistical support.",
    }
