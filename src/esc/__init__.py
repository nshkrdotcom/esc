"""ESC: Epistemic State Compilation.

Separating persistent epistemic state from LLM cognition.
"""

from esc.benchmark.corpus import CorpusDocument, DocumentCorpus
from esc.benchmark.generator import generate_epidag_task, generate_task_suite
from esc.benchmark.tasks import EpiDAGNode, EpiDAGTask
from esc.core.kernel import StateKernel
from esc.core.signatures import ContinuousSolve, ResolveStep
from esc.core.types import AuditEntry, EpistemicLevel, Evidence, Fact, StepResult, StepSpec, level_ok
from esc.core.witness import WitnessResult, evaluate_witness
from esc.eval.decay import HorizonDecayResult, check_hypothesis_h1, fit_horizon_decay
from esc.eval.gepa_metric import epistemic_gepa_metric
from esc.eval.metrics import EvalVector, compute_eval_vector, compute_pass_at_k, compute_pass_pow_k
from esc.runner import ExperimentResult, run_experiment_1
from esc.systems.base import BaseSystem, SystemResult
from esc.systems.system_a import SystemAContinuous
from esc.systems.system_b import SystemBSearchHeavy
from esc.systems.system_c import SystemCIsolated
from esc.systems.system_d import SystemDGEPA
from esc.workers.continuous import ContinuousWorker
from esc.workers.epistemic import EpistemicWorker
from esc.workers.mock import MockContinuousWorker, MockEpistemicWorker

__version__ = "0.1.0"

__all__ = [
    "AuditEntry",
    "BaseSystem",
    "ContinuousSolve",
    "ContinuousWorker",
    "CorpusDocument",
    "DocumentCorpus",
    "EpiDAGNode",
    "EpiDAGTask",
    "EpistemicLevel",
    "EpistemicWorker",
    "EvalVector",
    "Evidence",
    "ExperimentResult",
    "Fact",
    "HorizonDecayResult",
    "MockContinuousWorker",
    "MockEpistemicWorker",
    "ResolveStep",
    "StateKernel",
    "StepResult",
    "StepSpec",
    "SystemAContinuous",
    "SystemBSearchHeavy",
    "SystemCIsolated",
    "SystemDGEPA",
    "SystemResult",
    "WitnessResult",
    "compute_eval_vector",
    "compute_pass_at_k",
    "compute_pass_pow_k",
    "epistemic_gepa_metric",
    "evaluate_witness",
    "fit_horizon_decay",
    "generate_epidag_task",
    "generate_task_suite",
    "level_ok",
    "run_experiment_1",
    "check_hypothesis_h1",
]
