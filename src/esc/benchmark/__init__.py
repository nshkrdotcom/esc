"""EpiDAG benchmark module for epistemic composability and horizon scaling."""

from esc.benchmark.corpus import CorpusDocument, DocumentCorpus
from esc.benchmark.generator import generate_epidag_task, generate_synthetic_corpus, generate_task_suite
from esc.benchmark.tasks import EpiDAGNode, EpiDAGTask

__all__ = [
    "CorpusDocument",
    "DocumentCorpus",
    "EpiDAGNode",
    "EpiDAGTask",
    "generate_epidag_task",
    "generate_synthetic_corpus",
    "generate_task_suite",
]
