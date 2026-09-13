"""Task and node models for EpiDAG benchmark."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

from esc.benchmark.corpus import DocumentCorpus
from esc.core.types import Evidence, StepSpec


class EpiDAGNode(BaseModel):
    """A single node in the hidden dependency DAG."""

    node_id: str
    step_spec: StepSpec
    true_value: str
    true_evidence: list[Evidence] = Field(default_factory=list)
    witness_type: Literal["type_1", "type_2", "type_3"] = "type_2"
    injected_error_value: str | None = None
    level_depth: int = 0


class EpiDAGTask(BaseModel):
    """Complete EpiDAG evaluation task instance."""

    task_id: str
    depth: int
    question: str
    nodes: list[EpiDAGNode]
    final_node_id: str
    ground_truth_map: dict[str, str]
    injected_error_node_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Corpus dictionary doc_id -> text content
    corpus_data: dict[str, str] = Field(default_factory=dict)

    def get_corpus(self) -> DocumentCorpus:
        from esc.benchmark.corpus import CorpusDocument, DocumentCorpus

        docs = [
            CorpusDocument(doc_id=k, title=f"Doc {k}", content=v)
            for k, v in self.corpus_data.items()
        ]
        return DocumentCorpus(docs)

    def node_by_id(self, node_id: str) -> EpiDAGNode | None:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def target_answer(self) -> str:
        return self.ground_truth_map.get(self.final_node_id, "")

    def public_description(self) -> str:
        """Expose identical task instructions to A/B and C, without evaluator labels."""
        import json
        return self.question + "\nReturn the final step value only as final_answer.\nPublic step specifications:\n" + json.dumps(
            [node.step_spec.model_dump() for node in self.nodes]
        )

    @property
    def dependency_depth(self) -> int:
        levels: dict[str, int] = {}
        for node in self.nodes:
            step = node.step_spec
            levels[step.expected_key or step.step_id] = 1 + max(
                (levels[key] for key in step.requires), default=0
            )
        return levels[self.final_node_id]
