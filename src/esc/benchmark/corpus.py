"""Corpus representation and projection for EpiDAG tasks."""

from __future__ import annotations
import json

from pydantic import BaseModel, Field


class CorpusDocument(BaseModel):
    """Document in the synthetic corpus with provenance tracking."""

    doc_id: str
    title: str
    content: str
    is_distractor: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentCorpus:
    """Manages synthetic documents and provides context projection for epistemic workers."""

    def __init__(self, documents: list[CorpusDocument] | None = None) -> None:
        self.documents: dict[str, CorpusDocument] = {
            d.doc_id: d for d in (documents or [])
        }

    def add(self, doc: CorpusDocument) -> None:
        self.documents[doc.doc_id] = doc

    def get(self, doc_id: str) -> CorpusDocument | None:
        return self.documents.get(doc_id)

    def as_dict(self) -> dict[str, str]:
        return {doc_id: d.content for doc_id, d in self.documents.items()}

    def project(self, allowed_ids: list[str]) -> str:
        """Context compiler for documents: project only permitted document sources."""
        snippets: list[dict[str, str]] = []
        for doc_id in allowed_ids:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                snippets.append(dict(source_id=doc.doc_id, title=doc.title, content=doc.content))
        return json.dumps(snippets, ensure_ascii=False)

    def full_context(self) -> str:
        """Return the same JSON document interface used by projected contexts."""
        return self.project(list(self.documents))
