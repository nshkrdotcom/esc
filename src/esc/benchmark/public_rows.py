"""Lossless, task-independent parsing shared by every study condition."""
import json
from esc.core.lookup import parse_row


def parse_documents(context: str) -> list[dict[str, str]]:
    """Parse ALL JSON documents into rows with source_id and exact span.

    The caller selects subject/relation. No task, target, or evaluator is used.
    Non-relational content is rejected rather than silently dropped.
    """
    documents = json.loads(context)
    if not isinstance(documents, list):
        raise ValueError('Expected a JSON document list')
    rows = []
    for document in documents:
        for span in document['content'].splitlines():
            row = parse_row(span)
            if row is None:
                raise ValueError('Expected authoritative relational rows')
            rows.append(dict(row, source_id=document['source_id'], span=span))
    return rows
