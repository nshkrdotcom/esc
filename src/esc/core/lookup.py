"""Deterministic witnesses over public authoritative subject/relation/object rows."""
import json

from esc.core.types import Fact, StepResult, StepSpec, level_ok


def parse_row(line: str) -> dict[str, str] | None:
    try:
        row = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(row, dict) or set(row) != {"subject", "relation", "object"}:
        return None
    return row if all(isinstance(value, str) and value for value in row.values()) else None


def verify_lookup(step: StepSpec, result: StepResult, inputs: list[Fact],
                  corpus: dict[str, str]) -> tuple[bool, str]:
    """Validate a cited row against the operation and supplied state, not labels.

    Rows are a public authoritative database. This strong Type I witness makes
    no semantic-entailment claim; the subject, relation and object are explicit.
    """
    spec = step.lookup
    if spec is None or step.witness_type != "type_1":
        return False, "Lookup witness requires an explicit Type I lookup contract."
    if spec.subject_key is not None:
        if step.requires != [spec.subject_key]:
            return False, "Lookup subject key must match the single required parent."
        matches = [fact for fact in inputs if fact.key == spec.subject_key]
        if len(matches) != 1 or not level_ok(matches[0].level, step.required_level):
            return False, "Missing or insufficiently assured lookup subject."
        subject = matches[0].value
    else:
        if step.requires:
            return False, "Literal lookup must not bypass required parents."
        subject = spec.subject

    allowed = {key: text for key, text in corpus.items() if key in step.permitted_sources}
    objects = set()
    for document in allowed.values():
        for line in document.splitlines():
            row = parse_row(line)
            if row and row["subject"] == subject and row["relation"] == spec.relation:
                objects.add(row["object"])
    if objects != {result.value}:
        return False, "Public rows do not uniquely support the claimed subject/relation/object."
    if not result.evidence:
        return False, "Lookup requires a citation to the complete supporting row."
    for evidence in result.evidence:
        # A complete row must be cited in its actual permitted document.
        document = allowed.get(evidence.source_id)
        if document is None or evidence.span not in document.splitlines():
            return False, "Cited row is absent from the permitted source."
        row = parse_row(evidence.span)
        if row != {"subject": subject, "relation": spec.relation, "object": result.value}:
            return False, "Citation belongs to a different subject, relation or object."
    return True, "Authoritative row matches the public lookup contract and supplied subject."
