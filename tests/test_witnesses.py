import pytest
from esc.core.types import Evidence, Fact, StepResult, StepSpec
from esc.core.witness import evaluate_witness, verify_deterministic_arithmetic, verify_text_span


def test_verify_text_span():
    corpus = {"doc_1": "Elena Vance founded AcroDyn in 2012 with initial capital."}
    ev_valid = [Evidence(source_id="doc_1", span="founded AcroDyn in 2012")]
    ok, msg = verify_text_span(ev_valid, corpus)
    assert ok is True

    ev_missing_doc = [Evidence(source_id="doc_unknown", span="some span")]
    ok_miss, _ = verify_text_span(ev_missing_doc, corpus)
    assert ok_miss is False

    ev_hallucinated_span = [Evidence(source_id="doc_1", span="AcroDyn acquired Google")]
    ok_hal, _ = verify_text_span(ev_hallucinated_span, corpus)
    assert ok_hal is False


def test_verify_arithmetic():
    f1 = Fact(key="target_rev", value="600", level="supported")
    f2 = Fact(key="alpha_rev", value="200", level="supported")

    ok, msg = verify_deterministic_arithmetic(
        value="3.0",
        goal="Calculate ratio of target rev to alpha rev",
        input_facts=[f1, f2],
    )
    assert ok is True

    # Bad math claim
    ok_bad, _ = verify_deterministic_arithmetic(
        value="5.5",
        goal="Calculate ratio of target rev to alpha rev",
        input_facts=[f1, f2],
    )
    assert ok_bad is False


def test_evaluate_witness_type_1_and_2():
    corpus = {"doc_bio": "Sarah Jenkins oversaw the acquisition of BioVect."}
    step_type_2 = StepSpec(
        step_id="N2",
        goal="Identify target company",
        requires=[],
        permitted_sources=["doc_bio"],
        witness_type="type_2",
    )
    result_supported = StepResult(
        status="supported",
        value="BioVect",
        evidence=[Evidence(source_id="doc_bio", span="acquisition of BioVect")],
    )
    w_res = evaluate_witness(step_type_2, result_supported, [], corpus)
    assert w_res.passed is True
    assert w_res.promoted_level == "supported"
