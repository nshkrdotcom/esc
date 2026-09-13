import pytest
from esc.core.kernel import StateKernel
from esc.core.types import EpistemicLevel, Evidence, Fact, StepResult, StepSpec, level_ok
from esc.core.witness import WitnessResult


def test_level_ordering():
    assert level_ok("verified", "supported") is True
    assert level_ok("supported", "supported") is True
    assert level_ok("candidate", "supported") is False
    assert level_ok("disputed", "candidate") is False
    assert level_ok("verified", "verified") is True


def test_state_kernel_project_context_compiler():
    kernel = StateKernel()
    f1 = Fact(key="alpha_owner", value="Elena Vance", level="supported")
    f2 = Fact(key="beta_rev", value="500", level="candidate")  # only candidate
    kernel.facts["alpha_owner"] = f1
    kernel.facts["beta_rev"] = f2

    # Step requires alpha_owner at level 'supported' -> should succeed and project f1
    step_valid = StepSpec(
        step_id="S1",
        goal="Get owner",
        requires=["alpha_owner"],
        required_level="supported",
    )
    projected = kernel.project(step_valid)
    assert len(projected) == 1
    assert projected[0].key == "alpha_owner"

    # Step requires beta_rev at level 'supported' -> f2 is only 'candidate', so project must exclude it!
    step_insufficient = StepSpec(
        step_id="S2",
        goal="Get rev",
        requires=["beta_rev"],
        required_level="supported",
    )
    projected_insuff = kernel.project(step_insufficient)
    assert len(projected_insuff) == 0


def test_state_kernel_commit():
    kernel = StateKernel()
    candidate = Fact(key="revenue", value="300", level="candidate")

    # Witness failed
    failed_witness = WitnessResult(
        passed=False,
        promoted_level="candidate",
        witness_type="type_1",
        detail="Mismatch",
    )
    committed = kernel.commit(candidate, failed_witness, required_level="verified")
    assert committed is False
    assert "revenue" not in kernel.facts

    # Witness passed and promoted to verified
    passed_witness = WitnessResult(
        passed=True,
        promoted_level="verified",
        witness_type="type_1",
        detail="Verified arithmetic",
    )
    committed = kernel.commit(candidate, passed_witness, required_level="verified")
    assert committed is True
    assert "revenue" in kernel.facts
    assert kernel.facts["revenue"].level == "verified"
