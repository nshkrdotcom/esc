import pytest
from esc.benchmark.generator import generate_epidag_task
from esc.systems.system_a import SystemAContinuous
from esc.systems.system_b import SystemBSearchHeavy
from esc.systems.system_c import SystemCIsolated
from esc.systems.system_d import SystemDGEPA


def test_system_c_isolated_execution():
    task = generate_epidag_task(task_id="t_c", depth=4, seed=42)
    sys_c = SystemCIsolated(use_mock=True, mock_error_rate=0.0)
    res = sys_c.run(task)

    assert res.is_correct is True
    assert res.final_answer == task.target_answer()
    assert res.contract_violations == 0
    assert res.abstained is False


def test_system_c_error_containment():
    """Verify that an injected upstream error halts at boundary rather than contaminating state."""
    task = generate_epidag_task(
        task_id="t_error",
        depth=4,
        inject_error_at_node="N1",
    )
    sys_c = SystemCIsolated(use_mock=True)
    res = sys_c.run(task)

    # In Condition C, an invalid candidate fails witness verification and is not committed
    assert res.error_propagated is False
    assert res.abstained is True


def test_systems_a_b_d():
    task = generate_epidag_task(task_id="t_multi", depth=2, seed=42)

    sys_a = SystemAContinuous(use_mock=True, mock_error_rate=0.0)
    res_a = sys_a.run(task)
    assert res_a.system_name == "Condition_A_Continuous"

    sys_b = SystemBSearchHeavy(use_mock=True, n_samples=2, mock_error_rate=0.0)
    res_b = sys_b.run(task)
    assert res_b.system_name == "Condition_B_SearchHeavy"

    sys_d = SystemDGEPA(use_mock=True)
    res_d = sys_d.run(task)
    assert res_d.system_name == "Condition_D_Isolated_GEPA"
