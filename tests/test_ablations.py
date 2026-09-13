import pytest
from esc.benchmark.generator import generate_epidag_task
from esc.systems.system_c import SystemCIsolated


def test_ablation_no_verification():
    task = generate_epidag_task(task_id="t_ab_noverif", depth=4, seed=42)
    sys = SystemCIsolated(use_mock=True, ablation_mode="no_verification")
    res = sys.run(task)
    assert "no_verification" in res.system_name
    assert res.is_correct is True


def test_ablation_shared_history():
    task = generate_epidag_task(task_id="t_ab_shared", depth=4, seed=42)
    sys = SystemCIsolated(use_mock=True, ablation_mode="shared_history")
    res = sys.run(task)
    assert "shared_history" in res.system_name


def test_ablation_no_typing():
    task = generate_epidag_task(task_id="t_ab_notyping", depth=4, seed=42)
    sys = SystemCIsolated(use_mock=True, ablation_mode="no_typing")
    res = sys.run(task)
    assert "no_typing" in res.system_name


def test_ablation_raw_summaries():
    task = generate_epidag_task(task_id="t_ab_rawsum", depth=4, seed=42)
    sys = SystemCIsolated(use_mock=True, ablation_mode="raw_summaries")
    res = sys.run(task)
    assert "raw_summaries" in res.system_name
