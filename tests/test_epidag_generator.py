import pytest
from esc.benchmark.generator import generate_epidag_task, generate_task_suite


def test_epidag_depths():
    for d in [2, 4, 8, 16]:
        task = generate_epidag_task(task_id=f"test_d{d}", depth=d, seed=42)
        assert task.depth == d
        assert len(task.nodes) == d
        assert task.final_node_id in task.ground_truth_map
        assert task.target_answer() != ""

        # Verify corpus projection works
        corpus = task.get_corpus()
        assert len(corpus.documents) > 0
        all_text = corpus.full_context()
        assert len(all_text) > 0


def test_epidag_error_injection():
    task = generate_epidag_task(
        task_id="test_error",
        depth=4,
        inject_error_at_node="N2",
    )
    assert task.injected_error_node_id == "N2"
    n2 = task.node_by_id("N2")
    assert n2 is not None
    assert n2.injected_error_value is not None
    assert "CORRUPT" in n2.injected_error_value


def test_generate_task_suite():
    suite = generate_task_suite(depths=[2, 4], tasks_per_depth=2, seed=10)
    assert len(suite) == 4
