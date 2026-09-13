"""Controls for the relational-depth benchmark, using public-input solvers."""
import json
import re
from unittest.mock import Mock

import dspy
import pytest
from typer.testing import CliRunner

from esc import cli
from esc.benchmark.relational import generate_relational_suite
from esc.core.lookup import parse_row
from esc.core.types import Evidence, Fact, StepResult
from esc.core.witness import evaluate_witness
from esc.runner import run_experiment_1
from esc.systems.system_a import SystemAContinuous
from esc.systems.system_c import SystemCIsolated


def public_rows(task):
    return [parse_row(line) for text in task.corpus_data.values() for line in text.splitlines()]


def follow(rows, subject, relations):
    for relation in relations:
        matches = [row for row in rows if row['subject'] == subject and row['relation'] == relation]
        assert len(matches) == 1
        subject = matches[0]['object']
    return subject


def test_paired_depths_preserve_corpus_and_operation():
    tasks = generate_relational_suite(tasks_per_depth=1, seed=8)
    assert [task.dependency_depth for task in tasks] == [2, 4, 8, 16]
    assert len({task.metadata['corpus_sha256'] for task in tasks}) == 1
    assert len({task.metadata['world_id'] for task in tasks}) == 1
    assert len({task.nodes[0].step_spec.lookup.subject for task in tasks}) == 1
    for task in tasks:
        assert task.corpus_data == tasks[0].corpus_data
        assert len(task.nodes) == task.depth
        assert len(public_rows(task)) == 16 * 16
        assert all(node.step_spec.permitted_sources == list(task.corpus_data) for node in task.nodes)
        assert all(re.fullmatch(r's[0-9a-f]{12}', source) for source in task.corpus_data)
        assert re.fullmatch(r'e[0-9a-f]{12}', task.target_answer())
        assert [node.step_spec.lookup.relation for node in task.nodes] == [
            node.step_spec.lookup.relation for node in tasks[-1].nodes[:task.depth]
        ]


@pytest.mark.parametrize('seed', range(5))
def test_every_intermediate_changes_the_suffix_answer(seed):
    task = generate_relational_suite(depths=[16], tasks_per_depth=1, seed=seed, width=8)[0]
    rows = public_rows(task)
    relations = [node.step_spec.lookup.relation for node in task.nodes]
    start = task.nodes[0].step_spec.lookup.subject
    assert follow(rows, start, relations) == task.target_answer()
    entities = {row['subject'] for row in rows}
    for i, node in enumerate(task.nodes[:-1]):
        wrong = next(entity for entity in entities if entity != node.true_value)
        assert follow(rows, wrong, relations[i+1:]) != task.target_answer()


def test_depth_selection_and_split_do_not_contaminate_worlds():
    whole = generate_relational_suite(tasks_per_depth=2, seed=9)
    subset = generate_relational_suite(depths=[8, 2], tasks_per_depth=2, seed=9)
    by_id = {task.task_id: task for task in whole}
    for task in subset:
        assert task == by_id[task.task_id]
    worlds = []
    for split in ['dev', 'train', 'validation', 'test']:
        tasks = generate_relational_suite(tasks_per_depth=2, seed=9, split=split)
        worlds.append({task.metadata['world_id'] for task in tasks})
    for i, first in enumerate(worlds):
        for second in worlds[i+1:]:
            assert first.isdisjoint(second)


def test_lookup_witness_binds_all_three_fields_and_source():
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    first = task.nodes[0]
    result = StepResult(status='supported', value=first.true_value, evidence=first.true_evidence)
    assert evaluate_witness(first.step_spec, result, [], task.corpus_data).promoted_level == 'verified'
    wrong = next(row['object'] for row in public_rows(task) if row['object'] != first.true_value)
    assert not evaluate_witness(first.step_spec, result.model_copy(update={'value': wrong}), [], task.corpus_data).passed
    wrong_row = next((source, line) for source, text in task.corpus_data.items() for line in text.splitlines()
                     if parse_row(line)['object'] == first.true_value and line != first.true_evidence[0].span)
    unrelated = result.model_copy(update={'evidence': [Evidence(source_id=wrong_row[0], span=wrong_row[1])]})
    assert not evaluate_witness(first.step_spec, unrelated, [], task.corpus_data).passed
    forbidden = first.step_spec.model_copy(update={'permitted_sources': []})
    assert not evaluate_witness(forbidden, result, [], task.corpus_data).passed


def test_conflicting_authoritative_rows_fail_closed():
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    node = task.nodes[0]
    corpus = dict(task.corpus_data)
    source = node.true_evidence[0].source_id
    contradictory = json.loads(node.true_evidence[0].span)
    contradictory['object'] = 'different_entity'
    corpus[source] += '\n' + json.dumps(contradictory)
    result = StepResult(status='supported', value=node.true_value, evidence=node.true_evidence)
    assert not evaluate_witness(node.step_spec, result, [], corpus).passed


def test_lookup_does_not_bypass_missing_or_low_assurance_parent():
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    node = task.nodes[1]
    result = StepResult(status='supported', value=node.true_value, evidence=node.true_evidence)
    for facts in ([], [Fact(key='N1', value=task.nodes[0].true_value, level='candidate')]):
        assert not evaluate_witness(node.step_spec, result, facts, task.corpus_data).passed


class PublicLookupWorker:
    """Deterministic fixture that sees exactly the live worker's public inputs."""
    def __init__(self):
        self.calls = []

    def forward(self, goal, accepted_facts, evidence_context):
        self.calls.append((goal, accepted_facts, evidence_context))
        subject = accepted_facts[0].value if accepted_facts else re.search(r'subject is entity (e\w+)', goal)[1]
        relation = re.search(r'whose relation is (r\w+)', goal)[1]
        for document in json.loads(evidence_context):
            for line in document['content'].splitlines():
                row = parse_row(line)
                if row and row['subject'] == subject and row['relation'] == relation:
                    dspy.settings.usage_tracker.add_usage('fixture', {'prompt_tokens': 1, 'completion_tokens': 1})
                    return StepResult(status='supported', value=row['object'],
                                      evidence=[Evidence(source_id=document['source_id'], span=line)])
        raise AssertionError('Missing public row')


def test_live_input_path_solves_without_labels_and_with_equal_corpus():
    task = generate_relational_suite(depths=[16], tasks_per_depth=1)[0]
    expected = task.target_answer()
    # Poison every evaluator value; the running state and witness must not change.
    task.ground_truth_map = {key: 'poisoned' for key in task.ground_truth_map}
    for node in task.nodes:
        node.true_value = 'poisoned'
        node.true_evidence = []
    worker = PublicLookupWorker()
    result = SystemCIsolated(worker=worker).run(task)
    assert result.final_answer == expected
    assert not result.is_correct  # Only evaluator scoring sees poisoned labels.
    assert len(worker.calls) == 16
    full = task.get_corpus().full_context()
    assert all(call[2] == full for call in worker.calls)
    assert not worker.calls[0][1]
    for i, (_, facts, _) in enumerate(worker.calls[1:], 1):
        assert [fact.key for fact in facts] == [f'N{i}']

    class CaptureContinuous:
        def forward(self, task_description, corpus_context):
            assert corpus_context == full
            assert 'poisoned' not in task_description
            assert 'true_value' not in task_description
            dspy.settings.usage_tracker.add_usage('fixture', {'prompt_tokens': 1, 'completion_tokens': 1})
            return dspy.Prediction(final_answer=expected)
    assert SystemAContinuous(worker=CaptureContinuous()).run(task).final_answer == expected


def test_relational_runner_records_family_pairing_and_real_depth(tmp_path):
    result = run_experiment_1(benchmark='relational_v2', depths=[2, 8, 16], tasks_per_depth=1,
                             repetitions=1, use_mock=True, world_width=4, output_dir=tmp_path)
    assert result.h1_hypothesis['h1_supported'] is None
    assert result.configuration['benchmark'] == 'relational_v2'
    assert set(result.eval_vectors_by_system_and_depth['Condition_C_Isolated']) == {2, 8, 16}
    assert all(run.is_correct for run in result.all_runs if run.system_name == 'Condition_C_Isolated')
    assert len({run.details['world_id'] for run in result.all_runs}) == 1


@pytest.mark.parametrize('args', [['--benchmark', 'unknown'], ['--benchmark', 'legacy', '--split', 'test']])
def test_cli_rejects_uncontrolled_split_configuration_before_run(args, monkeypatch):
    runner = Mock()
    monkeypatch.setattr(cli, 'run_experiment_1', runner)
    result = CliRunner().invoke(cli.app, ['run', *args])
    assert result.exit_code != 0
    runner.assert_not_called()
