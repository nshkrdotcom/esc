"""Offline regressions for issues found before the first live pilot."""
import json
from unittest.mock import Mock

import dspy
import pytest

from esc.benchmark.generator import generate_epidag_task, generate_task_suite
from esc.core.answers import answers_equal
from esc.core.kernel import StateKernel
from esc.core.types import Evidence, Fact, StepResult, StepSpec
from esc.core.witness import WitnessResult, evaluate_witness, verify_deterministic_arithmetic
from esc.eval.metrics import compute_eval_vector
from esc.runner import run_experiment_1
from esc.systems.system_a import SystemAContinuous
from esc.systems.system_b import SystemBSearchHeavy
from esc.systems.system_c import SystemCIsolated
from esc.systems.system_d import SystemDGEPA
from esc.workers.epistemic import EpistemicWorker
from esc.workers.continuous import ContinuousWorker
from esc.workers.usage import invoke_with_usage


@pytest.mark.parametrize('value', ['0.3333', '3 and also 4', 'nan', 'inf'])
def test_arithmetic_rejects_inverse_and_non_scalar_claims(value):
    facts = [Fact(key='a', value='600', level='supported'), Fact(key='b', value='200', level='supported')]
    assert not verify_deterministic_arithmetic(value, 'Calculate ratio', facts)[0]


def test_arithmetic_does_not_accept_identity_for_unknown_operation():
    assert not verify_deterministic_arithmetic('600', 'Calculate ratio', [Fact(key='a', value='600', level='supported')])[0]


@pytest.mark.parametrize('value,source,assumptions', [
    ('999', 'allowed', []), ('200', 'forbidden', []), ('200', 'allowed', ['Maybe this is a different company']),
])
def test_grounding_rejects_unrelated_value_source_and_assumptions(value, source, assumptions):
    step = StepSpec(step_id='n', goal='Retrieve revenue', permitted_sources=['allowed'])
    result = StepResult(status='supported', value=value, assumptions=assumptions,
                        evidence=[Evidence(source_id=source, span='revenue 200')])
    witness = evaluate_witness(step, result, [], {'allowed': 'revenue 200', 'forbidden': 'revenue 200'})
    assert not witness.passed


@pytest.mark.parametrize('value', ['not approved', 'approved and rejected', 'stage_99_cleared'])
def test_transition_requires_exact_output(value):
    step = StepSpec(step_id='n', goal='Generate final recommendation: approved if threshold passed, else rejected.', witness_type='type_1')
    result = StepResult(status='supported', value=value)
    assert not evaluate_witness(step, result, [Fact(key='p', value='true', level='verified')], {}).passed


def test_kernel_projection_cannot_mutate_canonical_state():
    kernel = StateKernel()
    fact = Fact(key='a', value='200', level='supported')
    witness = WitnessResult(passed=True, promoted_level='supported', witness_type='type_2', detail='test')
    assert kernel.commit(fact, witness)
    fact.value = 'corrupt'
    projected = kernel.project(StepSpec(step_id='b', goal='next', requires=['a']))
    projected[0].value = 'corrupt'
    assert kernel.get_fact('a').value == '200'


@pytest.mark.parametrize('size,depth', [(2, 2), (4, 4), (8, 7), (16, 15)])
def test_public_instructions_depth_and_both_decision_branches(size, depth):
    for seed in range(8):
        task = generate_epidag_task('t', depth=size, seed=seed)
        assert task.dependency_depth == depth
        public = task.public_description()
        assert 'true_value' not in public and 'ground_truth' not in public
        for node in task.nodes:
            if 'Calculate' in node.step_spec.goal:
                assert '(' not in node.step_spec.goal
        result = SystemCIsolated(use_mock=True).run(task)
        assert result.is_correct, (seed, size, result.details)
    if size >= 8:
        assert {generate_epidag_task('t', depth=size, seed=s).target_answer() for s in range(20)} == (
            {'approved', 'rejected'} if size == 8 else {'stage_16_cleared', 'stage_16_halted'}
        )


def test_clean_and_injected_task_ids_are_disjoint_and_paired():
    clean = generate_task_suite(depths=[2], seed=1)
    injected = generate_task_suite(depths=[2], seed=1, inject_errors=True)
    assert not {t.task_id for t in clean} & {t.task_id for t in injected}
    assert [t.corpus_data for t in clean] == [t.corpus_data for t in injected]


def test_missing_epc_and_promotions_are_not_zero_success():
    task = generate_epidag_task('t', depth=2)
    result = SystemAContinuous(use_mock=True).run(task)
    vector = compute_eval_vector('a', {'t': [result]}, k=1)
    assert vector.error_propagation is None
    assert vector.false_promotion_rate is None


def test_fail_loudly_instead_of_changing_architecture(monkeypatch):
    monkeypatch.setattr(dspy, 'RLM', Mock(side_effect=RuntimeError('interpreter unavailable')))
    for worker in (EpistemicWorker, ContinuousWorker):
        with pytest.raises(RuntimeError, match='interpreter unavailable'):
            worker()


def test_malformed_typed_output_is_not_a_supported_claim():
    worker = EpistemicWorker(use_rlm=False)
    worker.solve = Mock(return_value=dspy.Prediction(result='untyped free text'))
    with pytest.raises(ValueError):
        worker.forward(goal='g', accepted_facts=[], evidence_context='')


def test_usage_includes_multiple_calls_and_models():
    class Worker:
        def forward(self):
            dspy.settings.usage_tracker.add_usage('root', {'prompt_tokens': 100, 'completion_tokens': 10})
            dspy.settings.usage_tracker.add_usage('sub', {'prompt_tokens': 20, 'completion_tokens': 5})
            dspy.settings.usage_tracker.add_usage('sub', {'prompt_tokens': 30, 'completion_tokens': 6})
            return 'result'
    _, usage = invoke_with_usage(Worker())
    assert usage['total_tokens'] == 171
    assert usage['lm_calls'] == 3
    with pytest.raises(RuntimeError, match='usage'):
        invoke_with_usage(Mock(forward=Mock(return_value='no usage')))


def test_runtime_errors_are_not_abstentions():
    worker = Mock(forward=Mock(side_effect=RuntimeError('backend failed')))
    task = generate_epidag_task('t', depth=2)
    for system in (SystemAContinuous(worker=worker), SystemCIsolated(worker=worker)):
        with pytest.raises(RuntimeError, match='backend failed'):
            system.run(task)


def test_b_keeps_rng_across_rollouts():
    system = SystemBSearchHeavy(use_mock=True)
    before = system.sub_system.worker.rng.getstate()
    system.run(generate_epidag_task('t', depth=2))
    assert system.sub_system.worker.rng.getstate() != before


def test_unimplemented_features_rejected():
    with pytest.raises(NotImplementedError):
        SystemDGEPA()
    with pytest.raises(NotImplementedError):
        SystemCIsolated(use_mock=True, ablation_mode='llm_verifier')


def test_runner_persists_clean_runs_and_refuses_overwrite(tmp_path):
    result = run_experiment_1(depths=[2], tasks_per_depth=1, repetitions=1, output_dir=tmp_path)
    runs = [json.loads(line) for line in (tmp_path / 'runs.jsonl').read_text().splitlines()]
    assert len(runs) == 3
    assert all(not run['details'].get('injected_error') for run in runs)
    assert result.h1_hypothesis['h1_supported'] is None
    summary = json.loads((tmp_path / 'experiment_1_summary.json').read_text())
    assert len(summary['eval_vectors_by_system_and_depth']) == 3
    assert runs[-1]['details']['audit_log']
    with pytest.raises(FileExistsError):
        run_experiment_1(depths=[2], tasks_per_depth=1, repetitions=1, output_dir=tmp_path)


def test_runner_preserves_completed_episode_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(SystemBSearchHeavy, 'run', Mock(side_effect=RuntimeError('failure')))
    with pytest.raises(RuntimeError):
        run_experiment_1(depths=[2], tasks_per_depth=1, repetitions=1, output_dir=tmp_path)
    assert len((tmp_path / 'runs.jsonl').read_text().splitlines()) == 1
    assert json.loads((tmp_path / 'failure.json').read_text())['error_type'] == 'RuntimeError'
    assert not (tmp_path / 'experiment_1_summary.json').exists()


@pytest.mark.parametrize('value,target,expected', [('3.0000','3.0',True),('3.01','3',False),('TRUE','true',True),('nan','nan',False)])
def test_answer_normalization(value, target, expected):
    assert answers_equal(value,target) is expected


def test_gepa_rejects_substring_answers_and_missing_target():
    from types import SimpleNamespace
    from esc.eval.gepa_metric import epistemic_gepa_metric
    assert epistemic_gepa_metric(SimpleNamespace(target_answer='42'), SimpleNamespace(final_answer='142')).score == 0
    assert epistemic_gepa_metric(SimpleNamespace(target_answer='approved'), SimpleNamespace(final_answer='not approved')).score == 0
    with pytest.raises(ValueError, match='target'):
        epistemic_gepa_metric(SimpleNamespace(), SimpleNamespace(final_answer='anything'))


def test_live_runner_rejects_cached_lm_without_inference(tmp_path):
    lm = dspy.LM('ollama_chat/qwen3:14b', cache=True)
    with pytest.raises(ValueError, match='cache=False'):
        run_experiment_1(depths=[2], tasks_per_depth=1, repetitions=1, use_mock=False, sub_lm=lm, output_dir=tmp_path)
    assert not (tmp_path / 'manifest.json').exists()


@pytest.mark.parametrize('arguments', [
    ['--tasks-per-depth', '0'], ['--repetitions', '0'], ['--depths', '2,2'], ['--depths', '3'],
    ['--depths', 'bad'], ['--reflection-model', 'unused'],
])
def test_cli_rejects_bad_settings_before_work(arguments, monkeypatch):
    from typer.testing import CliRunner
    from esc import cli
    runner = Mock(side_effect=AssertionError('runner should not execute'))
    monkeypatch.setattr(cli, 'run_experiment_1', runner)
    result = CliRunner().invoke(cli.app, ['run', *arguments])
    assert result.exit_code != 0
    runner.assert_not_called()


def test_cli_renders_nullable_metrics_and_actual_depths(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from esc import cli
    result = run_experiment_1(depths=[8], tasks_per_depth=1, repetitions=1, output_dir=tmp_path)
    monkeypatch.setattr(cli, 'run_experiment_1', Mock(return_value=result))
    rendered = CliRunner().invoke(cli.app, ['run', '--depths', '8'])
    assert rendered.exit_code == 0, rendered.exception
    assert 'N/A' in rendered.output
    assert 'Not tested' in rendered.output
    assert all(run.depth == 7 for run in result.all_runs)


def test_vote_groups_equivalent_scalar_spellings():
    from esc.core.answers import vote_key
    assert vote_key('3.0000') == vote_key('3.0') == vote_key('3')
    assert vote_key(' TRUE ') == 'true'


def test_invalid_injection_node_rejected():
    with pytest.raises(ValueError, match='injection node'):
        generate_epidag_task('t', depth=2, inject_error_at_node='N99')
