import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from esc.benchmark.relational import generate_relational_suite
from esc.core.types import StepResult
from esc.runner import run_experiment_1
from esc.systems.system_a import SystemAContinuous
from esc.systems.system_b import SystemBSearchHeavy
from esc.systems.system_c import SystemCIsolated


class MalformedWorker:
    def forward(self, **inputs):
        dspy.settings.usage_tracker.add_usage('fixture', {'prompt_tokens':20, 'completion_tokens':3})
        raise AdapterParseError('ChatAdapter', dspy.Signature('x -> y'), 'missing field markers')


@pytest.mark.parametrize('answer', [None, '', '   '])
def test_explicit_abstentions_are_charged_and_do_not_vote(answer):
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    class Worker:
        calls = 0
        def forward(self, **inputs):
            self.calls += 1
            dspy.settings.usage_tracker.add_usage('fixture', {'prompt_tokens':20, 'completion_tokens':3})
            return dspy.Prediction(final_answer=answer if self.calls < 3 else task.target_answer())
    system = SystemBSearchHeavy(use_mock=True)
    system.sub_system = SystemAContinuous(worker=Worker())
    result = system.run(task)
    assert result.is_correct and not result.model_output_error
    assert result.tokens_used == 69 and result.details['rollouts_count'] == 3
    assert all(r['abstained'] for r in result.details['rollouts'][:2])


def test_missing_answer_is_not_silently_treated_as_abstention():
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    class Worker:
        def forward(self, **inputs):
            dspy.settings.usage_tracker.add_usage('fixture', {'prompt_tokens':20, 'completion_tokens':3})
            return dspy.Prediction(reasoning_steps=[])
    result = SystemAContinuous(worker=Worker()).run(task)
    assert result.model_output_error and result.tokens_used == 23


def test_model_format_error_is_measured_unsuccessful_attempt():
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    for system in [SystemAContinuous(worker=MalformedWorker()), SystemCIsolated(worker=MalformedWorker())]:
        result = system.run(task)
        assert result.model_output_error and result.abstained and not result.is_correct
        assert result.tokens_used == 23 and result.final_answer is None


def test_b_counts_bad_rollout_but_votes_over_valid_answers():
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    class Worker(MalformedWorker):
        calls = 0
        def forward(self, **inputs):
            self.calls += 1
            if self.calls == 1:
                return super().forward(**inputs)
            dspy.settings.usage_tracker.add_usage('fixture', {'prompt_tokens':20, 'completion_tokens':3})
            return dspy.Prediction(final_answer=task.target_answer())
    system = SystemBSearchHeavy(use_mock=True)
    system.sub_system = SystemAContinuous(worker=Worker())
    result = system.run(task)
    assert result.is_correct and result.model_output_error
    assert result.tokens_used == 69 and result.details['rollouts_count'] == 3
    assert result.details['rollouts'][0]['model_output_error']


def test_c_preserves_committed_state_before_malformed_output():
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    class Worker(MalformedWorker):
        calls = 0
        def forward(self, **inputs):
            self.calls += 1
            if self.calls > 1:
                return super().forward(**inputs)
            dspy.settings.usage_tracker.add_usage('fixture', {'prompt_tokens':20, 'completion_tokens':3})
            return StepResult(status='supported', value=task.nodes[0].true_value,
                              evidence=task.nodes[0].true_evidence)
    result = SystemCIsolated(worker=Worker()).run(task)
    assert result.model_output_error and not result.is_correct
    assert result.tokens_used == 46
    assert result.intermediate_answers == {'N1':task.nodes[0].true_value}
    assert len(result.details['audit_log']) == 2
    assert not result.details['audit_log'][-1]['committed']


def test_runner_finishes_all_episodes_with_bad_model_format(tmp_path, monkeypatch):
    import esc.runner as runner
    monkeypatch.setattr(runner, 'SystemAContinuous', lambda **k: SystemAContinuous(worker=MalformedWorker()))
    monkeypatch.setattr(runner, 'SystemCIsolated', lambda **k: SystemCIsolated(worker=MalformedWorker()))
    def search(**kwargs):
        system = SystemBSearchHeavy(use_mock=True)
        system.sub_system = SystemAContinuous(worker=MalformedWorker())
        return system
    monkeypatch.setattr(runner, 'SystemBSearchHeavy', search)
    result = run_experiment_1(depths=[2], tasks_per_depth=2, repetitions=1, use_mock=False,
        sub_lm=dspy.LM('openai/fixture', cache=False), output_dir=tmp_path)
    assert len(result.all_runs) == 6
    assert all(r.model_output_error for r in result.all_runs)
    assert (tmp_path/'experiment_1_summary.json').exists()
    assert not (tmp_path/'failure.json').exists()
    assert len((tmp_path/'runs.jsonl').read_text().splitlines()) == 6
