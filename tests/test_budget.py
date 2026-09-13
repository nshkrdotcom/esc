import contextvars
from concurrent.futures import ThreadPoolExecutor
import asyncio
import dspy
import pytest

from esc.budget import BudgetedLM, EpisodeLedger, EpisodeBudgetExhausted, BudgetAccountingError
from esc.workers.usage import invoke_with_usage


def lm():
    return BudgetedLM("openai/test", cache=False, num_retries=0, max_tokens=10)


def test_reservation_concurrency_and_reconciliation():
    ledger = EpisodeLedger(20)
    def reserve():
        try:
            return ledger.reserve(10)
        except EpisodeBudgetExhausted:
            return None
    with ThreadPoolExecutor(8) as pool:
        handles = [h for h in pool.map(lambda _: reserve(), range(8)) if h]
    assert len(handles) == 2
    for handle, cap in handles:
        ledger.commit(handle, {"prompt_tokens": 8, "completion_tokens": cap})
    state = ledger.snapshot()
    assert state['measured_tokens'] == 36 and state['overshoot'] == 16
    assert state['pending_reserved'] == 0 and state['exhausted']


def test_unknown_usage_is_not_measured_or_refunded():
    ledger = EpisodeLedger(20)
    handle, _ = ledger.reserve(10)
    ledger.fail(handle)
    assert ledger.snapshot()['unknown_reserved'] == 10
    assert ledger.snapshot()['measured_tokens'] == 0
    with pytest.raises(BudgetAccountingError):
        ledger.reserve(1)


def test_budget_reduced_truncation_and_invalid_usage(monkeypatch):
    model = lm()
    monkeypatch.setattr(dspy.LM, 'forward', lambda *a, **k: {
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
        'choices': [{'finish_reason': 'length'}]})
    with dspy.context(esc_ledger=EpisodeLedger(5)):
        with pytest.raises(EpisodeBudgetExhausted):
            model.forward(prompt='x')
    ledger = EpisodeLedger(100)
    handle, _ = ledger.reserve(10)
    with pytest.raises(BudgetAccountingError):
        ledger.commit(handle, {})
    assert ledger.snapshot()['unknown_reserved'] == 10


def test_failed_backend_remains_unknown(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError('provider failed')
    monkeypatch.setattr(dspy.LM, 'forward', fail)
    ledger = EpisodeLedger(100)
    with dspy.context(esc_ledger=ledger):
        with pytest.raises(RuntimeError, match='provider failed'):
            lm().forward(prompt='x')
    assert ledger.snapshot()['accounting_failed']
    assert ledger.snapshot()['measured_tokens'] == 0


def test_wrapper_caps_and_blocks_both_paths(monkeypatch):
    calls = []
    def response(self, **kwargs):
        calls.append(kwargs['max_tokens'])
        return {'usage': {'prompt_tokens': 4, 'completion_tokens': kwargs['max_tokens']}}
    async def async_response(self, **kwargs):
        return response(self, **kwargs)
    monkeypatch.setattr(dspy.LM, 'forward', response)
    monkeypatch.setattr(dspy.LM, 'aforward', async_response)
    model = lm()
    for asynchronous in [False, True]:
        ledger = EpisodeLedger(7)
        with dspy.context(esc_ledger=ledger):
            with pytest.raises(EpisodeBudgetExhausted):
                if asynchronous:
                    asyncio.run(model.aforward(prompt='hello'))
                else:
                    model.forward(prompt='hello')
            with pytest.raises(EpisodeBudgetExhausted):
                model.forward(prompt='blocked')
        assert ledger.snapshot()['measured_tokens'] == 11
    assert calls == [7, 7]


def test_swallowed_recursive_error_cannot_return_prediction(monkeypatch):
    model = lm()
    monkeypatch.setattr(dspy.LM, 'forward', lambda *a, **k:
                        {'usage': {'prompt_tokens': 10, 'completion_tokens': 1}})
    class Worker:
        def forward(self):
            try:
                model.forward(prompt='subcall')
            except EpisodeBudgetExhausted:
                pass
            return dspy.Prediction(final_answer='fabricated after exhaustion')
    with dspy.context(esc_ledger=EpisodeLedger(5)):
        with pytest.raises(EpisodeBudgetExhausted):
            invoke_with_usage(Worker())


def test_context_copied_to_threads_not_other_episodes(monkeypatch):
    model = lm()
    monkeypatch.setattr(dspy.LM, 'forward', lambda *a, **k:
                        {'usage': {'prompt_tokens': 2, 'completion_tokens': 1}})
    for _ in range(2):
        ledger = EpisodeLedger(100)
        with dspy.context(esc_ledger=ledger), ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(contextvars.copy_context().run, model.forward, prompt='x') for _ in range(2)]
            for job in jobs:
                job.result()
        assert ledger.snapshot()['measured_tokens'] == 6


def test_runner_continues_exhausted_episodes(tmp_path, monkeypatch):
    from esc.runner import run_experiment_1
    from esc.systems.system_a import SystemAContinuous
    from esc.systems.system_c import SystemCIsolated
    from esc.budget import current_ledger
    def exhaust(self, task):
        ledger = current_ledger()
        handle, _ = ledger.reserve(5)
        ledger.commit(handle, {'prompt_tokens': 5, 'completion_tokens': 5})
        ledger.check()
    monkeypatch.setattr(SystemAContinuous, 'run', exhaust)
    monkeypatch.setattr(SystemCIsolated, 'run', exhaust)
    result = run_experiment_1(depths=[2], tasks_per_depth=2, repetitions=1,
        use_mock=False, sub_lm=lm(), episode_token_budget=5, output_dir=tmp_path)
    assert len(result.all_runs) == 6
    assert all(r.budget_exhausted and r.tokens_used == 10 for r in result.all_runs)
    assert not (tmp_path/'failure.json').exists()
    assert result.configuration['compute_matched'] is False


def test_b_preserves_completed_vote(monkeypatch):
    from esc.systems.system_b import SystemBSearchHeavy
    from esc.systems.base import SystemResult
    from esc.benchmark.relational import generate_relational_suite
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    system = SystemBSearchHeavy(use_mock=True)
    calls = []
    def run(task):
        calls.append(1)
        if len(calls) > 1:
            raise EpisodeBudgetExhausted()
        return SystemResult(system_name='A', task_id=task.task_id, depth=2,
                            final_answer=task.target_answer(), target_answer=task.target_answer(), is_correct=True)
    monkeypatch.setattr(system.sub_system, 'run', run)
    result = system.run(task)
    assert result.is_correct and result.budget_exhausted
    assert result.details['rollouts_count'] == 1


def test_c_keeps_audit_when_budget_stops_next_step(monkeypatch):
    from esc.systems.system_c import SystemCIsolated
    from esc.benchmark.relational import generate_relational_suite
    from esc.core.types import StepResult
    task = generate_relational_suite(depths=[2], tasks_per_depth=1)[0]
    first = task.nodes[0]
    calls = []
    def invoke(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise EpisodeBudgetExhausted()
        return StepResult(status='supported', value=first.true_value,
                          evidence=first.true_evidence), {'total_tokens': 10}
    monkeypatch.setattr('esc.systems.system_c.invoke_with_usage', invoke)
    result = SystemCIsolated(worker=object()).run(task)
    assert result.budget_exhausted and result.abstained
    assert result.intermediate_answers == {'N1': first.true_value}
    assert len(result.details['audit_log']) == 2


@pytest.mark.parametrize('invalid', [0, -1, True, 1.5, float('nan'), None])
def test_invalid_allowances_cannot_mutate_ledger(invalid):
    with pytest.raises(ValueError, match='positive integer'):
        EpisodeLedger(invalid)
    ledger = EpisodeLedger(10)
    with pytest.raises(ValueError, match='positive integer'):
        ledger.reserve(invalid)
    assert ledger.snapshot()['calls_dispatched'] == 0
    assert ledger.snapshot()['pending_reserved'] == 0


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('response', [None, {'usage': 'malformed'}, {'usage': {}}])
def test_malformed_response_durably_invalidates_accounting(monkeypatch, asynchronous, response):
    def reply(*a, **k):
        return response
    async def async_reply(*a, **k):
        return response
    monkeypatch.setattr(dspy.LM, 'forward', reply)
    monkeypatch.setattr(dspy.LM, 'aforward', async_reply)
    ledger = EpisodeLedger(100)
    with dspy.context(esc_ledger=ledger):
        with pytest.raises(BudgetAccountingError):
            if asynchronous:
                asyncio.run(lm().aforward(prompt='x'))
            else:
                lm().forward(prompt='x')
        with pytest.raises(BudgetAccountingError):
            ledger.reserve(1)
    assert ledger.snapshot()['unknown_reserved'] == 10
    assert ledger.snapshot()['pending_reserved'] == 0
    assert ledger.snapshot()['measured_tokens'] == 0


@pytest.mark.parametrize('error', [KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()])
def test_worker_cancellation_is_not_replaced_by_exhaustion(error):
    ledger = EpisodeLedger(5)
    handle, _ = ledger.reserve(5)
    ledger.commit(handle, {'prompt_tokens': 5, 'completion_tokens': 1})
    class Worker:
        def forward(self):
            raise error
    with dspy.context(esc_ledger=ledger), pytest.raises(type(error)) as caught:
        invoke_with_usage(Worker())
    assert caught.value is error


def test_runner_preserves_original_provider_error(tmp_path, monkeypatch):
    import json
    from esc.runner import run_experiment_1
    from esc.systems.system_a import SystemAContinuous
    original = RuntimeError('provider connection lost')
    def fail(*a, **k):
        raise original
    monkeypatch.setattr(dspy.LM, 'forward', fail)
    model = lm()
    class Worker:
        def forward(self):
            model.forward(prompt='x')
    monkeypatch.setattr(SystemAContinuous, 'run', lambda *a: invoke_with_usage(Worker()))
    with pytest.raises(RuntimeError) as caught:
        run_experiment_1(depths=[2], tasks_per_depth=1, repetitions=1, use_mock=False,
                         sub_lm=model, episode_token_budget=100, output_dir=tmp_path)
    assert caught.value is original
    record = json.loads((tmp_path/'failure.json').read_text())
    assert record['error_type'] == 'RuntimeError'
    assert record['error'] == 'provider connection lost'
    assert record['ledger']['unknown_reserved'] == 10
    assert not (tmp_path/'experiment_1_summary.json').exists()


@pytest.mark.parametrize('invalid', [True, 1.5])
def test_runner_rejects_noninteger_allowance_before_artifacts(tmp_path, invalid):
    from esc.runner import run_experiment_1
    with pytest.raises(ValueError, match='positive integer'):
        run_experiment_1(use_mock=False, sub_lm=lm(), episode_token_budget=invalid,
                         output_dir=tmp_path)
    assert not list(tmp_path.iterdir())
