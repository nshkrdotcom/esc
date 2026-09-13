import asyncio
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import dspy
import pytest

from esc.audit import audit_budget_run
from esc.budget import BudgetAccountingError, BudgetedLM, EpisodeLedger, EpisodeBudgetExhausted
from esc.journal import RequestJournal
from esc.runner import run_experiment_1
from esc.systems.base import SystemResult
from esc.systems.system_a import SystemAContinuous
from esc.systems.system_c import SystemCIsolated


def fixture_run(path, monkeypatch, budget):
    model = BudgetedLM('openai/fixture', cache=False, num_retries=0, max_tokens=10)
    monkeypatch.setattr(dspy.LM, 'forward', lambda *a, **k: {
        'usage': {'prompt_tokens': 20, 'completion_tokens': 3},
        'choices': [{'finish_reason': 'stop'}]})
    def run(self, task):
        model.forward(prompt='fixture')
        return SystemResult(system_name=self.name, task_id=task.task_id, depth=task.depth,
                            final_answer=task.target_answer(), target_answer=task.target_answer(),
                            is_correct=True)
    monkeypatch.setattr(SystemAContinuous, 'run', run)
    monkeypatch.setattr(SystemCIsolated, 'run', run)
    return run_experiment_1(depths=[2, 4], tasks_per_depth=1, repetitions=2, use_mock=False,
        sub_lm=model, episode_token_budget=budget, output_dir=path, benchmark='relational_v2')


@pytest.mark.parametrize('budget,exhausted,tokens', [(1000, 0, 460), (10, 12, 276)])
def test_independent_audit_both_budget_regimes(tmp_path, monkeypatch, budget, exhausted, tokens):
    fixture_run(tmp_path, monkeypatch, budget)
    report = audit_budget_run(tmp_path)
    assert report['valid'] and report['episodes'] == 12
    assert report['exhausted_episodes'] == exhausted
    assert report['measured_tokens'] == tokens
    assert report['compute_matched'] is False
    assert report['cost_by_depth'][2]['largest_to_smallest_mean_ratio'] == (3 if budget == 1000 else 1)


@pytest.mark.parametrize('mutation', ['duplicate_end', 'usage', 'missing_run', 'sequence', 'late_dispatch'])
def test_auditor_rejects_corrupted_evidence(tmp_path, monkeypatch, mutation):
    fixture_run(tmp_path, monkeypatch, 10)
    journal = tmp_path/'requests.jsonl'
    events = [json.loads(x) for x in journal.read_text().splitlines()]
    if mutation == 'duplicate_end':
        events.insert(2, events[1])
    elif mutation == 'usage':
        events[1]['prompt_tokens'] += 1
    elif mutation == 'sequence':
        events[0]['sequence'] = 8
    elif mutation == 'late_dispatch':
        extra = {**events[0], 'sequence': 3, 'request_id': 2}
        events.insert(2, extra)
    else:
        runs = tmp_path/'runs.jsonl'
        runs.write_text('\n'.join(runs.read_text().splitlines()[1:])+'\n')
    journal.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError):
        audit_budget_run(tmp_path)


def test_concurrent_responses_have_correct_request_identity(tmp_path, monkeypatch):
    journal = RequestJournal(tmp_path/'requests.jsonl')
    ledger = EpisodeLedger(1000, event_sink=journal.for_episode({'task_id':'t', 'system':'A', 'repetition':0}))
    barrier = Barrier(2)
    def reply(self, prompt=None, **kwargs):
        barrier.wait(timeout=10)
        return {'usage': {'prompt_tokens': int(prompt), 'completion_tokens': 1}}
    monkeypatch.setattr(dspy.LM, 'forward', reply)
    model = BudgetedLM('openai/fixture', cache=False, num_retries=0, max_tokens=10)
    with dspy.context(esc_ledger=ledger), ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(contextvars.copy_context().run, model.forward, prompt=str(n)) for n in [7, 31]]
        for future in futures:
            future.result()
    events = [json.loads(x) for x in journal.path.read_text().splitlines()]
    assert [e['sequence'] for e in events] == [1, 2, 3, 4]
    assert {e['prompt_tokens'] for e in events if e['event']=='request_end'} == {7, 31}
    assert len({e['request_id'] for e in events if e['event']=='request_end'}) == 2
    assert ledger.snapshot()['measured_tokens'] == 40


def test_async_rejected_response_preserves_usage_and_finish_reason(tmp_path, monkeypatch):
    journal = RequestJournal(tmp_path/'requests.jsonl')
    ledger = EpisodeLedger(5, event_sink=journal.for_episode({'task_id':'t', 'system':'A', 'repetition':0}))
    async def reply(*a, **k):
        return {'usage': {'prompt_tokens': 7, 'completion_tokens': 5},
                'choices': [{'finish_reason': 'length'}]}
    monkeypatch.setattr(dspy.LM, 'aforward', reply)
    model = BudgetedLM('openai/fixture', cache=False, num_retries=0, max_tokens=10)
    with dspy.context(esc_ledger=ledger), pytest.raises(EpisodeBudgetExhausted):
        asyncio.run(model.aforward(prompt='x'))
    end = json.loads(journal.path.read_text().splitlines()[-1])
    assert end['completion_tokens'] == 5 and end['prompt_tokens'] == 7
    assert end['finish_reasons'] == ['length'] and end['budget_truncated']


def test_journal_failure_blocks_backend_dispatch(monkeypatch):
    def broken_sink(event):
        raise OSError('disk full')
    called = []
    monkeypatch.setattr(dspy.LM, 'forward', lambda *a, **k: called.append(1))
    ledger = EpisodeLedger(100, event_sink=broken_sink)
    model = BudgetedLM('openai/fixture', cache=False, num_retries=0, max_tokens=10)
    with dspy.context(esc_ledger=ledger), pytest.raises(BudgetAccountingError, match='journal'):
        model.forward(prompt='x')
    assert not called
    assert ledger.snapshot()['accounting_failed']
