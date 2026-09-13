"""Opt-in Deno/Pyodide integration using scripted responses, never live inference.

Run with ESC_TEST_RLM=1 uv run pytest tests/test_rlm_integration.py
"""
import os

import dspy
import pytest
from dspy.utils import DummyLM

from esc.core.types import Fact
from esc.workers.continuous import ContinuousWorker
from esc.workers.epistemic import EpistemicWorker
from esc.workers.usage import invoke_with_usage

pytestmark = pytest.mark.skipif(os.environ.get('ESC_TEST_RLM') != '1', reason='Opt-in Deno/Pyodide check')


def test_real_rlm_accepts_typed_null_abstention():
    lm = DummyLM([{'reasoning': 'insufficient evidence',
                   'code': 'SUBMIT(reasoning_steps=[], final_answer=None)'}])
    with dspy.context(adapter=dspy.ChatAdapter()):
        result, usage = invoke_with_usage(ContinuousWorker(sub_lm=lm, max_iters=1),
                                         task_description='unknown', corpus_context='[]')
    assert result.final_answer is None and usage['lm_calls'] == 1


def test_real_rlm_serialization_usage_and_fresh_interpreters():
    lm = DummyLM([
        {'reasoning': 'scripted fixture', 'code': "assert 'previous_call' not in globals()\nprevious_call = 1\nSUBMIT(result={'status': 'supported', 'value': accepted_facts[0]['value'], 'evidence': [], 'assumptions': []})"},
        {'reasoning': 'scripted fixture', 'code': "assert 'previous_call' not in globals()\nSUBMIT(result={'status': 'supported', 'value': '201', 'evidence': [], 'assumptions': []})"},
        {'reasoning': 'scripted fixture', 'code': "SUBMIT(reasoning_steps=['scripted fixture'], final_answer='true')"},
    ])
    worker = EpistemicWorker(sub_lm=lm, max_iters=1)
    with dspy.context(adapter=dspy.ChatAdapter()):
        for expected in ['200', '201']:
            result, usage = invoke_with_usage(worker, goal='scripted test',
                accepted_facts=[Fact(key='x', value='200', level='supported')], evidence_context='')
            assert result.value == expected
            assert usage['lm_calls'] == 1
            assert worker.last_trajectory
        result, usage = invoke_with_usage(ContinuousWorker(sub_lm=lm, max_iters=1),
            task_description='scripted test', corpus_context='')
        assert result.final_answer == 'true'
        assert usage['lm_calls'] == 1


def test_budget_survives_real_recursive_sandbox_bridge(monkeypatch):
    from litellm import ModelResponse
    from esc.budget import BudgetedLM, EpisodeLedger, EpisodeBudgetExhausted
    calls = []
    def response(self, **kwargs):
        calls.append(kwargs)
        content = ("[[ ## reasoning ## ]]\nfixture\n[[ ## code ## ]]\n"
                   "llm_query('recursive')\nSUBMIT(final_answer='invalid', reasoning_steps=[])\n"
                   "[[ ## completed ## ]]")
        return ModelResponse(model='openai/test', choices=[{'message': {'role': 'assistant',
            'content': content}, 'finish_reason': 'stop'}],
            usage={'prompt_tokens': 5 if len(calls) == 1 else 60,
                   'completion_tokens': 5, 'total_tokens': 10 if len(calls) == 1 else 65})
    monkeypatch.setattr(dspy.LM, 'forward', response)
    model = BudgetedLM('openai/test', max_tokens=10, cache=False, num_retries=0)
    ledger = EpisodeLedger(50)
    with dspy.context(esc_ledger=ledger, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        with pytest.raises(EpisodeBudgetExhausted):
            invoke_with_usage(ContinuousWorker(sub_lm=model, max_iters=1),
                              task_description='fixture', corpus_context='')
    assert len(calls) == 2
    assert ledger.snapshot()['measured_tokens'] == 75
    assert ledger.snapshot()['calls_blocked'] >= 1


def test_real_batched_subcalls_share_ledger_and_journal(monkeypatch):
    from threading import Barrier
    from litellm import ModelResponse
    from esc.budget import BudgetedLM, EpisodeLedger, EpisodeBudgetExhausted
    barrier = Barrier(2)
    events = []
    def response(self, prompt=None, messages=None, **kwargs):
        if prompt is not None:
            barrier.wait(timeout=10)
            content, prompt_tokens = 'sub-answer', 25
        else:
            content = ("[[ ## reasoning ## ]]\nfixture\n[[ ## code ## ]]\n"
                       "llm_query_batched(['one', 'two'])\n"
                       "SUBMIT(final_answer='invalid', reasoning_steps=[])\n[[ ## completed ## ]]")
            prompt_tokens = 5
        return ModelResponse(model='openai/test', choices=[{'message': {'role':'assistant', 'content':content},
            'finish_reason':'stop'}], usage={'prompt_tokens':prompt_tokens,'completion_tokens':5,
                                          'total_tokens':prompt_tokens+5})
    monkeypatch.setattr(dspy.LM, 'forward', response)
    model = BudgetedLM('openai/test', max_tokens=10, cache=False, num_retries=0)
    ledger = EpisodeLedger(50, event_sink=events.append)
    with dspy.context(esc_ledger=ledger, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        with pytest.raises(EpisodeBudgetExhausted):
            invoke_with_usage(ContinuousWorker(sub_lm=model, max_iters=1),
                              task_description='fixture', corpus_context='')
    state = ledger.snapshot()
    assert state['measured_tokens'] == 70 and state['calls_dispatched'] == 3
    assert state['pending_reserved'] == state['unknown_reserved'] == 0
    assert sum(e.get('prompt_tokens', 0) + e.get('completion_tokens', 0) for e in events) == 70
    assert len({e['request_id'] for e in events if e['event']=='request_end'}) == 3
