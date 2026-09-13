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


def test_study_receipts_preserve_real_interpreter_and_inject_inside_trajectory():
    from esc.study_systems import run_variant
    from esc.benchmark.relational import generate_relational_suite
    from esc.config import RLMConfig
    task = generate_relational_suite(depths=[2],tasks_per_depth=1,width=4)[0]
    code = '''
import json
assert private_scratch == 42
specs = json.loads(task.split('Public step specifications:\\n')[1])
rows = read_rows()
subject = specs[0]['lookup']['subject']
for spec in specs:
    row = next(r for r in rows if r['subject'] == subject and r['relation'] == spec['lookup']['relation'])
    receipt = emit(spec['step_id'], subject, row['object'], [{'source_id':row['source_id'], 'span':row['span']}])
    subject = receipt['value']
    if receipt['stop']:
        break
SUBMIT(answer=subject)
'''
    lm = DummyLM([{'reasoning':'initialize private scratch', 'code':'private_scratch = 42'},
                  {'reasoning':'follow receipts', 'code':code}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        result = run_variant(task,'continuous_emit',lm,RLMConfig(max_iters=2),site=1)
    assert not result['correct'] and not result['abstained']
    assert result['rollouts'][0]['events'][0]['applied']
    assert len(result['rollouts'][0]['events']) == 2
    assert not result['rollouts'][0]['protocol_violations']


def test_premature_submission_can_recover_in_same_interpreter_within_limits():
    from esc.study_systems import run_variant
    from esc.benchmark.relational import generate_relational_suite
    from esc.config import RLMConfig
    task = generate_relational_suite(depths=[2],tasks_per_depth=1,width=4)[0]
    first="private_scratch = 42\nSUBMIT(answer='premature')"
    second='''
import json
assert private_scratch == 42
specs = json.loads(task.split('Public step specifications:\\n')[1])
rows = read_rows()
subject = specs[0]['lookup']['subject']
for spec in specs:
    row = next(r for r in rows if r['subject']==subject and r['relation']==spec['lookup']['relation'])
    receipt = emit(spec['step_id'],subject,row['object'],[{'source_id':row['source_id'],'span':row['span']}])
    subject = receipt['value']
    if receipt['stop']: break
SUBMIT(answer=subject)
'''
    lm=DummyLM([{'reasoning':'premature','code':first},{'reasoning':'follow public protocol','code':second}])
    with dspy.context(lm=lm,adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        result=run_variant(task,'continuous_emit',lm,RLMConfig(max_iters=2),site=1)
    assert not result['correct'] and not result['abstained']
    assert len(result['rollouts'][0]['submission_rejections'])==1
    assert not result['rollouts'][0]['protocol_violations']
    assert len(lm.history)==2


def test_extraction_cannot_bypass_missing_receipts_after_iteration_limit():
    from esc.study_systems import run_variant
    from esc.benchmark.relational import generate_relational_suite
    from esc.config import RLMConfig
    task=generate_relational_suite(depths=[2],tasks_per_depth=1,width=4)[0]
    lm=DummyLM([{'reasoning':'skip receipt','code':"SUBMIT(answer='premature')"},
                {'answer':task.target_answer()}])
    with dspy.context(lm=lm,adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        result=run_variant(task,'continuous_emit',lm,RLMConfig(max_iters=1))
    assert not result['correct'] and result['abstained']
    assert result['rollouts'][0]['protocol_violations']
    assert len(lm.history)==2  # one configured iteration plus the normal fallback


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
