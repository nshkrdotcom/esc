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
