import copy
import dspy
import pytest
from esc.benchmark.relational import generate_relational_suite
from esc.benchmark.public_rows import parse_documents
from esc.study_protocol import TransitionProtocol
from esc.study_systems import run_variant, VARIANTS
from esc.config import RLMConfig


def task():
    return generate_relational_suite(depths=[2], tasks_per_depth=1, width=4)[0]


def protocol(t, **kwargs):
    rows = parse_documents(t.get_corpus().full_context())
    return TransitionProtocol([n.step_spec for n in t.nodes], t.corpus_data,
                              [r['subject'] for r in rows], **kwargs)


def test_parser_preserves_every_source_and_exact_span():
    t = task()
    rows = parse_documents(t.get_corpus().full_context())
    assert len(rows) == 64
    assert {r['source_id'] for r in rows} == set(t.corpus_data)
    for row in rows:
        assert row['span'] in t.corpus_data[row['source_id']].splitlines()


def test_same_candidate_fault_is_rejected_only_with_witness():
    t = task()
    n = t.nodes[0]
    plain, verified = protocol(t, site=1), protocol(t, site=1, verify=True)
    args = (n.node_id, n.step_spec.lookup.subject, n.true_value,
            [e.model_dump() for e in n.true_evidence])
    a, c = plain.emit(*args), verified.emit(*args)
    assert a['value'] != n.true_value and not a['stop']
    assert c == dict(value=None, stop=True)
    assert plain.events[0]['value'] == verified.events[0]['value']
    assert plain.events[0]['applied'] and verified.events[0]['applied']


def test_ignored_receipt_is_a_protocol_violation_not_valid_containment():
    t = task()
    p = protocol(t, site=1)
    n = t.nodes[0]
    p.emit('N1', n.step_spec.lookup.subject, n.true_value, [])
    assert p.emit('N2', n.true_value, t.nodes[1].true_value, [])['stop']
    assert p.violations and p.finish(t.target_answer()) is None


def test_accepted_state_corruption_is_distinct_from_candidate_rejection():
    t=task()
    p=protocol(t,site=1,verify=True,intervention_mode='accepted_state')
    n=t.nodes[0]
    receipt=p.emit(n.node_id,n.step_spec.lookup.subject,n.true_value,[e.model_dump() for e in n.true_evidence])
    assert not receipt['stop'] and receipt['value'] != n.true_value
    assert p.events[0]['witness_value']==n.true_value
    assert p.events[0]['accepted'] and p.events[0]['applied']
    assert receipt['fact']['value']==receipt['value']


def test_typed_unverified_state_preserves_evidence_without_promoting():
    t=task()
    p=protocol(t,typed=True)
    n=t.nodes[0]
    p.emit(n.node_id,n.step_spec.lookup.subject,n.true_value,[e.model_dump() for e in n.true_evidence])
    assert p.facts[0].level=='candidate'
    assert p.facts[0].evidence==n.true_evidence


def test_search_keeps_completed_vote_when_next_rollout_exhausts():
    from esc.budget import EpisodeBudgetExhausted
    t=task()
    count=[]
    class Script:
        def __init__(self, signature, tools, **kwargs):
            self.tools={f.__name__:f for f in tools}
        def forward(self, **inputs):
            count.append(1)
            if len(count)>1: raise EpisodeBudgetExhausted('fixture')
            dspy.settings.usage_tracker.add_usage('fixture',{'prompt_tokens':3,'completion_tokens':2})
            subject=t.nodes[0].step_spec.lookup.subject
            for n in t.nodes:
                row=next(r for r in self.tools['read_rows']() if r['subject']==subject and r['relation']==n.step_spec.lookup.relation)
                receipt=self.tools['emit'](n.node_id,subject,row['object'],[dict(source_id=row['source_id'],span=row['span'])])
                subject=receipt['value']
            return dspy.Prediction(answer=subject)
    result=run_variant(t,'search_emit',None,RLMConfig(),factory=Script)
    assert result['correct'] and result['exhausted'] and not result['abstained']
    assert len(result['rollouts'])==2 and result['selected_rollout']==0


@pytest.mark.parametrize('variant', VARIANTS)
def test_public_only_scripted_architectures_and_boundaries(variant):
    t = task()
    expected = t.target_answer()
    t.ground_truth_map = {k:'poison' for k in t.ground_truth_map}
    for n in t.nodes:
        n.true_value, n.true_evidence = 'poison', []
    invocations = []
    class Script:
        def __init__(self, signature, tools, **kwargs):
            self.tools = {f.__name__: f for f in tools}
        def forward(self, **inputs):
            invocations.append(copy.deepcopy(inputs))
            rows = self.tools['read_rows']()
            def lookup(subject, relation):
                return next(r for r in rows if r['subject']==subject and r['relation']==relation)
            dspy.settings.usage_tracker.add_usage('fixture', {'prompt_tokens':3,'completion_tokens':2})
            if 'task' in inputs:
                subject = t.nodes[0].step_spec.lookup.subject
                for n in t.nodes:
                    row = lookup(subject, n.step_spec.lookup.relation)
                    subject = row['object']
                    if 'emit' in self.tools:
                        receipt = self.tools['emit'](n.node_id, row['subject'], subject,
                            [dict(source_id=row['source_id'], span=row['span'])])
                        subject = receipt['value']
                        if receipt['stop']: break
                return dspy.Prediction(answer=subject, trajectory=['private thoughts'])
            step = inputs['step']
            subject = inputs.get('state') or (inputs['facts'][0].value if inputs['facts'] else step['lookup']['subject'])
            row = lookup(subject, step['lookup']['relation'])
            return dspy.Prediction(answer=row['object'], result=dict(status='supported',value=row['object'],
                evidence=[dict(source_id=row['source_id'],span=row['span'])]), trajectory=['private thoughts'])
    result = run_variant(t, variant, None, RLMConfig(), factory=Script)
    assert result['answer'] == expected and not result['correct']
    assert all('poison' not in str(i) for i in invocations)
    assert len(invocations) == (3 if variant=='search_emit' else 2 if variant.startswith('isolated') else 1)
    if variant.startswith('isolated'):
        assert 'private thoughts' not in str(invocations[1])
        assert ('state' in invocations[1]) == (variant=='isolated_raw')
        assert ('facts' in invocations[1]) == (variant!='isolated_raw')
