import json
import dspy
import pytest
from esc.budget import EpisodeBudgetExhausted
from esc.study import StudyConfig, freeze_study, run_study, load_plan
from esc.study_audit import audit_study
from esc.study_analysis import analyze_study, decay, propagation


def prepare(tmp_path, monkeypatch, *, budget=1000):
    import esc.study as study
    monkeypatch.setattr(study, 'model_identity', lambda model: {'fixture':True})
    monkeypatch.setattr(dspy.LM, 'forward', lambda *a,**k: dict(
        usage={'prompt_tokens':20,'completion_tokens':3},choices=[{'finish_reason':'stop'}]))
    def variant(task, variant, lm, config, site=None, **kwargs):
        exhausted=False
        try: lm.forward(prompt='fixture')
        except EpisodeBudgetExhausted: exhausted=True
        return dict(variant=variant,task_id=task.task_id,world_id=task.metadata['world_id'],depth=task.depth,
            truth=dict(task.ground_truth_map),answer=None if exhausted else task.target_answer(),
            correct=not exhausted,abstained=exhausted,exhausted=exhausted,site=site,rollouts=[])
    monkeypatch.setattr(study,'run_variant',variant)
    path=tmp_path/'study'
    config=StudyConfig(split='test',worlds=2,repetitions=5,depths=[2,4],budgets=[budget],
                       variants=['continuous','isolated_verified'],bootstrap_samples=100)
    freeze_study(path,config)
    return path


@pytest.mark.parametrize('budget',[10,1000])
def test_frozen_execution_and_independent_audit(tmp_path,monkeypatch,budget):
    path=prepare(tmp_path,monkeypatch,budget=budget)
    run_study(path)
    audit,plan,runs=audit_study(path)
    assert audit['episodes']==60 and audit['measured_tokens']==1380
    assert all(r['ledger']['exhausted']==(budget==10) for r in runs)
    assert [r['episode'] for r in runs]==plan['episodes']
    assert (path/'source.zip').exists()
    with pytest.raises(FileExistsError): run_study(path)


@pytest.mark.parametrize('mutation',['missing','duplicate','score','journal','archive'])
def test_study_integrity_rejects_changed_or_partial_data(tmp_path,monkeypatch,mutation):
    path=prepare(tmp_path,monkeypatch)
    run_study(path)
    file=path/'runs.jsonl'
    rows=[json.loads(line) for line in file.read_text().splitlines()]
    if mutation=='missing': rows.pop()
    elif mutation=='duplicate': rows.append(rows[0])
    elif mutation=='score': rows[0]['correct']=False
    elif mutation=='archive': (path/'source.zip').write_bytes(b'changed')
    else:
        events=(path/'requests.jsonl').read_text().splitlines()
        (path/'requests.jsonl').write_text('\n'.join(events[1:])+'\n')
    file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): audit_study(path)


def test_provider_failure_persists_and_blocks_analysis(tmp_path,monkeypatch):
    path=prepare(tmp_path,monkeypatch)
    def fail(*a,**k): raise RuntimeError('backend down')
    monkeypatch.setattr(dspy.LM,'forward',fail)
    with pytest.raises(RuntimeError,match='backend down'): run_study(path)
    assert (path/'failure.json').exists() and not (path/'complete.json').exists()
    with pytest.raises(ValueError,match='Failed'): audit_study(path)


def test_analysis_produces_four_figures_and_honest_saturated_slopes(tmp_path,monkeypatch):
    path=prepare(tmp_path,monkeypatch)
    run_study(path)
    report=analyze_study(path)
    assert all(r['beta'] is None for r in report['decay'])
    assert all(r['pass_all_5']==1 for r in report['repeatability'])
    assert len(list((path/'analysis').glob('figure_*.png')))==4
    with pytest.raises(FileExistsError): analyze_study(path)


def test_binomial_decay_and_unidentifiable_cases():
    import math
    depths=[2,4,8,16]
    successes=[round(100000*.8*math.exp(-.05*d)) for d in depths]
    assert decay(depths,successes,[100000]*4)==pytest.approx(.05,abs=.001)
    assert decay([2],[5],[10]) is None
    assert decay(depths,[0]*4,[10]*4) is None


def test_protocol_violation_cannot_be_presented_as_zero_epc():
    run=dict(variant='continuous_emit',site=1,depth=2,truth={'N1':'right','N2':'right'},
             episode={'episode_id':'x','budget':10},rollouts=[dict(
                 events=[dict(step_id='N1',applied=True,value='wrong')],
                 protocol_violations=['ignored receipt'])])
    points,attempts=propagation([run])
    assert points[0]['epc'] is None and points[0]['unreached']==1
    assert attempts[0]['applied']


def test_holdout_requires_repeated_worlds_and_valid_dimensions():
    with pytest.raises(ValueError): StudyConfig(split='test',worlds=1)
    with pytest.raises(ValueError): StudyConfig(depths=[2,2])
    with pytest.raises(ValueError): StudyConfig(budgets=[True])
