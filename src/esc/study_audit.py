"""Independent request replay for complete frozen study batches."""
from collections import defaultdict
import json
from pathlib import Path
from esc.study import load_plan
from esc.core.answers import answers_equal


def replay(events, state, budget, model):
    pending, requested, seen = {}, {}, set()
    prompt = completion = blocked = 0
    exhausted = False
    for index, event in enumerate(events, 1):
        if event['sequence'] != index:
            raise ValueError('Journal sequence gap')
        kind = event['event']
        if kind == 'request_start':
            key, cap, generation = event['request_id'], event['capped_generation'], event['requested_generation']
            remaining = budget - prompt - completion - sum(pending.values())
            if (exhausted or key in seen or type(cap) is not int or cap < 1
                or type(generation) is not int or generation < 1 or event['model'] != model
                or cap != min(remaining, generation)):
                raise ValueError('Invalid dispatch')
            seen.add(key)
            pending[key], requested[key] = cap, generation
        elif kind == 'request_end':
            key = event['request_id']
            if key not in pending or event['outcome'] != 'measured':
                raise ValueError('Unmatched response or unknown consumption')
            cap = pending.pop(key)
            p, c = event['prompt_tokens'], event['completion_tokens']
            if any(type(v) is not int or v < 0 for v in (p,c)) or p+c == 0 or c > cap:
                raise ValueError('Invalid provider usage')
            truncated = cap < requested[key] and 'length' in event['finish_reasons']
            prompt += p
            completion += c
            exhausted = exhausted or prompt+completion > budget or truncated
            if truncated != event['budget_truncated'] or exhausted != event['exhausted']:
                raise ValueError('Incorrect exhaustion state')
        elif kind == 'request_blocked':
            if event['reason'] != 'budget_exhausted' or (
                not exhausted and budget-prompt-completion-sum(pending.values()) > 0):
                raise ValueError('Invalid blocked request')
            blocked += 1
            exhausted = True
        else:
            raise ValueError('Unknown event')
    total = prompt+completion
    expected = dict(budget=budget, prompt_tokens=prompt, completion_tokens=completion,
        measured_tokens=total, overshoot=max(0,total-budget), remaining=max(0,budget-total),
        pending_reserved=0, unknown_reserved=0, accounting_failed=False,
        calls_dispatched=len(seen), calls_blocked=blocked, exhausted=exhausted,
        policy='soft_generation_reservation_v1')
    if pending or any(state.get(k) != v for k,v in expected.items()):
        raise ValueError('Journal differs from settled ledger')


def audit_study(directory):
    path = Path(directory)
    plan = load_plan(path)
    if (path/'failure.json').exists():
        raise ValueError('Failed batch is not analyzable')
    complete = json.loads((path/'complete.json').read_text())
    expected = {e['episode_id']:e for e in plan['episodes']}
    if len(expected) != len(plan['episodes']) or complete != dict(plan_hash=plan['plan_hash'],episodes=len(expected)):
        raise ValueError('Invalid completion marker or duplicated planned episode')
    runs = [json.loads(line) for line in (path/'runs.jsonl').read_text().splitlines()]
    events = defaultdict(list)
    roles = defaultdict(int)
    for line in (path/'requests.jsonl').read_text().splitlines():
        event = json.loads(line)
        key = event['episode_id']
        if key not in expected or any(event.get(k) != v for k,v in expected[key].items()):
            raise ValueError('Orphaned or misattributed request')
        events[key].append(event)
        if event['event'] == 'request_start':
            role = event.get('call_role') or 'unclassified'
            if plan.get('request_roles') and role not in {'root','recursive'}:
                raise ValueError('Study request has no declared root/recursive role')
            roles[role] += 1
    seen = set()
    tasks = {t['task_id']:t for t in plan['tasks']}
    for run in runs:
        key = run['episode']['episode_id']
        if key in seen or run['episode'] != expected.get(key):
            raise ValueError('Duplicate/unplanned result')
        seen.add(key)
        task = tasks[run['episode']['task_id']]
        if run['truth'] != task['ground_truth_map'] or run['depth'] != task['depth']:
            raise ValueError('Task labels/depth differ from frozen plan')
        if (run['world_id'] != task['metadata']['world_id'] or
            run['correct'] != answers_equal(run['answer'], task['ground_truth_map'][task['final_node_id']]) or
            run['abstained'] != (run['answer'] is None)):
            raise ValueError('Scoring/identity differs from frozen task')
        if run['variant'] != run['episode']['variant'] or run['site'] != run['episode']['site']:
            raise ValueError('Variant/intervention differs from schedule')
        if run['task_id'] != run['episode']['task_id']:
            raise ValueError('Result task differs from schedule')
        replay(events.pop(key, []), run['ledger'], run['episode']['budget'], plan['configuration']['model'])
    if seen != set(expected) or events:
        raise ValueError('Incomplete batch')
    return dict(valid=True, episodes=len(runs), measured_tokens=sum(r['ledger']['measured_tokens'] for r in runs),
                compute_matched=False, call_roles=dict(roles), plan_hash=plan['plan_hash']), plan, runs
