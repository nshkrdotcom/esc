"""Read-only validation of completed budgeted runs and their request journals."""
import json
from collections import defaultdict
from pathlib import Path


def audit_budget_run(directory):
    path = Path(directory)
    if (path / "failure.json").exists():
        raise ValueError("Failed batch: failure.json exists")
    manifest = json.loads((path / "manifest.json").read_text())
    summary = json.loads((path / "experiment_1_summary.json").read_text())
    tasks = json.loads((path / "tasks.json").read_text())
    runs = [json.loads(line) for line in (path / "runs.jsonl").read_text().splitlines()]
    records = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    if manifest.get("request_journal_version") != 1 or not manifest.get("episode_token_budget"):
        raise ValueError("Run has no supported request journal / episode allowance")
    if (manifest.get("use_mock") or len(tasks) != len(manifest["task_sizes"]) * manifest["tasks_per_depth"]
            or len({t["task_id"] for t in tasks}) != len(tasks)):
        raise ValueError("Invalid task manifest or simulated run")
    if summary["configuration"] != manifest:
        raise ValueError("Summary configuration differs from manifest")
    conditions = {"Condition_A_Continuous", "Condition_B_SearchHeavy", "Condition_C_Isolated"}
    expected = {(t["task_id"], system, rep) for t in tasks for system in conditions
                for rep in range(manifest["repetitions"])}
    grouped = defaultdict(list)
    for event in records:
        grouped[event["task_id"], event["system"], event["repetition"]].append(event)
    seen = set()
    reports = []
    for run in runs:
        key = (run["task_id"], run["system_name"], run["details"]["repetition"])
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicated episode: {key}")
        seen.add(key)
        state = run["details"]["ledger"]
        pending, started, requested = {}, set(), {}
        prompt = completion = blocked = 0
        exhausted = False
        budget = manifest["episode_token_budget"]
        for seq, event in enumerate(grouped.pop(key, []), 1):
            if event["sequence"] != seq:
                raise ValueError(f"Journal sequence gap: {key}")
            kind = event["event"]
            if kind == "request_start":
                request = event["request_id"]
                cap = event["capped_generation"]
                generation = event["requested_generation"]
                remaining = budget - prompt - completion - sum(pending.values())
                if (exhausted or request in started or type(cap) is not int or cap < 1
                        or type(generation) is not int or generation < 1
                        or event["model"] != manifest["model"]
                        or cap != min(generation, remaining)):
                    raise ValueError(f"Invalid dispatch/reservation: {key}")
                pending[request] = cap
                requested[request] = generation
                started.add(request)
            elif kind == "request_end":
                if event["request_id"] not in pending:
                    raise ValueError(f"Unmatched/duplicate response: {key}")
                cap = pending.pop(event["request_id"])
                if event["outcome"] != "measured":
                    raise ValueError(f"Unknown provider consumption: {key}")
                values = [event["prompt_tokens"], event["completion_tokens"]]
                if any(type(v) is not int or v < 0 for v in values) or sum(values) == 0:
                    raise ValueError(f"Invalid usage: {key}")
                if values[1] > cap:
                    raise ValueError(f"Provider exceeded generation cap: {key}")
                truncated = cap < requested[event["request_id"]] and "length" in event["finish_reasons"]
                if event["budget_truncated"] != truncated:
                    raise ValueError(f"Budget truncation mismatch: {key}")
                prompt += values[0]
                completion += values[1]
                exhausted = exhausted or prompt + completion > budget or event["budget_truncated"]
                if event["exhausted"] != exhausted:
                    raise ValueError(f"Exhaustion state mismatch: {key}")
            elif kind == "request_blocked":
                if event["reason"] != "budget_exhausted":
                    raise ValueError(f"Accounting failure: {key}")
                if not exhausted and budget - prompt - completion - sum(pending.values()) > 0:
                    raise ValueError(f"Premature budget rejection: {key}")
                blocked += 1
                exhausted = True
            else:
                raise ValueError(f"Unknown journal event: {kind}")
        total = prompt + completion
        checks = dict(budget=budget, prompt_tokens=prompt, completion_tokens=completion,
                      measured_tokens=total, overshoot=max(0, total-budget),
                      pending_reserved=0, unknown_reserved=0, calls_dispatched=len(started),
                      calls_blocked=blocked, exhausted=exhausted, accounting_failed=False,
                      remaining=max(0, budget-total))
        if pending or any(state.get(k) != v for k, v in checks.items()):
            raise ValueError(f"Journal/ledger mismatch: {key}")
        if run["tokens_used"] != total or run["budget_exhausted"] != exhausted:
            raise ValueError(f"Run/ledger mismatch: {key}")
        reports.append(dict(task_id=key[0], system=key[1], repetition=key[2],
                            depth=run["depth"], is_correct=run["is_correct"], abstained=run["abstained"],
                            model_output_error=run.get("model_output_error", False),
                            tokens=total, exhausted=exhausted, overshoot=max(0, total-budget)))
    if seen != expected or grouped:
        raise ValueError("Missing episodes or orphaned journal records")
    cost_by_depth = {}
    for depth in sorted({r['depth'] for r in reports}):
        rows = [r for r in reports if r['depth'] == depth]
        means = {system: sum(r['tokens'] for r in rows if r['system'] == system)
                 / sum(r['system'] == system for r in rows) for system in sorted(conditions)}
        lowest = min(means.values())
        cost_by_depth[depth] = dict(mean_tokens_by_system=means,
            largest_to_smallest_mean_ratio=max(means.values()) / lowest if lowest else None,
            maximum_episode_overshoot=max(r['overshoot'] for r in rows))
    return dict(valid=True, scope="Request accounting only; no hypothesis inference",
                compute_matched=False, cost_by_depth=cost_by_depth,
                episodes=len(reports), measured_tokens=sum(r["tokens"] for r in reports),
                exhausted_episodes=sum(r["exhausted"] for r in reports),
                model_output_error_episodes=sum(r["model_output_error"] for r in reports), runs=reports)
