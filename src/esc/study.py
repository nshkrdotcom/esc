"""Frozen, randomized study plans and durable live execution."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time
import urllib.request
import zipfile
import platform
import subprocess
from importlib.metadata import version
import dspy
from pydantic import BaseModel, ConfigDict, Field, model_validator
from esc.benchmark.relational import generate_relational_suite
from esc.benchmark.tasks import EpiDAGTask
from esc.budget import EpisodeLedger
from esc.config import DEFAULT_MODEL, RLMConfig, task_lm
from esc.journal import RequestJournal
from esc.study_systems import VARIANTS, run_variant


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    split: str = 'dev'
    seed: int = 80
    schedule_seed: int = 901
    worlds: int = Field(default=2, ge=1)
    repetitions: int = Field(default=5, ge=1)
    depths: list[int] = [2, 4, 8, 16]
    budgets: list[int] = [24000, 48000]
    variants: list[str] = list(VARIANTS)
    interventions: bool = True
    intervention_mode: str = 'candidate'
    intervention_sites: list[int] = [1]
    n_samples: int = Field(default=3, ge=1)
    model: str = DEFAULT_MODEL
    rlm: RLMConfig = RLMConfig()
    max_tokens: int = Field(default=1024, ge=1)
    num_ctx: int = Field(default=8192, ge=1024)
    bootstrap_samples: int = Field(default=1000, ge=100)

    @model_validator(mode='after')
    def validate_plan(self):
        if self.split not in {'dev', 'test'}:
            raise ValueError('Study split must be dev or test')
        for values in (self.depths, self.budgets, self.variants):
            if not values or len(values) != len(set(values)):
                raise ValueError('Empty or duplicated plan dimension')
        if any(d not in {2, 4, 8, 16} for d in self.depths) or any(b < 1 for b in self.budgets):
            raise ValueError('Invalid depth or budget')
        if set(self.variants) - set(VARIANTS):
            raise ValueError('Unknown variant')
        if self.intervention_mode not in {'candidate', 'accepted_state'}:
            raise ValueError('Unknown intervention boundary')
        if (not self.intervention_sites or len(set(self.intervention_sites)) != len(self.intervention_sites)
            or any(s < 1 or s >= min(self.depths) for s in self.intervention_sites)):
            raise ValueError('Distinct intervention sites must precede the shallowest final node')
        if self.split == 'test' and (self.worlds < 2 or self.repetitions < 5):
            raise ValueError('Held-out analysis requires >=2 worlds and >=5 repeats')
        if not self.model.startswith('ollama_chat/') or self.num_ctx <= self.max_tokens:
            raise ValueError('Frozen study currently supports local Ollama with context headroom')
        return self


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root/'src').rglob('*.py')) + [root/'uv.lock', root/'pyproject.toml']
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def runtime_identity():
    return dict(python=platform.python_version(), dspy=version('dspy'),
                deno=subprocess.check_output(['deno','--version'],text=True).strip())


def model_identity(model):
    def request(endpoint, payload=None):
        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request('http://localhost:11434/api/'+endpoint, data=data,
                                     headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.load(response)
    name = model.split('/', 1)[1]
    matches = [m for m in request('tags')['models'] if m['name'] == name]
    if len(matches) != 1:
        raise ValueError('Exact local model alias missing')
    show = request('show', {'model':name})
    return dict(name=name, digest=matches[0]['digest'], template=show.get('template'),
                parameters=show.get('parameters'), model_info=show.get('model_info'),
                ollama_version=request('version')['version'])


def freeze_study(directory, config: StudyConfig):
    path = Path(directory)
    if path.exists():
        raise FileExistsError('Use a new study directory')
    identity = model_identity(config.model)
    tasks = generate_relational_suite(depths=config.depths, tasks_per_depth=config.worlds,
                                     seed=config.seed, split=config.split)
    episodes = []
    for task, budget, variant, rep in itertools.product(tasks, config.budgets, config.variants,
                                                       range(config.repetitions)):
        for site in ([None, *config.intervention_sites] if config.interventions and variant != 'continuous' else [None]):
            episode = dict(task_id=task.task_id, variant=variant, budget=budget, repetition=rep, site=site)
            episode['episode_id'] = digest(episode)[:24]
            episodes.append(episode)
    random.Random(config.schedule_seed).shuffle(episodes)
    plan = dict(schema='esc_study_v1', configuration=config.model_dump(), source_identity=source_identity(),
        model_identity=identity, tasks=[t.model_dump() for t in tasks], episodes=episodes,
        cost_policy='soft_generation_reservation_v1', primary_analysis='quality_vs_measured_cost_v1',
        compute_matched=False, helper='all_public_rows_v1', intervention='transition_receipt_v2',
        request_roles='root_recursive_v1',
        submission_policy='in_context_receipt_guard_v1',
        caveat='Exploratory relational study; no equal-compute H1 claim')
    plan['plan_hash'] = digest(plan)
    path.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(path/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in sorted((root/'src').rglob('*.py')) + [root/'uv.lock', root/'pyproject.toml']:
            archive.write(source, str(source.relative_to(root)))
    # Archive the actual source, not merely a potentially dirty Git revision.
    plan.pop('plan_hash')
    plan['source_archive_sha256'] = hashlib.sha256((path/'source.zip').read_bytes()).hexdigest()
    plan['runtime'] = runtime_identity()
    plan['plan_hash'] = digest(plan)
    (path/'plan.json').write_text(json.dumps(plan, indent=2))
    return plan


def load_plan(path):
    plan = json.loads((Path(path)/'plan.json').read_text())
    if digest({k:v for k,v in plan.items() if k != 'plan_hash'}) != plan['plan_hash']:
        raise ValueError('Frozen plan has changed')
    if 'source_archive_sha256' in plan and hashlib.sha256((Path(path)/'source.zip').read_bytes()).hexdigest() != plan['source_archive_sha256']:
        raise ValueError('Frozen source archive has changed')
    if 'source_archive_sha256' in plan:
        with zipfile.ZipFile(Path(path)/'source.zip') as archive:
            if digest({name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}) != plan['source_identity']:
                raise ValueError('Source archive does not match the frozen source identity')
    return plan


def run_study(directory):
    path = Path(directory)
    plan = load_plan(path)
    config = StudyConfig.model_validate(plan['configuration'])
    if source_identity() != plan['source_identity'] or model_identity(config.model) != plan['model_identity']:
        raise ValueError('Code, lockfile, or served model differs from frozen plan; freeze a new plan')
    if runtime_identity() != plan.get('runtime'):
        raise ValueError('Python, DSPy, or Deno runtime differs from frozen plan')
    if any((path/name).exists() for name in ('runs.jsonl', 'requests.jsonl', 'complete.json', 'failure.json')):
        raise FileExistsError('Study already started; incomplete batches must not be silently resumed')
    lm = task_lm(config.model, max_tokens=config.max_tokens, num_ctx=config.num_ctx)
    lm.esc_role = 'root'
    tasks = {t['task_id']:EpiDAGTask.model_validate(t) for t in plan['tasks']}
    journal = RequestJournal(path/'requests.jsonl')
    with (path/'runs.jsonl').open('x') as handle:
        for index, episode in enumerate(plan['episodes']):
            print(f"Study {index+1}/{len(plan['episodes'])}: {episode}", flush=True)
            ledger = EpisodeLedger(episode['budget'], event_sink=journal.for_episode(episode))
            start = time.perf_counter()
            try:
                with dspy.context(lm=lm, esc_ledger=ledger,
                                  adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
                    result = run_variant(tasks[episode['task_id']], episode['variant'], lm, config.rlm,
                        site=episode['site'], seed=int(digest([episode['task_id'],episode['repetition']])[:12],16),
                        n_samples=config.n_samples, intervention_mode=config.intervention_mode)
                state = ledger.snapshot()
                if state['accounting_failed'] or state['pending_reserved'] or state['unknown_reserved']:
                    raise RuntimeError('Unsettled or unknown consumption invalidates study')
                result.update(episode=episode, ledger=state, elapsed_seconds=time.perf_counter()-start)
                handle.write(json.dumps(result, allow_nan=False)+'\n')
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException as exc:
                (path/'failure.json').write_text(json.dumps(dict(episode=episode,
                    error_type=type(exc).__name__, error=str(exc), ledger=ledger.snapshot()), indent=2))
                raise
    (path/'complete.json').write_text(json.dumps(dict(plan_hash=plan['plan_hash'], episodes=len(plan['episodes']))))
