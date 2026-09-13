"""Explicit study architectures; legacy pilot systems remain reproducible."""
from collections import Counter
import dspy
from pydantic import ValidationError
from esc.benchmark.public_rows import parse_documents
from esc.core.answers import answers_equal, vote_key
from esc.core.types import Fact, StepResult
from esc.study_protocol import TransitionProtocol
from esc.workers.usage import invoke_with_usage
from esc.workers.errors import ModelOutputError
from esc.budget import BudgetedLM, EpisodeBudgetExhausted

VARIANTS = ('continuous', 'continuous_emit', 'search_emit', 'isolated_raw',
            'isolated_typed', 'shared_verified', 'isolated_verified')


class WholeTask(dspy.Signature):
    """Solve the public task. read_rows() returns ALL rows, including source_id/span.
    Assign its result to a variable and filter by subject and relation yourself.
    Return the final entity, or null if unable to solve. Follow protocol exactly.
    When the emit tool is available, calling it is mandatory for EVERY transition:
    receipt = emit(step_id, input_value, proposed_value, evidence).
    Use receipt['value'] as the next subject, even if it differs from your proposal.
    If receipt['stop'] is true, return null immediately. Never compute the next
    transition before receiving this receipt. Your final answer must equal the
    last receipt value. Skipping emit invalidates the episode. Evidence entries
    must contain source_id and span from the selected public row.
    Without an emit tool, solve the chain directly.
    """
    task: str = dspy.InputField()
    protocol: str = dspy.InputField()
    answer: str | None = dspy.OutputField()


class RawStep(dspy.Signature):
    """Resolve one lookup. read_rows() returns ALL public rows with source_id/span.
    Select by the input entity and public relation. Return the object, or null.
    """
    step: dict = dspy.InputField()
    state: str = dspy.InputField()
    answer: str | None = dspy.OutputField()


class TypedStep(dspy.Signature):
    """Resolve exactly one lookup from supplied facts and the public step.
    read_rows() returns ALL public rows with source_id/span. Select by subject and
    relation, and copy the selected row's source_id and exact span into evidence.
    Return insufficient instead of inventing a missing value.
    """
    step: dict = dspy.InputField()
    facts: list[Fact] = dspy.InputField()
    result: StepResult = dspy.OutputField()


PROTOCOL = ('For every public step, call emit(step_id, input_value, value, evidence) '
            'before advancing. evidence is a list of {source_id, span}. Use the RETURNED '
            'value as the next input_value, even if it differs from your proposal. '
            'If stop is true, return null. Return the last receipt value as answer. '
            'Keep all reasoning and interpreter state in this invocation.')


def run_variant(task, variant, lm, config, *, site=None, seed=0, n_samples=3,
                intervention_mode='candidate', factory=dspy.RLM):
    if variant not in VARIANTS:
        raise ValueError('Unknown study variant')
    if variant == 'continuous' and site is not None:
        raise ValueError('Uninstrumented baseline cannot receive a transition intervention')
    if any(n.step_spec.lookup is None for n in task.nodes):
        raise ValueError('Study systems require public relational contracts')
    rows = parse_documents(task.get_corpus().full_context())
    recursive_lm = lm
    if isinstance(lm, BudgetedLM):
        # Runtime copies preserve provider/model/sampling parameters. Role is local
        # metadata, never a provider request kwarg or a separate episode budget.
        lm.esc_role = 'root'
        recursive_lm = lm.copy()
        recursive_lm.esc_role = 'recursive'
    entities = sorted({r['subject'] for r in rows} | {r['object'] for r in rows})
    rollouts = []
    answers = []
    exhausted = False
    for rollout in range(n_samples if variant == 'search_emit' else 1):
        protocol = TransitionProtocol([n.step_spec for n in task.nodes], task.corpus_data,
            entities, verify=variant in {'shared_verified', 'isolated_verified'},
            typed=variant in {'isolated_typed', 'shared_verified', 'isolated_verified'},
            site=site, seed=seed + rollout, intervention_mode=intervention_mode)
        reads = []
        traces = []
        inputs_log = []
        error = None
        answer = None

        def read_rows() -> list[dict[str, str]]:
            """Return ALL public rows with subject/relation/object/source_id/span.
            Assign to a variable, filter it yourself, and retain provenance.
            """
            reads.append(dict(documents=len(task.corpus_data), rows=len(rows)))
            return [dict(row) for row in rows]

        def call(signature, inputs, tools):
            # New module AND interpreter per isolated transition. Shared variants
            # call this exactly once, retaining the RLM's native history/REPL.
            inputs_log.append(inputs)
            worker = factory(signature, sub_lm=recursive_lm, tools=tools, **config.model_dump())
            prediction, usage = invoke_with_usage(worker, **inputs)
            traces.append(getattr(prediction, 'trajectory', None))
            return prediction

        try:
            if variant in {'continuous', 'continuous_emit', 'search_emit', 'shared_verified'}:
                tools = [read_rows] if variant == 'continuous' else [read_rows, protocol.emit]
                pred = call(WholeTask, dict(task=task.public_description(),
                    protocol='No emission protocol; solve directly.' if variant == 'continuous' else PROTOCOL), tools)
                answer = pred.answer
                if answer is not None and not isinstance(answer, str):
                    raise ModelOutputError('Final answer must be a string or null')
                if variant != 'continuous':
                    answer = protocol.finish(answer)
            else:
                for node in task.nodes:
                    step = node.step_spec.model_copy(deep=True)
                    if variant == 'isolated_typed':
                        step.required_level = 'candidate'
                    if variant == 'isolated_raw':
                        public_operation = dict(step_id=step.step_id, lookup=step.lookup.model_dump())
                        pred = call(RawStep, dict(step=public_operation, state=str(protocol.current)), [read_rows])
                        value, evidence = pred.answer, []
                        if value is not None and not isinstance(value, str):
                            raise ModelOutputError('Raw step answer must be a string or null')
                    else:
                        pred = call(TypedStep, dict(step=step.model_dump(),
                            facts=[f.model_copy(deep=True) for f in protocol.facts[-1:]]), [read_rows])
                        result = StepResult.model_validate(pred.result)
                        value = result.value if result.status == 'supported' else None
                        evidence = [e.model_dump() for e in result.evidence]
                    receipt = protocol.emit(step.step_id, protocol.current, value, evidence)
                    if receipt['stop']:
                        break
                answer = protocol.finish(protocol.current if not protocol.stopped else None)
        except EpisodeBudgetExhausted:
            exhausted = True
            answer = None
        except (ModelOutputError, ValidationError) as exc:
            error = str(exc)
            answer = None
        # No provider/accounting catch: unknown cost must invalidate the batch.
        rollouts.append(dict(answer=answer, events=protocol.events,
            protocol_violations=protocol.violations, model_output_error=error,
            reads=reads, trajectories=traces,
            inputs=[{k: [f.model_dump() for f in v] if k == 'facts' else v
                     for k, v in inp.items()} for inp in inputs_log],
            exhausted=exhausted))
        if answer is not None:
            answers.append(vote_key(answer))
        if exhausted:
            break
    selected = Counter(answers).most_common(1)[0][0] if answers else None
    # Evaluator-only scoring is deliberately after all model/tool execution.
    return dict(variant=variant, task_id=task.task_id, world_id=task.metadata['world_id'],
        depth=task.dependency_depth, answer=selected, correct=answers_equal(selected, task.target_answer()),
        abstained=selected is None, exhausted=exhausted, site=site, intervention_mode=intervention_mode,
        selected_rollout=next((i for i,r in enumerate(rollouts) if r['answer'] is not None
                               and vote_key(r['answer']) == selected), None),
        rollouts=rollouts,
        truth=dict(task.ground_truth_map))
