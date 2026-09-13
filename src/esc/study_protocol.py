"""Public transition receipts and paired fault interventions, without labels."""
import hashlib
from threading import Lock
from esc.core.types import Evidence, Fact, StepResult, StepSpec
from esc.core.lookup import verify_lookup


class TransitionProtocol:
    def __init__(self, steps: list[StepSpec], corpus: dict[str, str], entities: list[str],
                 *, verify=False, typed=False, site=None, seed=0, intervention_mode='candidate'):
        if intervention_mode not in {'candidate', 'accepted_state'}:
            raise ValueError('Unknown intervention boundary')
        if not steps or any(s.lookup is None for s in steps):
            raise ValueError('Public relational steps are required')
        if site is not None and (type(site) is not int or not 1 <= site <= len(steps)):
            raise ValueError('Intervention site must be a reached task step')
        self.steps = [s.model_copy(deep=True) for s in steps]
        self.corpus = dict(corpus)
        self.entities = sorted(set(entities))
        self.verify = verify
        self.typed = typed or verify
        self.site = site
        self.seed = seed
        self.intervention_mode = intervention_mode
        self.events = []
        self.facts = []
        self.current = steps[0].lookup.subject
        self.stopped = False
        self.violations = []
        self.lock = Lock()

    def emit(self, step_id: str, input_value: str, value: str | None,
             evidence: list[dict[str, str]]) -> dict:
        """Emit ONE transition in order. Use returned value as the next subject.

        evidence entries contain source_id and span. A null value abstains.
        Stop when returned stop is true. Do not advance before this receipt.
        """
        with self.lock:
            if self.stopped or len(self.events) >= len(self.steps):
                self.violations.append('Emission after terminal state')
                return dict(value=None, stop=True)
            step = self.steps[len(self.events)]
            if step_id != step.step_id or input_value != self.current:
                self.violations.append('Wrong step order or ignored previous receipt')
                self.stopped = True
                return dict(value=None, stop=True)
            original = value
            at_site = self.site == len(self.events) + 1
            applied = at_site and value in self.entities and self.intervention_mode == 'candidate'
            def replacement(proposal):
                alternatives = [e for e in self.entities if e != proposal]
                index = int(hashlib.sha256(f'{self.seed}:{step_id}'.encode()).hexdigest(), 16)
                return alternatives[index % len(alternatives)]
            if applied:
                value = replacement(value)
            accepted = value is not None
            reason = None
            if self.verify and accepted:
                try:
                    result = StepResult(status='supported', value=value,
                                        evidence=[Evidence.model_validate(e) for e in evidence])
                    accepted, reason = verify_lookup(step, result, self.facts[-1:], self.corpus)
                except (ValueError, TypeError) as exc:
                    accepted, reason = False, str(exc)
            witness_value = value
            if at_site and self.intervention_mode == 'accepted_state' and accepted and value in self.entities:
                value = replacement(value)
                applied = True
            self.events.append(dict(step_id=step_id, input_value=input_value,
                proposed=original, value=value, evidence=evidence, applied=applied,
                accepted=accepted, reason=reason, intervention_mode=self.intervention_mode,
                witness_value=witness_value, reached_intervention=at_site))
            if accepted:
                self.current = value
                self.facts.append(Fact(key=step.step_id, value=value,
                    level='verified' if self.verify else 'candidate',
                    evidence=[Evidence.model_validate(e) for e in evidence] if self.typed else [],
                    parents=step.requires))
            else:
                self.stopped = True
            receipt = dict(value=value if accepted else None, stop=self.stopped)
            if self.verify and accepted:
                receipt['fact'] = self.facts[-1].model_dump()
            return receipt

    def finish(self, answer):
        """Reject a final output that bypasses required transition receipts."""
        if self.violations:
            return None
        if len(self.events) != len(self.steps) or self.stopped:
            if answer is not None:
                self.violations.append('Final answer bypassed incomplete or rejected transition')
            return None
        if answer != self.current:
            self.violations.append('Final answer differs from last receipt')
            return None
        return answer
