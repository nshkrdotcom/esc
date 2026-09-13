"""Soft episode allowance: generation reservations, measured prompt reconciliation.

No preflight prompt tokenizer and no hard total-token or overshoot bound is claimed.
DSPy copies context into recursive threads; the ledger reference must stay shared.
"""
from threading import Lock
from collections.abc import Mapping
import dspy


class EpisodeBudgetExhausted(Exception):
    pass


class BudgetAccountingError(RuntimeError):
    pass


class EpisodeLedger:
    def __init__(self, budget: int):
        if type(budget) is not int or budget < 1:
            raise ValueError("Episode budget must be a positive integer")
        self.budget = budget
        self.lock = Lock()
        self.pending = {}
        self.next_id = 0
        self.prompt = self.completion = self.calls = self.blocked = 0
        self.unknown_reserved = 0
        self.exhausted = False
        self.failed = False

    def reserve(self, generation):
        if type(generation) is not int or generation < 1:
            raise ValueError("Generation allowance must be a positive integer")
        with self.lock:
            if self.failed:
                raise BudgetAccountingError("Prior request has unknown consumption")
            remaining = self.budget - self.prompt - self.completion - sum(self.pending.values())
            if self.exhausted or remaining < 1:
                self.exhausted = True
                self.blocked += 1
                raise EpisodeBudgetExhausted("Episode soft token allowance exhausted")
            cap = min(generation, remaining)
            self.next_id += 1
            self.pending[self.next_id] = cap
            self.calls += 1
            return self.next_id, cap

    def commit(self, handle, usage, budget_truncated=False):
        with self.lock:
            cap = self.pending.pop(handle)
            values = ([usage.get(k) for k in ("prompt_tokens", "completion_tokens")]
                      if isinstance(usage, Mapping) else [None, None])
            if any(type(v) is not int or v < 0 for v in values) or sum(values) == 0:
                self.failed = True
                self.unknown_reserved += cap
                raise BudgetAccountingError("Missing or invalid provider token usage")
            self.prompt += values[0]
            self.completion += values[1]
            if self.prompt + self.completion > self.budget or budget_truncated:
                self.exhausted = True

    def fail(self, handle):
        with self.lock:
            self.unknown_reserved += self.pending.pop(handle)
            self.failed = True

    def check(self):
        with self.lock:
            if self.failed:
                raise BudgetAccountingError("Request failed; consumption is unknown")
            if self.exhausted:
                raise EpisodeBudgetExhausted("Episode soft token allowance exhausted")

    def snapshot(self):
        with self.lock:
            spent = self.prompt + self.completion
            return dict(policy="soft_generation_reservation_v1", budget=self.budget,
                        prompt_tokens=self.prompt, completion_tokens=self.completion,
                        measured_tokens=spent, overshoot=max(0, spent-self.budget),
                        remaining=max(0, self.budget-spent-sum(self.pending.values())),
                        pending_reserved=sum(self.pending.values()),
                        unknown_reserved=self.unknown_reserved, calls_dispatched=self.calls,
                        calls_blocked=self.blocked, exhausted=self.exhausted,
                        accounting_failed=self.failed)


def current_ledger():
    return dspy.settings.get("esc_ledger")


class BudgetedLM(dspy.LM):
    @staticmethod
    def _complete(ledger, handle, response, reduced):
        # Parsing provider metadata can fail before commit. Mark that request's
        # consumption unknown too, so the REPL cannot hide the accounting error.
        try:
            usage = dict(response.get("usage") or {})
            truncated = reduced and any(
                c.get("finish_reason") == "length" for c in (response.get("choices") or [])
            )
        except Exception as exc:
            ledger.fail(handle)
            raise BudgetAccountingError("Malformed provider usage/response metadata") from exc
        ledger.commit(handle, usage, truncated)
        ledger.check()

    def _reserve(self, kwargs):
        ledger = current_ledger()
        if ledger is None:
            return None, None, False
        if kwargs.get("cache", self.cache) or self.num_retries:
            raise BudgetAccountingError("Budgeted calls require cache=False and num_retries=0")
        requested = kwargs.get("max_tokens", self.kwargs["max_tokens"])
        handle, cap = ledger.reserve(requested)
        kwargs["max_tokens"] = cap
        return ledger, handle, cap < requested

    def forward(self, prompt=None, messages=None, **kwargs):
        ledger, handle, reduced = self._reserve(kwargs)
        try:
            response = super().forward(prompt=prompt, messages=messages, **kwargs)
        except BaseException:
            if ledger:
                ledger.fail(handle)
            raise
        if ledger:
            self._complete(ledger, handle, response, reduced)
        return response

    async def aforward(self, prompt=None, messages=None, **kwargs):
        ledger, handle, reduced = self._reserve(kwargs)
        try:
            response = await super().aforward(prompt=prompt, messages=messages, **kwargs)
        except BaseException:
            if ledger:
                ledger.fail(handle)
            raise
        if ledger:
            self._complete(ledger, handle, response, reduced)
        return response
