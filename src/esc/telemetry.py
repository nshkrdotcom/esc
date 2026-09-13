"""Flush call and interpreter diagnostics while an episode is in progress."""
import json
import os
import threading
import time
from pathlib import Path

from dspy.utils.callback import BaseCallback


class RunTelemetry(BaseCallback):
    def __init__(self, path: Path):
        self.path = path
        self.episode = {}
        self._calls = {}
        self._lock = threading.Lock()

    def record(self, event, **fields):
        with self._lock:
            with self.path.open("a") as handle:
                handle.write(json.dumps({"time": time.time(), **self.episode,
                                         "event": event, **fields}, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def on_lm_start(self, call_id, instance, inputs):
        self._calls[call_id] = (instance, time.monotonic(), len(instance.history))
        self.record("lm_start", call_id=call_id, model=instance.model)

    def on_lm_end(self, call_id, outputs, exception=None):
        instance, started, history_size = self._calls.pop(call_id)
        entry = instance.history[-1] if len(instance.history) > history_size else {}
        response = entry.get("response")
        reasons = [choice.finish_reason for choice in getattr(response, "choices", [])]
        self.record("lm_end", call_id=call_id, seconds=time.monotonic() - started,
                    outputs=outputs, usage=entry.get("usage"), finish_reasons=reasons,
                    error=str(exception) if exception else None)

    def on_interpreter_execute_end(self, call_id, outputs, exception=None):
        self.record("interpreter_output", call_id=call_id, output=str(outputs)[:12000],
                    error=str(exception) if exception else None)
