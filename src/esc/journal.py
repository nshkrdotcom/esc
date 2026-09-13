"""Durable request-boundary records independent of DSPy callback/history timing."""
import json
import os
from pathlib import Path
from threading import Lock


class RequestJournal:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        # Refuse an existing file even when constructing this outside the runner.
        with path.open("x"):
            pass

    def for_episode(self, episode):
        identity = dict(episode)

        def write(record):
            with self.lock, self.path.open("a") as handle:
                handle.write(json.dumps({**identity, **record}, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

        return write
