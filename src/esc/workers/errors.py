"""Model-output failures are unsuccessful attempts, not transport failures."""


class ModelOutputError(ValueError):
    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage
