"""Shared provider error types for RD-TableBench."""


class ProviderRequestError(RuntimeError):
    """A remote provider request failed after its retry policy."""