"""Custom exceptions for ts_triage."""


class TSInforecastableError(Exception):
    """Raised when a series is determined to be unforecastable (strict=True)."""

    def __init__(self, reason: str, value: float, threshold: float) -> None:
        self.reason = reason
        self.value = value
        self.threshold = threshold
        super().__init__(
            f"Unforecastable series: {reason} (value={value:.4f}, threshold={threshold:.4f})"
        )
