"""Tests for ts_triage.exceptions."""

import pytest
from ts_triage.exceptions import TSInforecastableError

class TestTSInforecastableError:
    def test_initialization(self):
        reason = "low_spectral_score"
        value = 0.03
        threshold = 0.05
        exc = TSInforecastableError(reason, value, threshold)

        assert exc.reason == reason
        assert exc.value == value
        assert exc.threshold == threshold
        assert isinstance(exc, Exception)

    def test_message_formatting(self):
        exc = TSInforecastableError("too_sparse", 0.98, 0.95)
        expected_msg = "Unforecastable series: too_sparse (value=0.9800, threshold=0.9500)"
        assert str(exc) == expected_msg

    def test_message_formatting_precision(self):
        # Test that it rounds/formats to 4 decimal places
        exc = TSInforecastableError("test", 0.123456, 0.5)
        assert "value=0.1235" in str(exc)
        assert "threshold=0.5000" in str(exc)
