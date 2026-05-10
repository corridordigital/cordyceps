"""Tests for frequency-to-period mapping in ts_triage.pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from ts_triage.pipeline import _freq_to_period, triage_pipeline


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("S", 3600),
        ("T", 60),
        ("min", 60),
        ("H", 24),
        ("D", 7),
        ("W", 52),
        ("M", 12),
        ("MS", 12),
        ("ME", 12),
        ("Q", 4),
        ("QS", 4),
        ("QE", 4),
        ("A", 1),
        ("AS", 1),
        ("AE", 1),
        ("Y", 1),
        ("YS", 1),
        ("YE", 1),
        ("B", 5),
        ("BM", 12),
    ],
)
def test_freq_to_period_exact_matches(freq, expected):
    """Test exact matches in the mapping dictionary."""
    assert _freq_to_period(freq) == expected


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("2D", 7),
        ("3M", 12),
        ("52W", 52),
        ("12H", 24),
        ("3QS", 4),
        ("-M", 12),  # lstrip handles '-' as well
    ],
)
def test_freq_to_period_with_multipliers(freq, expected):
    """Test that multipliers are stripped correctly."""
    assert _freq_to_period(freq) == expected


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("monthly", 12),
        ("Daily", 7),
        ("WEEKLY", 52),
        ("h", 24),
    ],
)
def test_freq_to_period_prefix_matches(freq, expected):
    """Test prefix-based matching (case-insensitive)."""
    assert _freq_to_period(freq) == expected


@pytest.mark.parametrize(
    "freq",
    [
        "UNKNOWN",
        "XYZ",
        "123",
        "",
        None,
    ],
)
def test_freq_to_period_fallbacks(freq):
    """Test that unknown or empty frequencies fallback to a period of 12."""
    assert _freq_to_period(freq) == 12


def test_triage_pipeline_unknown_freq_fallback():
    """Verify triage_pipeline handles unknown frequency strings gracefully."""
    # Create a mock series with a DatetimeIndex
    y = MagicMock(spec=pd.Series)
    y.index = MagicMock(spec=pd.DatetimeIndex)

    # We need to mock the internal components of triage_pipeline to avoid
    # execution of complex logic and dependencies during this unit test.
    # Note: In a real environment with all dependencies, we'd use proper fixtures.
    with pytest.MonkeyPatch().context() as mp:
        # Mocking profiler.compute to verify the 'm' parameter
        mock_compute = MagicMock()
        mock_compute.return_value.spectral_score = 0.5
        mock_compute.return_value.effective_length = 100
        mp.setattr("ts_triage.pipeline.profiler.compute", mock_compute)

        # Mocking other components to avoid failure
        mp.setattr("ts_triage.pipeline.triage.decide", MagicMock())
        mp.setattr("ts_triage.pipeline.param_infer.infer", MagicMock())
        mp.setattr("ts_triage.pipeline.TriageResult", MagicMock())

        # Call the pipeline with an unknown frequency
        triage_pipeline(y, freq="UNKNOWN_FREQ")

        # Verify that _freq_to_period resulted in m=12 being passed to the profiler
        mock_compute.assert_called()
        args, _ = mock_compute.call_args
        # args[1] is 'm' in profiler.compute(y, m)
        assert args[1] == 12
