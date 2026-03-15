"""Tests for ts_triage.profiler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_triage.profiler import compute, _fill_nan, _compute_spectral, _compute_demand_metrics


class TestFillNan:
    def test_no_nan(self):
        y = np.array([1.0, 2.0, 3.0])
        out = _fill_nan(y)
        np.testing.assert_array_equal(out, y)

    def test_with_nan(self):
        y = np.array([1.0, np.nan, 3.0])
        out = _fill_nan(y)
        assert not np.any(np.isnan(out))
        assert out[1] == pytest.approx(2.0)

    def test_all_nan(self):
        y = np.array([np.nan, np.nan, np.nan])
        out = _fill_nan(y)
        assert out.shape == (3,)
        assert np.all(out == 0.0)


class TestComputeSpectral:
    def test_periodic_signal_high_score(self):
        # Pure sine → spectral score should be high
        t = np.arange(200)
        y = np.sin(2 * np.pi * t / 12)
        period, score = _compute_spectral(y, 12)
        assert score > 0.3
        assert period == pytest.approx(12, abs=2)

    def test_white_noise_low_score(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 500)
        _, score = _compute_spectral(y, 12)
        assert score < 0.5

    def test_short_series_returns_default(self):
        y = np.array([1.0, 2.0])
        period, score = _compute_spectral(y, 7)
        assert period == 7
        assert score == 0.0


class TestComputeDemandMetrics:
    def test_dense_series(self):
        y = np.ones(50)
        p, cv2 = _compute_demand_metrics(y)
        assert p == pytest.approx(1.0, abs=1e-6)
        assert cv2 == pytest.approx(0.0, abs=1e-6)

    def test_sparse_series(self):
        y = np.zeros(20)
        y[0] = 1.0
        y[10] = 2.0
        p, cv2 = _compute_demand_metrics(y)
        assert p == pytest.approx(10.0)

    def test_single_demand(self):
        y = np.zeros(10)
        y[5] = 3.0
        p, cv2 = _compute_demand_metrics(y)
        # Only one non-zero, no intervals
        assert cv2 == pytest.approx(0.0)


class TestComputeProfile:
    def test_basic_fields_populated(self, monthly_seasonal):
        profile = compute(monthly_seasonal, m=12)
        assert profile.length == 120
        assert 0.0 <= profile.missing_rate <= 1.0
        assert 0.0 <= profile.sparsity <= 1.0
        assert 0.0 <= profile.spectral_score <= 1.0
        assert 0.0 <= profile.seasonality_strength <= 1.0
        assert 0.0 <= profile.trend_strength <= 1.0
        assert profile.variance_type in ("add", "mul")
        assert profile.dominant_period >= 2

    def test_monthly_seasonal_detects_period(self, monthly_seasonal):
        profile = compute(monthly_seasonal, m=12)
        # dominant period should be close to 12 for monthly data with annual seasonality
        assert 8 <= profile.dominant_period <= 16

    def test_seasonal_strength_high_for_seasonal_series(self, monthly_seasonal):
        profile = compute(monthly_seasonal, m=12)
        assert profile.seasonality_strength > 0.3

    def test_sparse_series_sparsity(self, sparse_series):
        profile = compute(sparse_series, m=52)
        assert profile.sparsity > 0.5

    def test_short_series(self, short_series):
        profile = compute(short_series, m=12)
        assert profile.length == 10

    def test_series_with_nans(self, series_with_nans):
        profile = compute(series_with_nans, m=12)
        assert profile.missing_rate > 0.0
        assert profile.missing_rate < 1.0

    def test_effective_length(self):
        y = pd.Series(
            [0.0, 1.0, np.nan, 2.0, 0.0],
            index=pd.date_range("2020-01-01", periods=5, freq="ME"),
        )
        profile = compute(y, m=12)
        assert profile.effective_length == 2  # only 1.0 and 2.0

    def test_to_dict_serialisable(self, monthly_seasonal):
        profile = compute(monthly_seasonal, m=12)
        d = profile.to_dict()
        assert isinstance(d, dict)
        import json
        json.dumps(d)   # must not raise

    def test_obsolescence_flag_type(self, sparse_series):
        profile = compute(sparse_series, m=52)
        assert isinstance(profile.obsolescence_flag, bool)

    def test_trend_strength_random_walk(self, random_walk):
        # Random walk has some accumulated trend but also high instability
        profile = compute(random_walk, m=7)
        assert isinstance(profile.trend_strength, float)

    def test_performance_500(self, benchmark=None):
        """Profile computation for T=500 should be fast (not a strict benchmark here)."""
        import time
        rng = np.random.default_rng(0)
        n = 500
        t = np.arange(n)
        y = pd.Series(
            10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n),
            index=pd.date_range("2010-01-01", periods=n, freq="ME"),
        )
        start = time.perf_counter()
        compute(y, m=12)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Allow generous budget in CI
        assert elapsed_ms < 500, f"Profiling took {elapsed_ms:.1f}ms (budget 500ms)"
