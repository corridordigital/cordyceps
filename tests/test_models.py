"""Tests for ts_triage.models (fit/predict wrappers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_triage.models import (
    _croston_fit,
    _fit_croston,
    _fit_naive,
    _infer_prophet_freq,
    fit_and_forecast,
)
from ts_triage.schemas import ModelParams, ModelRecommendation


def _monthly(n=60, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 10 + 0.05 * t + rng.normal(0, 0.5, n)
    return pd.Series(y, index=pd.date_range("2015-01-01", periods=n, freq="ME"))


class TestNaiveModel:

    def test_future_index_fallback(self):
        # Create an irregular series where freq cannot be inferred
        # Ensure we use ns resolution so asi8 returns nanoseconds
        idx = pd.DatetimeIndex([
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-06", # break the pattern to fail infer_freq
        ]).as_unit("ns")

        y = pd.Series([1, 2, 3, 4, 5], index=idx)

        # infer_freq should fail
        assert pd.infer_freq(idx) is None

        from ts_triage.models import _future_index
        future_idx = _future_index(y, horizon=3)

        assert len(future_idx) == 3
        # Diffs: 1 day, 1 day, 1 day, 2 days. Median diff is 1 day.
        # Last date is 2020-01-06. So next dates should be 7th, 8th, 9th.
        assert future_idx[0] == pd.Timestamp("2020-01-07")
        assert future_idx[1] == pd.Timestamp("2020-01-08")
        assert future_idx[2] == pd.Timestamp("2020-01-09")

    def test_last_value_forecast(self):
        y = _monthly()
        _, forecast, aic = _fit_naive(y, "last", horizon=3)
        assert forecast is not None
        assert len(forecast) == 3
        assert aic is None
        last_val = float(y.dropna().iloc[-1])
        assert np.allclose(forecast.values, last_val)

    def test_mean_value_forecast(self):
        y = _monthly()
        _, forecast, aic = _fit_naive(y, "mean", horizon=5)
        assert len(forecast) == 5
        mean_val = float(y.dropna().mean())
        assert np.allclose(forecast.values, mean_val)

    def test_no_horizon_no_forecast(self):
        y = _monthly()
        _, forecast, _ = _fit_naive(y, "last", horizon=None)
        assert forecast is None

    def test_forecast_index_is_datetime(self):
        y = _monthly()
        _, forecast, _ = _fit_naive(y, "last", horizon=6)
        assert isinstance(forecast.index, pd.DatetimeIndex)
        assert len(forecast.index) == 6


class TestCrostonFit:
    def _sparse_series(self, n=50, seed=0):
        rng = np.random.default_rng(seed)
        y = np.zeros(n)
        for i in rng.integers(0, n, size=10):
            y[i] = rng.exponential(5)
        return y

    def test_classic_returns_positive(self):
        y = self._sparse_series()
        val = _croston_fit(y, alpha=0.1, beta=0.05, variant="CLASSIC")
        assert val >= 0.0

    def test_sba_lower_than_classic(self):
        y = self._sparse_series()
        classic = _croston_fit(y, alpha=0.1, beta=0.05, variant="CLASSIC")
        sba = _croston_fit(y, alpha=0.1, beta=0.05, variant="SBA")
        # SBA applies (1 - alpha/2) correction, so SBA <= classic
        assert sba <= classic + 1e-6

    def test_tsb_returns_positive(self):
        y = self._sparse_series()
        val = _croston_fit(y, alpha=0.1, beta=0.05, variant="TSB")
        assert val >= 0.0

    def test_all_zero_returns_zero(self):
        y = np.zeros(20)
        for variant in ("CLASSIC", "SBA", "TSB"):
            val = _croston_fit(y, alpha=0.1, beta=0.05, variant=variant)
            assert val == pytest.approx(0.0)

    def test_croston_wrapper_forecast_length(self):
        y = pd.Series(
            self._sparse_series(),
            index=pd.date_range("2020-01-01", periods=50, freq="W"),
        )
        params = ModelParams(variant="SBA", alpha=0.1)
        _, forecast, aic = _fit_croston(y, params, horizon=4)
        assert forecast is not None
        assert len(forecast) == 4
        assert aic is None


class TestETSFitSmoke:
    """Smoke tests for ETS — verify it runs and returns expected types."""

    def test_ets_ann_fits(self):
        from ts_triage.models import _fit_ets
        y = _monthly()
        params = ModelParams(error="add", trend=None, damped_trend=False,
                             seasonal=None, initialization_method="estimated")
        fitted, forecast, aic = _fit_ets(y, params, horizon=6)
        assert fitted is not None
        assert forecast is not None
        assert len(forecast) == 6
        assert isinstance(aic, float)

    def test_ets_aan_with_trend_fits(self):
        from ts_triage.models import _fit_ets
        y = _monthly()
        params = ModelParams(error="add", trend="add", damped_trend=True,
                             seasonal=None, initialization_method="estimated")
        fitted, forecast, aic = _fit_ets(y, params, horizon=3)
        assert fitted is not None

    def test_ets_fallback_on_nonpositive_for_mul(self):
        """Multiplicative error with non-positive values should not crash."""
        from ts_triage.models import _fit_ets
        rng = np.random.default_rng(0)
        y = pd.Series(
            rng.normal(0, 1, 60),    # can be negative
            index=pd.date_range("2015-01-01", periods=60, freq="ME"),
        )
        params = ModelParams(error="mul", trend=None, damped_trend=False,
                             seasonal=None, initialization_method="estimated")
        fitted, forecast, aic = _fit_ets(y, params, horizon=3)
        # Should fall back to additive internally without raising
        assert fitted is not None


class TestFitAndForecast:
    """Tests for the main fit_and_forecast entry point and its fallback mechanism."""

    def test_fallback_to_naive_on_unknown_model(self):
        y = _monthly(n=20)
        rec = ModelRecommendation(
            model="unknown_model_type",
            variant="last",
            confidence=0.0,
            fallback="naive"
        )
        params = ModelParams()
        # Should NOT raise ValueError, but log a warning and return naive forecast
        model, forecast, aic = fit_and_forecast(y, rec, params, horizon=5)

        assert model is None
        assert aic is None
        assert forecast is not None
        assert len(forecast) == 5
        # Naive "last" variant should be used
        assert np.allclose(forecast.values, float(y.iloc[-1]))

    def test_fallback_to_naive_on_exception(self, monkeypatch):
        from ts_triage.models import _fit_ets
        y = _monthly(n=20)
        rec = ModelRecommendation(
            model="ets",
            variant="ANN",
            confidence=1.0,
            fallback="naive"
        )
        params = ModelParams(error="add", trend=None, seasonal=None)

        def mock_fit_ets_fail(*args, **kwargs):
            raise RuntimeError("ETS fit exploded")

        monkeypatch.setattr("ts_triage.models._fit_ets", mock_fit_ets_fail)

        # Should catch RuntimeError and fall back to naive
        model, forecast, aic = fit_and_forecast(y, rec, params, horizon=3)
        assert model is None
        assert forecast is not None
        assert len(forecast) == 3


class TestProphetFreqInference:
    def test_explicit_freq(self):
        y = pd.Series(
            [1, 2, 3],
            index=pd.date_range("2020-01-01", periods=3, freq="MS")
        )
        assert _infer_prophet_freq(y) == "MS"

    def test_inferred_freq(self):
        # Index without explicit freq but inferrable
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        y = pd.Series([1, 2, 3], index=idx)
        assert y.index.freq is None
        assert _infer_prophet_freq(y) == "D"

    def test_fallback_freq(self):
        # Irregular index that cannot be inferred
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-05"])
        y = pd.Series([1, 2, 3], index=idx)
        assert y.index.freq is None
        # Prophet doesn't like None, so we expect "D"
        assert _infer_prophet_freq(y) == "D"
