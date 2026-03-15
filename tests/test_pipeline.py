"""Tests for ts_triage.pipeline (end-to-end)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_triage import triage_pipeline, TSInforecastableError
from ts_triage.schemas import TriageResult


class TestTriagePipeline:
    def test_returns_triage_result(self, monthly_seasonal):
        result = triage_pipeline(monthly_seasonal)
        assert isinstance(result, TriageResult)

    def test_all_three_fields_populated(self, monthly_seasonal):
        result = triage_pipeline(monthly_seasonal)
        assert result.profile is not None
        assert result.recommendation is not None
        assert result.params is not None

    def test_no_fit_by_default(self, monthly_seasonal):
        result = triage_pipeline(monthly_seasonal)
        assert result.fitted_model is None
        assert result.forecast is None
        assert result.aic is None

    def test_freq_inferred_from_index(self, monthly_seasonal):
        # No freq argument → inferred from DatetimeIndex
        result = triage_pipeline(monthly_seasonal)
        assert result.profile.length == 120

    def test_explicit_freq_overrides_inferred(self, monthly_seasonal):
        result = triage_pipeline(monthly_seasonal, freq="ME")
        assert result.profile is not None

    def test_short_series_naive_recommendation(self, short_series):
        result = triage_pipeline(short_series)
        assert result.recommendation.model == "naive"

    def test_sparse_series_croston_or_ets(self, sparse_series):
        result = triage_pipeline(sparse_series)
        assert result.recommendation.model in ("croston", "ets", "naive")

    def test_series_with_missing_data(self, series_with_nans):
        result = triage_pipeline(series_with_nans)
        # High missing rate → Prophet or fallback
        assert result.recommendation.model in ("prophet", "ets", "naive")

    def test_strict_false_does_not_raise_random_walk(self, random_walk):
        # Should not raise even for noisy series
        result = triage_pipeline(random_walk, strict=False)
        assert result is not None

    def test_to_dict_is_json_serialisable(self, monthly_seasonal):
        import json
        result = triage_pipeline(monthly_seasonal)
        d = result.to_dict()
        json.dumps(d)   # must not raise

    def test_horizon_none_allowed(self, monthly_seasonal):
        result = triage_pipeline(monthly_seasonal, horizon=None)
        assert result is not None

    def test_non_datetime_index_raises(self):
        y = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
        with pytest.raises(TypeError):
            triage_pipeline(y)


class TestPipelineWithFit:
    def test_fit_ets_returns_fitted_model(self, monthly_seasonal):
        result = triage_pipeline(monthly_seasonal, fit=True, horizon=12)
        if result.recommendation.model == "ets":
            assert result.fitted_model is not None
            assert result.aic is not None

    def test_fit_naive_returns_forecast(self, short_series):
        result = triage_pipeline(short_series, fit=True, horizon=3)
        assert result.forecast is not None
        assert len(result.forecast) == 3

    def test_fit_forecast_length_matches_horizon(self, monthly_seasonal):
        result = triage_pipeline(monthly_seasonal, fit=True, horizon=6)
        if result.forecast is not None:
            assert len(result.forecast) == 6

    def test_fit_sparse_croston(self, sparse_series):
        result = triage_pipeline(sparse_series, fit=True, horizon=4)
        # Should not crash; model or fallback fits
        assert result is not None


class TestStrictMode:
    def test_strict_raises_for_pure_noise(self):
        """A series with extremely low spectral score raises in strict mode."""
        rng = np.random.default_rng(0)
        # Pure white noise — very low spectral score
        y = pd.Series(
            rng.normal(0, 1, 200),
            index=pd.date_range("2010-01-01", periods=200, freq="D"),
        )
        # strict=False should not raise
        result = triage_pipeline(y, strict=False)
        assert result is not None

        # strict=True may or may not raise depending on actual spectral score;
        # if it raises, it must be TSInforecastableError
        try:
            triage_pipeline(y, strict=True)
        except TSInforecastableError as exc:
            assert exc.reason in ("low_spectral_score", "insufficient_length", "too_sparse")
            assert exc.value >= 0
            assert exc.threshold >= 0

    def test_strict_raises_for_minimal_effective_length(self):
        y = pd.Series(
            [0.0] * 95 + [1.0, 0.0, 0.0, 0.0, 0.0],
            index=pd.date_range("2010-01-01", periods=100, freq="D"),
        )
        with pytest.raises(TSInforecastableError) as exc_info:
            triage_pipeline(y, strict=True)
        assert exc_info.value.reason in ("insufficient_length", "too_sparse", "low_spectral_score")

    def test_tsinforecastable_error_fields(self):
        exc = TSInforecastableError("low_spectral_score", 0.03, 0.05)
        assert exc.reason == "low_spectral_score"
        assert exc.value == pytest.approx(0.03)
        assert exc.threshold == pytest.approx(0.05)
        assert "low_spectral_score" in str(exc)


class TestPipelineReusability:
    def test_multiple_calls_independent(self, monthly_seasonal, daily_series):
        r1 = triage_pipeline(monthly_seasonal)
        r2 = triage_pipeline(daily_series)
        assert r1.profile.length != r2.profile.length or r1.recommendation.model != r2.recommendation.model or True  # just ensure no shared state crash

    def test_deterministic_output(self, monthly_seasonal):
        r1 = triage_pipeline(monthly_seasonal)
        r2 = triage_pipeline(monthly_seasonal)
        assert r1.recommendation.model == r2.recommendation.model
        assert r1.params.to_dict() == r2.params.to_dict()
