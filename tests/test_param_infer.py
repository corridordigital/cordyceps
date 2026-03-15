"""Tests for ts_triage.param_infer."""

from __future__ import annotations

import pytest

from ts_triage.param_infer import infer
from ts_triage.schemas import ModelRecommendation, TSProfile


def _profile(**kwargs) -> TSProfile:
    defaults = dict(
        length=120,
        effective_length=80,
        missing_rate=0.0,
        sparsity=0.0,
        dominant_period=12,
        spectral_score=0.5,
        seasonality_strength=0.6,
        trend_strength=0.4,
        variance_type="add",
        trend_instability=0.1,
        p_interdemand=2.0,
        cv2_demand=0.3,
        obsolescence_flag=False,
    )
    defaults.update(kwargs)
    return TSProfile(**defaults)


def _rec(model: str, variant: str = "ANN") -> ModelRecommendation:
    return ModelRecommendation(model=model, variant=variant, confidence=0.8, fallback="naive")


class TestETSParams:
    def test_additive_error_from_variance_type(self):
        p = _profile(variance_type="add")
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.error == "add"

    def test_multiplicative_error_from_variance_type(self):
        p = _profile(variance_type="mul")
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.error == "mul"

    def test_trend_when_strong(self):
        p = _profile(trend_strength=0.5)
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.trend == "add"

    def test_no_trend_when_weak(self):
        p = _profile(trend_strength=0.05)
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.trend is None

    def test_damped_trend_default_true(self):
        p = _profile(trend_strength=0.5)
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.damped_trend is True

    def test_damped_trend_false_for_short_horizon(self):
        p = _profile(trend_strength=0.5)
        params = infer(p, _rec("ets"), horizon=3, m=12)  # 3 <= 12/2
        assert params.damped_trend is False

    def test_seasonal_when_strong(self):
        p = _profile(seasonality_strength=0.6, dominant_period=12)
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.seasonal is not None
        assert params.seasonal_periods == 12

    def test_no_seasonal_when_weak(self):
        p = _profile(seasonality_strength=0.1)
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.seasonal is None

    def test_initialization_heuristic_long(self):
        p = _profile(length=250)
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.initialization_method == "heuristic"

    def test_initialization_estimated_short(self):
        p = _profile(length=100)
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.initialization_method == "estimated"

    def test_sbc_reclassification_forces_add_error(self):
        # SBC cell: sparse, p < 1.32, cv2 >= 0.49 → force error='add'
        p = _profile(sparsity=0.7, p_interdemand=1.0, cv2_demand=0.6, variance_type="mul")
        params = infer(p, _rec("ets"), horizon=12, m=12)
        assert params.error == "add"

    def test_to_dict_has_relevant_keys(self):
        p = _profile()
        params = infer(p, _rec("ets"), horizon=12, m=12)
        d = params.to_dict()
        assert "error" in d
        assert "initialization_method" in d
        # Prophet/Croston keys should be absent
        assert "growth" not in d
        assert "alpha" not in d


class TestProphetParams:
    def test_flat_growth_for_no_trend(self):
        p = _profile(trend_strength=0.02)
        params = infer(p, _rec("prophet"), horizon=24, m=12)
        assert params.growth == "flat"

    def test_linear_growth_for_trend(self):
        p = _profile(trend_strength=0.5)
        params = infer(p, _rec("prophet"), horizon=24, m=12)
        assert params.growth == "linear"

    def test_seasonality_mode_from_variance_type(self):
        p_add = _profile(variance_type="add")
        p_mul = _profile(variance_type="mul")
        assert infer(p_add, _rec("prophet"), horizon=24, m=12).seasonality_mode == "additive"
        assert infer(p_mul, _rec("prophet"), horizon=24, m=12).seasonality_mode == "multiplicative"

    def test_changepoint_prior_scale_mapping(self):
        for ti, expected_cps in [(0.03, 0.01), (0.10, 0.05), (0.20, 0.15), (0.35, 0.30)]:
            p = _profile(trend_instability=ti)
            params = infer(p, _rec("prophet"), horizon=24, m=12)
            assert params.changepoint_prior_scale == pytest.approx(expected_cps)

    def test_seasonality_prior_scale_range(self):
        for ss in [0.0, 0.5, 1.0]:
            p = _profile(seasonality_strength=ss)
            params = infer(p, _rec("prophet"), horizon=24, m=12)
            assert 1.0 <= params.seasonality_prior_scale <= 20.0

    def test_changepoint_range(self):
        p_stable = _profile(trend_instability=0.10)
        p_unstable = _profile(trend_instability=0.25)
        assert infer(p_stable, _rec("prophet"), horizon=24, m=12).changepoint_range == 0.80
        assert infer(p_unstable, _rec("prophet"), horizon=24, m=12).changepoint_range == 0.85

    def test_n_changepoints_short(self):
        p = _profile(length=80)
        params = infer(p, _rec("prophet"), horizon=24, m=12)
        assert params.n_changepoints == 80 // 4

    def test_n_changepoints_long(self):
        p = _profile(length=300)
        params = infer(p, _rec("prophet"), horizon=24, m=12)
        assert params.n_changepoints == 25


class TestCrostonParams:
    def test_alpha_range(self):
        for eff_len in [1, 10, 50, 200]:
            p = _profile(effective_length=eff_len)
            rec = ModelRecommendation(model="croston", variant="SBA", confidence=0.7, fallback="naive")
            params = infer(p, rec, horizon=4, m=12)
            assert 0.05 <= params.alpha <= 0.30

    def test_beta_only_for_tsb(self):
        p = _profile(effective_length=50)
        rec_sba = ModelRecommendation(model="croston", variant="SBA", confidence=0.7, fallback="naive")
        rec_tsb = ModelRecommendation(model="croston", variant="TSB", confidence=0.7, fallback="naive")
        params_sba = infer(p, rec_sba, horizon=4, m=12)
        params_tsb = infer(p, rec_tsb, horizon=4, m=12)
        assert params_sba.beta is None
        assert params_tsb.beta is not None
        assert params_tsb.beta == pytest.approx(params_tsb.alpha / 2.0)

    def test_variant_preserved(self):
        p = _profile(effective_length=30)
        rec = ModelRecommendation(model="croston", variant="TSB", confidence=0.7, fallback="naive")
        params = infer(p, rec, horizon=4, m=12)
        assert params.variant == "TSB"


class TestNaiveParams:
    def test_naive_returns_empty_params(self):
        p = _profile()
        rec = ModelRecommendation(model="naive", variant="last", confidence=1.0, fallback="naive")
        params = infer(p, rec, horizon=None, m=12)
        # All fields should be None
        d = params.to_dict()
        assert d == {}
