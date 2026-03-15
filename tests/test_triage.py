"""Tests for ts_triage.triage decision tree."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from ts_triage.schemas import TSProfile
from ts_triage.triage import decide


def _profile(**kwargs) -> TSProfile:
    """Build a TSProfile with sensible defaults, overriding with kwargs."""
    defaults = dict(
        length=120,
        effective_length=120,
        missing_rate=0.0,
        sparsity=0.0,
        dominant_period=12,
        spectral_score=0.5,
        seasonality_strength=0.6,
        trend_strength=0.4,
        variance_type="add",
        trend_instability=0.1,
        p_interdemand=1.0,
        cv2_demand=0.2,
        obsolescence_flag=False,
    )
    defaults.update(kwargs)
    return TSProfile(**defaults)


class TestTriageRules:
    # Rule 1 — too short
    def test_short_series_naive_mean(self):
        p = _profile(length=10)
        rec = decide(p, m=12)
        assert rec.model == "naive"
        assert rec.variant == "mean"

    def test_length_23_naive(self):
        p = _profile(length=23)
        rec = decide(p, m=12)
        assert rec.model == "naive"

    def test_length_24_not_naive_by_rule1(self):
        p = _profile(length=24)
        rec = decide(p, m=12)
        # May still be naive if spectral_score is low, but Rule 1 should not fire
        assert not (rec.model == "naive" and rec.variant == "mean")

    # Rule 2 — low spectral score
    def test_low_spectral_naive_last(self):
        p = _profile(spectral_score=0.05)
        rec = decide(p, m=12)
        assert rec.model == "naive"
        assert rec.variant == "last"
        assert rec.confidence == 0.0

    def test_spectral_exactly_threshold_still_naive(self):
        p = _profile(spectral_score=0.14)
        rec = decide(p, m=12)
        assert rec.model == "naive"

    # Rule 3 — sparse → Croston / ETS via SBC
    def test_sparse_high_p_low_cv2_sba(self):
        # p >= 1.32, cv2 < 0.49 → SBA
        p = _profile(sparsity=0.7, p_interdemand=2.0, cv2_demand=0.3)
        rec = decide(p, m=12)
        assert rec.model == "croston"
        assert rec.variant == "SBA"

    def test_sparse_high_p_high_cv2_tsb_with_obsolescence(self):
        # p >= 1.32, cv2 >= 0.49, obsolescence → TSB
        p = _profile(sparsity=0.7, p_interdemand=2.0, cv2_demand=0.6, obsolescence_flag=True)
        rec = decide(p, m=12)
        assert rec.model == "croston"
        assert rec.variant == "TSB"

    def test_sparse_high_p_high_cv2_sba_no_obsolescence(self):
        # p >= 1.32, cv2 >= 0.49, no obsolescence → SBA
        p = _profile(sparsity=0.7, p_interdemand=2.0, cv2_demand=0.6, obsolescence_flag=False)
        rec = decide(p, m=12)
        assert rec.model == "croston"
        assert rec.variant == "SBA"

    def test_sparse_low_p_low_cv2_reclassified_ets(self):
        # p < 1.32, cv2 < 0.49 → ETS (reclassification)
        p = _profile(sparsity=0.7, p_interdemand=1.0, cv2_demand=0.3)
        rec = decide(p, m=12)
        assert rec.model == "ets"

    def test_sparse_low_p_high_cv2_reclassified_ets(self):
        # p < 1.32, cv2 >= 0.49 → ETS error='add' (reclassification)
        p = _profile(sparsity=0.7, p_interdemand=1.0, cv2_demand=0.6)
        rec = decide(p, m=12)
        assert rec.model == "ets"

    # Rule 4 — long horizon → Prophet
    def test_long_horizon_prophet(self):
        p = _profile()
        rec = decide(p, m=12, horizon=30)  # 30 > 2*12
        assert rec.model == "prophet"

    def test_horizon_exactly_2m_not_prophet(self):
        p = _profile()
        rec = decide(p, m=12, horizon=24)  # 24 == 2*12, not > 2*12
        assert rec.model != "prophet"

    # Rule 5 — missing data → Prophet
    def test_high_missing_rate_prophet(self):
        p = _profile(missing_rate=0.15)
        rec = decide(p, m=12)
        assert rec.model == "prophet"

    # Default → ETS
    def test_default_ets(self):
        p = _profile()
        rec = decide(p, m=12)
        assert rec.model == "ets"

    def test_ets_fallback_is_set(self):
        p = _profile()
        rec = decide(p, m=12)
        assert rec.fallback is not None
        assert len(rec.fallback) > 0

    def test_confidence_range(self):
        p = _profile()
        rec = decide(p, m=12)
        assert 0.0 <= rec.confidence <= 1.0

    def test_to_dict(self):
        p = _profile()
        rec = decide(p, m=12)
        d = rec.to_dict()
        assert "model" in d
        assert "variant" in d
        assert "confidence" in d
        assert "fallback" in d
