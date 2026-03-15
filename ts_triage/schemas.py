"""Dataclass schemas for ts_triage outputs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import pandas as pd


@dataclass
class TSProfile:
    """Minimal statistical properties of a univariate time series."""

    length: int
    effective_length: int
    missing_rate: float
    sparsity: float
    dominant_period: int
    spectral_score: float
    seasonality_strength: float
    trend_strength: float
    variance_type: str          # 'add' | 'mul'
    trend_instability: float
    p_interdemand: float
    cv2_demand: float
    obsolescence_flag: bool

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class ModelRecommendation:
    """Model selection result from the triage decision tree."""

    model: str          # 'naive' | 'croston' | 'ets' | 'prophet'
    variant: str        # 'last'|'mean' / 'SBA'|'TSB'|'classic' / 'ANN'|'AAN'|'AAA'|... / 'prophet'
    confidence: float   # [0,1] — distance to triage thresholds
    fallback: str       # fallback model if fit fails

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class ModelParams:
    """Analytically-inferred hyperparameters for the recommended model."""

    # ETS
    error: Optional[str] = None
    trend: Optional[str] = None
    damped_trend: Optional[bool] = None
    seasonal: Optional[str] = None
    seasonal_periods: Optional[int] = None
    initialization_method: Optional[str] = None

    # Prophet
    growth: Optional[str] = None
    seasonality_mode: Optional[str] = None
    changepoint_prior_scale: Optional[float] = None
    seasonality_prior_scale: Optional[float] = None
    changepoint_range: Optional[float] = None
    n_changepoints: Optional[int] = None
    yearly_seasonality: Optional[bool] = None
    weekly_seasonality: Optional[bool] = None

    # Croston
    variant: Optional[str] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: v for f in fields(self) if (v := getattr(self, f.name)) is not None}


@dataclass
class TriageResult:
    """Full pipeline output: profile + recommendation + params + optional fit."""

    profile: TSProfile
    recommendation: ModelRecommendation
    params: ModelParams
    fitted_model: Any = field(default=None, repr=False)
    forecast: Optional[pd.Series] = field(default=None, repr=False)
    aic: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "profile": self.profile.to_dict(),
            "recommendation": self.recommendation.to_dict(),
            "params": self.params.to_dict(),
            "aic": self.aic,
        }
        if self.forecast is not None:
            d["forecast"] = self.forecast.to_dict()
        return d
