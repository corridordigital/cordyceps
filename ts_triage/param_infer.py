"""
Infer near-optimal hyperparameters analytically from TSProfile.

No grid search, no cross-validation, no model fitting at this stage.
"""

import numpy as np
from typing import Optional

from .schemas import ModelParams, ModelRecommendation, TSProfile


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def infer(
    profile: TSProfile,
    rec: ModelRecommendation,
    horizon: Optional[int],
    m: int,
) -> ModelParams:
    """Return hyperparameters analytically inferred from *profile*."""
    model = rec.model

    if model == "ets":
        return _ets_params(profile, rec, horizon, m)
    if model == "prophet":
        return _prophet_params(profile, m)
    if model == "croston":
        return _croston_params(profile, rec)
    # naive — no meaningful hyperparameters
    return ModelParams()


# ---------------------------------------------------------------------------
# ETS
# ---------------------------------------------------------------------------

def _ets_params(
    profile: TSProfile,
    rec: ModelRecommendation,
    horizon: Optional[int],
    m: int,
) -> ModelParams:
    # --- error type ----------------------------------------------------------
    # SBC reclassification with forced additive error:
    # identified by high cv2 + short inter-demand + sparse origin
    sbc_force_add = (
        profile.sparsity > 0.5
        and profile.cv2_demand >= 0.49
        and profile.p_interdemand < 1.32
    )
    error = "add" if (sbc_force_add or profile.variance_type == "add") else "mul"

    # --- trend ---------------------------------------------------------------
    trend: Optional[str] = "add" if profile.trend_strength > 0.1 else None

    # --- damped trend --------------------------------------------------------
    # Default True (Hyndman et al. recommendation).
    # Disable when horizon is short relative to seasonality.
    damped_trend = True
    if trend is not None and horizon is not None and horizon <= m / 2:
        damped_trend = False

    # --- seasonal ------------------------------------------------------------
    seasonal: Optional[str] = None
    seasonal_periods: Optional[int] = None
    if profile.seasonality_strength > 0.2:
        seasonal = profile.variance_type  # 'add' or 'mul' mirrors error component
        seasonal_periods = profile.dominant_period

    # --- initialisation ------------------------------------------------------
    initialization_method = "heuristic" if profile.length > 200 else "estimated"

    return ModelParams(
        error=error,
        trend=trend,
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method=initialization_method,
    )


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def _prophet_params(profile: TSProfile, m: int) -> ModelParams:
    # --- growth --------------------------------------------------------------
    growth = "flat" if profile.trend_strength < 0.05 else "linear"

    # --- seasonality mode mirrors variance type ------------------------------
    seasonality_mode = (
        "multiplicative" if profile.variance_type == "mul" else "additive"
    )

    # --- changepoint_prior_scale from trend_instability ---------------------
    ti = profile.trend_instability
    if ti < 0.05:
        cps = 0.01
    elif ti < 0.15:
        cps = 0.05
    elif ti < 0.30:
        cps = 0.15
    else:
        cps = 0.30

    # --- seasonality_prior_scale → [1, 20] -----------------------------------
    seasonality_prior_scale = float(1.0 + 19.0 * profile.seasonality_strength)

    # --- changepoint_range ---------------------------------------------------
    changepoint_range = 0.85 if profile.trend_instability > 0.20 else 0.80

    # --- n_changepoints ------------------------------------------------------
    n_changepoints = profile.length // 4 if profile.length < 100 else 25

    # --- built-in seasonality toggles ----------------------------------------
    dp = profile.dominant_period
    yearly_seasonality = dp in (365, 364, 366) or (m == 12 and dp == 12)
    weekly_seasonality = dp == 7

    return ModelParams(
        growth=growth,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=float(cps),
        seasonality_prior_scale=seasonality_prior_scale,
        changepoint_range=changepoint_range,
        n_changepoints=n_changepoints,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
    )


# ---------------------------------------------------------------------------
# Croston / SBA / TSB
# ---------------------------------------------------------------------------

def _croston_params(profile: TSProfile, rec: ModelRecommendation) -> ModelParams:
    n_nonzero = max(1, profile.effective_length)
    alpha = float(np.clip(2.0 / (n_nonzero + 1), 0.05, 0.30))
    beta: Optional[float] = float(alpha / 2.0) if rec.variant == "TSB" else None

    return ModelParams(
        variant=rec.variant,
        alpha=alpha,
        beta=beta,
    )
