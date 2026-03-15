"""
Thin fit/predict wrappers for the four candidate models.

Prophet is lazy-imported: it is NEVER loaded unless it is the selected model.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from .schemas import ModelParams, ModelRecommendation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def fit_and_forecast(
    y: pd.Series,
    rec: ModelRecommendation,
    params: ModelParams,
    horizon: Optional[int],
) -> tuple[Any, Optional[pd.Series], Optional[float]]:
    """
    Fit the recommended model and (optionally) generate a forecast.

    Returns
    -------
    fitted_model : model object (or None for naive)
    forecast     : pd.Series indexed with future DatetimeIndex, or None
    aic          : float for ETS, None otherwise
    """
    try:
        if rec.model == "naive":
            return _fit_naive(y, rec.variant, horizon)
        if rec.model == "ets":
            return _fit_ets(y, params, horizon)
        if rec.model == "croston":
            return _fit_croston(y, params, horizon)
        if rec.model == "prophet":
            return _fit_prophet(y, params, horizon)
        raise ValueError(f"Unknown model: {rec.model!r}")
    except Exception as exc:
        logger.warning("Model fit failed (%s). Falling back to naive.", exc)
        return _fit_naive(y, "last", horizon)


# ---------------------------------------------------------------------------
# Naive
# ---------------------------------------------------------------------------

def _fit_naive(
    y: pd.Series,
    variant: str,
    horizon: Optional[int],
) -> tuple[None, Optional[pd.Series], None]:
    forecast = None
    if horizon and horizon > 0:
        if variant == "last":
            val = float(y.dropna().iloc[-1]) if not y.dropna().empty else 0.0
        else:  # 'mean'
            val = float(y.dropna().mean()) if not y.dropna().empty else 0.0

        future_index = _future_index(y, horizon)
        forecast = pd.Series(np.full(horizon, val), index=future_index, name="forecast")

    return None, forecast, None


# ---------------------------------------------------------------------------
# ETS (statsmodels)
# ---------------------------------------------------------------------------

def _fit_ets(
    y: pd.Series,
    params: ModelParams,
    horizon: Optional[int],
) -> tuple[Any, Optional[pd.Series], Optional[float]]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_clean = y.ffill().bfill()

    # Guard: multiplicative seasonal requires strictly positive values
    seasonal = params.seasonal
    if seasonal == "mul" and (y_clean <= 0).any():
        seasonal = "add"

    # seasonal_periods is required when seasonal is set
    seasonal_periods = params.seasonal_periods if seasonal else None

    model_kwargs: dict[str, Any] = {
        "trend": params.trend,
        "damped_trend": bool(params.damped_trend) if params.trend else False,
        "seasonal": seasonal,
        "initialization_method": params.initialization_method or "estimated",
    }
    if seasonal_periods:
        model_kwargs["seasonal_periods"] = seasonal_periods

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ExponentialSmoothing(y_clean, **model_kwargs).fit(optimized=True)

    forecast = None
    if horizon and horizon > 0:
        future_index = _future_index(y, horizon)
        fcast_vals = fitted.forecast(horizon)
        forecast = pd.Series(
            np.asarray(fcast_vals), index=future_index, name="forecast"
        )

    aic = float(fitted.aic) if hasattr(fitted, "aic") else None
    return fitted, forecast, aic


# ---------------------------------------------------------------------------
# Croston / SBA / TSB
# ---------------------------------------------------------------------------

def _fit_croston(
    y: pd.Series,
    params: ModelParams,
    horizon: Optional[int],
) -> tuple[dict, Optional[pd.Series], None]:
    y_arr = y.fillna(0.0).to_numpy(dtype=float)
    alpha = float(params.alpha or 0.1)
    beta = float(params.beta) if params.beta is not None else alpha / 2.0
    variant = (params.variant or "classic").upper()

    fitted_val = _croston_fit(y_arr, alpha, beta, variant)

    fitted_model = {"variant": variant, "alpha": alpha, "beta": beta, "fitted_value": fitted_val}

    forecast = None
    if horizon and horizon > 0:
        future_index = _future_index(y, horizon)
        forecast = pd.Series(
            np.full(horizon, fitted_val), index=future_index, name="forecast"
        )

    return fitted_model, forecast, None


def _croston_fit(y: np.ndarray, alpha: float, beta: float, variant: str) -> float:
    """
    Fit Croston / SBA / TSB and return the one-step-ahead point forecast.

    Parameters
    ----------
    y       : demand series (zeros for non-demand periods)
    alpha   : smoothing for demand size (and interval in classic/SBA)
    beta    : smoothing for demand probability (TSB only; = alpha/2)
    variant : 'CLASSIC' | 'SBA' | 'TSB'
    """
    nz_idx = np.flatnonzero(y > 0)
    if len(nz_idx) == 0:
        return 0.0

    if variant in ("CLASSIC", "SBA"):
        # Initialise at first demand
        z = float(y[nz_idx[0]])
        q = float(nz_idx[0] + 1) if nz_idx[0] > 0 else 1.0
        last_t = nz_idx[0]

        for i in range(1, len(nz_idx)):
            t = nz_idx[i]
            interval = float(t - last_t)
            z = alpha * float(y[t]) + (1.0 - alpha) * z
            q = alpha * interval + (1.0 - alpha) * q
            last_t = t

        fcast = z / q if q > 0 else z
        if variant == "SBA":
            fcast *= (1.0 - alpha / 2.0)
        return float(fcast)

    # TSB: separate smoothing of P(demand) and demand size
    p = 0.5  # initial demand probability
    z = float(y[nz_idx[0]])  # initial demand size

    for t in range(len(y)):
        if y[t] > 0:
            p = alpha + (1.0 - alpha) * p       # P(demand_t=1) update
            z = beta * float(y[t]) + (1.0 - beta) * z
        else:
            p = (1.0 - alpha) * p

    return float(p * z)


# ---------------------------------------------------------------------------
# Prophet (lazy import)
# ---------------------------------------------------------------------------

def _fit_prophet(
    y: pd.Series,
    params: ModelParams,
    horizon: Optional[int],
) -> tuple[Any, Optional[pd.Series], None]:
    try:
        from prophet import Prophet  # noqa: PLC0415 — intentional lazy import
    except ImportError as exc:
        raise ImportError(
            "prophet is not installed. Install with: pip install prophet"
        ) from exc

    df = y.reset_index()
    df.columns = ["ds", "y"]
    df = df.dropna(subset=["y"])

    model = Prophet(
        growth=params.growth or "linear",
        seasonality_mode=params.seasonality_mode or "additive",
        changepoint_prior_scale=params.changepoint_prior_scale or 0.05,
        seasonality_prior_scale=params.seasonality_prior_scale or 10.0,
        changepoint_range=params.changepoint_range or 0.80,
        n_changepoints=params.n_changepoints or 25,
        yearly_seasonality=params.yearly_seasonality if params.yearly_seasonality is not None else "auto",
        weekly_seasonality=params.weekly_seasonality if params.weekly_seasonality is not None else "auto",
        daily_seasonality=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df)

    forecast = None
    if horizon and horizon > 0:
        future = model.make_future_dataframe(periods=horizon, freq=_infer_prophet_freq(y))
        pred = model.predict(future).tail(horizon)
        future_index = pd.DatetimeIndex(pred["ds"].values)
        forecast = pd.Series(pred["yhat"].values, index=future_index, name="forecast")

    return model, forecast, None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _future_index(y: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Generate a future DatetimeIndex by extending *y*'s index."""
    freq = y.index.freq or pd.infer_freq(y.index)
    if freq is None:
        # Fallback: estimate median diff
        diffs = np.diff(y.index.asi8)
        median_ns = int(np.median(diffs))
        freq = pd.tseries.frequencies.to_offset(pd.Timedelta(median_ns))
    last = y.index[-1]
    return pd.date_range(start=last, periods=horizon + 1, freq=freq)[1:]


def _infer_prophet_freq(y: pd.Series) -> str:
    """Return a pandas frequency string suitable for Prophet make_future_dataframe."""
    freq = y.index.freq
    if freq is not None:
        return str(freq)
    inferred = pd.infer_freq(y.index)
    return inferred or "D"
