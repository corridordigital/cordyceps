"""
Single entry-point for the ts_triage pipeline.

Execution order
---------------
1. infer_freq(y)                            # if freq=None
2. profiler.compute(y, m)                   # TSProfile — one STL pass
3. triage.decide(profile, m, horizon)       # ModelRecommendation
4. param_infer.infer(profile, rec, h, m)    # ModelParams
5. [optional] models.fit_and_forecast(…)    # if fit=True
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import models, param_infer, profiler, triage
from .exceptions import TSInforecastableError
from .schemas import ModelRecommendation, ModelParams, TriageResult, TSProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frequency → default seasonal period
# ---------------------------------------------------------------------------

_FREQ_TO_PERIOD: dict[str, int] = {
    "S": 3600,
    "T": 60,   "min": 60,
    "H": 24,
    "D": 7,
    "W": 52,
    "M": 12,   "MS": 12,  "ME": 12,
    "Q": 4,    "QS": 4,   "QE": 4,
    "A": 1,    "AS": 1,   "AE": 1,
    "Y": 1,    "YS": 1,   "YE": 1,
    "B": 5,    "BM": 12,
}

_NS_FALLBACK: dict[str, float] = {
    "S": 1e9,
    "T": 60e9,
    "H": 3600e9,
    "D": 86400e9,
    "W": 7 * 86400e9,
    "M": 30.44 * 86400e9,
    "Q": 91.31 * 86400e9,
    "A": 365.25 * 86400e9,
}


def _freq_to_period(freq: str) -> int:
    """Map a pandas frequency alias to a default seasonal period."""
    if freq is None:
        return 12
    # Normalise: strip trailing digits/modifiers (e.g. '2D' → 'D')
    base = freq.lstrip("0123456789-")
    # Try exact match first
    if base in _FREQ_TO_PERIOD:
        return _FREQ_TO_PERIOD[base]
    # Try prefix match
    for key in sorted(_FREQ_TO_PERIOD, key=len, reverse=True):
        if base.upper().startswith(key.upper()):
            return _FREQ_TO_PERIOD[key]
    return 12   # safe default


def _infer_freq(y: pd.Series) -> Optional[str]:
    """Best-effort frequency inference from DatetimeIndex."""
    if not isinstance(y.index, pd.DatetimeIndex):
        return None

    # 1. Honour existing freq attribute
    if y.index.freq is not None:
        return str(y.index.freqstr)

    # 2. pandas infer_freq
    if len(y) >= 3:
        try:
            inferred = pd.infer_freq(y.index)
            if inferred:
                return inferred
        except Exception:
            pass

    # 3. Fallback: closest named period from median diff
    if len(y) >= 2:
        diffs_ns = np.diff(y.index.asi8).astype(float)
        median_ns = float(np.median(diffs_ns))
        best = min(_NS_FALLBACK.items(), key=lambda kv: abs(median_ns - kv[1]))
        return best[0]

    return None


# ---------------------------------------------------------------------------
# Inforecastability checks
# ---------------------------------------------------------------------------

_SPECTRAL_THRESHOLD = 0.05
_MIN_EFFECTIVE_LENGTH = 10
_MAX_SPARSITY_STRICT = 0.95


def _check_inforecastable(profile: TSProfile) -> None:
    """Raise TSInforecastableError if the series is clearly unforecastable."""
    if profile.spectral_score < _SPECTRAL_THRESHOLD:
        raise TSInforecastableError(
            reason="low_spectral_score",
            value=profile.spectral_score,
            threshold=_SPECTRAL_THRESHOLD,
        )
    if profile.effective_length < _MIN_EFFECTIVE_LENGTH:
        raise TSInforecastableError(
            reason="insufficient_length",
            value=float(profile.effective_length),
            threshold=float(_MIN_EFFECTIVE_LENGTH),
        )
    if profile.sparsity > _MAX_SPARSITY_STRICT:
        raise TSInforecastableError(
            reason="too_sparse",
            value=profile.sparsity,
            threshold=_MAX_SPARSITY_STRICT,
        )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def triage_pipeline(
    y: pd.Series,
    freq: Optional[str] = None,
    horizon: Optional[int] = None,
    fit: bool = False,
    strict: bool = False,
) -> TriageResult:
    """
    Full profiling → triage → param-inference pipeline.

    Parameters
    ----------
    y       : univariate time series (pd.Series, DatetimeIndex, float values)
    freq    : pandas frequency alias ('D','W','M',…); inferred if None
    horizon : number of future periods to forecast; None → no forecast intent
    fit     : if True, fit the selected model and populate forecast / aic
    strict  : if True, raise TSInforecastableError for unforecastable series

    Returns
    -------
    TriageResult with profile, recommendation, params always populated.
    fitted_model and forecast are None when fit=False.
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must have a DatetimeIndex.")

    # --- 1. Infer frequency --------------------------------------------------
    if freq is None:
        freq = _infer_freq(y)
        if freq is None:
            warnings.warn(
                "Could not infer frequency from index. Defaulting to 'M'.",
                UserWarning,
                stacklevel=2,
            )
            freq = "M"

    m: int = _freq_to_period(freq)

    # --- 2. Profile ----------------------------------------------------------
    profile = profiler.compute(y, m)

    # --- 3. Inforecastability gate -------------------------------------------
    if strict:
        _check_inforecastable(profile)
    elif profile.spectral_score < _SPECTRAL_THRESHOLD or profile.effective_length < _MIN_EFFECTIVE_LENGTH:
        logger.warning(
            "Series may be unforecastable (spectral_score=%.3f, effective_length=%d). "
            "Falling back to naive. Pass strict=True to raise instead.",
            profile.spectral_score,
            profile.effective_length,
        )

    # --- 4. Triage -----------------------------------------------------------
    rec: ModelRecommendation = triage.decide(profile, m, horizon)

    # --- 5. Param inference --------------------------------------------------
    params: ModelParams = param_infer.infer(profile, rec, horizon, m)

    # --- 6. Optional fit & forecast ------------------------------------------
    fitted_model: Any = None
    forecast: Optional[pd.Series] = None
    aic: Optional[float] = None

    if fit:
        fitted_model, forecast, aic = models.fit_and_forecast(y, rec, params, horizon)

    return TriageResult(
        profile=profile,
        recommendation=rec,
        params=params,
        fitted_model=fitted_model,
        forecast=forecast,
        aic=aic,
    )
