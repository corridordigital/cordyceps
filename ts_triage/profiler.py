"""
Compute TSProfile for a univariate time series.

Design constraints:
- Pure numpy/scipy in all compute loops — no pandas operations inside loops.
- STL is run exactly once; its components (trend, seasonal, resid) are cached.
- FFT is run exactly once with a Hann window.
- Every metric reuses arrays already in memory.
"""


import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats as sp_stats
from scipy.fft import rfft, rfftfreq
from statsmodels.tsa.seasonal import STL

from .schemas import TSProfile


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def compute(y_series, m: int) -> TSProfile:
    """
    Compute all profile metrics for *y_series*.

    Parameters
    ----------
    y_series : pd.Series with DatetimeIndex
    m        : default seasonal period inferred from frequency
    """
    y: np.ndarray = y_series.to_numpy(dtype=float)
    n: int = len(y)

    # ---- basic metrics  O(n) ------------------------------------------------
    not_na = ~np.isnan(y)
    not_zero = y != 0.0
    length = n
    effective_length = int(np.sum(not_na & not_zero))
    missing_rate = float(np.mean(~not_na))
    sparsity = float(np.mean(y == 0.0))   # NaN != 0, so NaN slots count as non-zero

    # ---- fill NaN for spectral / rolling analyses ---------------------------
    y_filled = _fill_nan(y)

    # ---- FFT — O(n log n) — one pass ----------------------------------------
    dominant_period, spectral_score = _compute_spectral(y_filled, m)

    # ---- STL — O(n) — one decomposition, cached  ----------------------------
    stl_period = max(2, dominant_period)
    seasonality_strength, trend_strength = _compute_stl(y_filled, stl_period, n)

    # ---- rolling stats — O(n) — one pass ------------------------------------
    win = max(2, m)
    rmean, rstd = _rolling_stats(y_filled, win)

    # ---- variance type — O(n) rolling corr ----------------------------------
    variance_type = _compute_variance_type(rmean, rstd)

    # ---- trend instability — O(n) -------------------------------------------
    trend_instability = _compute_trend_instability(rmean)

    # ---- Croston demand metrics — O(n) --------------------------------------
    p_interdemand, cv2_demand = _compute_demand_metrics(y)

    # ---- obsolescence — O(n) ------------------------------------------------
    obsolescence_flag = _compute_obsolescence(y, n)

    return TSProfile(
        length=length,
        effective_length=effective_length,
        missing_rate=missing_rate,
        sparsity=sparsity,
        dominant_period=dominant_period,
        spectral_score=spectral_score,
        seasonality_strength=seasonality_strength,
        trend_strength=trend_strength,
        variance_type=variance_type,
        trend_instability=trend_instability,
        p_interdemand=p_interdemand,
        cv2_demand=cv2_demand,
        obsolescence_flag=obsolescence_flag,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_nan(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values; returns a copy."""
    if not np.any(np.isnan(y)):
        return y.copy()
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.zeros(len(y), dtype=float)
    return np.interp(x, x[mask], y[mask])


def _compute_spectral(y: np.ndarray, m: int) -> tuple[int, float]:
    """
    Compute dominant period and normalised spectral entropy score Ω ∈ [0,1].

    Ω close to 1 → strong periodicity; Ω close to 0 → noisy / flat spectrum.
    """
    n = len(y)
    if n < 4:
        return m, 0.0

    # detrend + Hann window to reduce spectral leakage
    y_demeaned = y - np.mean(y)
    window = np.hanning(n)
    y_win = y_demeaned * window

    # one-sided FFT power spectrum
    spectrum = np.abs(rfft(y_win)) ** 2
    freqs = rfftfreq(n)          # [0, 1/n, 2/n, …, 1/2]

    # drop DC (index 0) and Nyquist
    spectrum = spectrum[1:]
    freqs = freqs[1:]

    if len(spectrum) == 0:
        return m, 0.0

    # Restrict search to periods in [2, min(n//2, 2*n//3)]
    min_freq = 1.0 / min(n // 2, max(n * 2 // 3, 2))
    valid = freqs >= min_freq
    if not np.any(valid):
        valid = np.ones(len(freqs), dtype=bool)

    peak_idx_local = int(np.argmax(spectrum[valid]))
    peak_idx = int(np.where(valid)[0][peak_idx_local])
    peak_freq = float(freqs[peak_idx])
    dominant_period = max(2, int(round(1.0 / peak_freq)) if peak_freq > 0 else m)

    # Normalised spectral entropy: Ω = 1 − H/H_max
    p = spectrum / spectrum.sum()
    p = np.clip(p, 1e-12, None)
    H = float(-np.dot(p, np.log(p)))
    H_max = float(np.log(len(spectrum)))
    spectral_score = float(np.clip(1.0 - H / H_max, 0.0, 1.0)) if H_max > 0 else 0.0

    return dominant_period, spectral_score


def _compute_stl(y: np.ndarray, period: int, n: int) -> tuple[float, float]:
    """
    Run STL once; return (seasonality_strength, trend_strength).

    Both require Var(resid) / Var(component + resid).
    """
    if n < 2 * period or period < 2:
        return 0.0, 0.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # seasonal_jump / trend_jump accelerate LOESS passes while keeping robust=True
            result = STL(y, period=period, robust=True,
                         seasonal_jump=2, trend_jump=2).fit()

        resid = result.resid
        seasonal = result.seasonal
        trend = result.trend

        var_resid = float(np.var(resid))

        seas_resid = seasonal + resid
        var_sr = float(np.var(seas_resid))
        ss = float(np.clip(1.0 - var_resid / var_sr, 0.0, 1.0)) if var_sr > 1e-12 else 0.0

        trend_resid = trend + resid
        var_tr = float(np.var(trend_resid))
        ts = float(np.clip(1.0 - var_resid / var_tr, 0.0, 1.0)) if var_tr > 1e-12 else 0.0

        return ss, ts

    except ValueError:
        return 0.0, 0.0


def _rolling_stats(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rolling_mean, rolling_std) of length len(y), NaN for first window-1 entries."""
    n = len(y)
    if n < window:
        return np.full(n, np.nan), np.full(n, np.nan)

    views = sliding_window_view(y, window)              # shape (n - window + 1, window)
    rmean = np.mean(views, axis=1)
    rstd = np.std(views, axis=1, ddof=1)

    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, rmean]), np.concatenate([pad, rstd])


def _compute_variance_type(rmean: np.ndarray, rstd: np.ndarray) -> str:
    """'mul' if Pearson corr(rolling_std, rolling_mean) > 0.7, else 'add'."""
    valid = ~(np.isnan(rmean) | np.isnan(rstd))
    if valid.sum() < 3:
        return "add"
    corr, _ = sp_stats.pearsonr(rstd[valid], rmean[valid])
    return "mul" if corr > 0.7 else "add"


def _compute_trend_instability(rmean: np.ndarray) -> float:
    """Ratio of sign-changes in diff(rolling_mean)."""
    rm = rmean[~np.isnan(rmean)]
    if len(rm) < 3:
        return 0.0
    d1 = np.diff(rm)
    sign_changes = int(np.sum(np.diff(np.sign(d1)) != 0))
    return float(sign_changes / len(rm))


def _compute_demand_metrics(y: np.ndarray) -> tuple[float, float]:
    """Return (p_interdemand, cv2_demand) for sparse/intermittent series."""
    nz_idx = np.flatnonzero(~np.isnan(y) & (y != 0.0))

    if len(nz_idx) < 2:
        p = float(len(y)) if len(nz_idx) == 0 else float(len(y))
        return p, 0.0

    intervals = np.diff(nz_idx).astype(float)
    p_interdemand = float(np.mean(intervals))

    demand_vals = y[nz_idx]
    mean_d = float(np.mean(demand_vals))
    if mean_d == 0.0:
        cv2_demand = 0.0
    else:
        cv2_demand = float((np.std(demand_vals) / mean_d) ** 2)

    return p_interdemand, cv2_demand


def _compute_obsolescence(y: np.ndarray, n: int) -> bool:
    """True if non-zero demand frequency is meaningfully lower in second half."""
    nz = np.flatnonzero(~np.isnan(y) & (y != 0.0))
    if len(nz) < 4:
        return False

    mid = len(nz) // 2
    pivot = nz[mid]

    if pivot == 0:
        return False
    denom_second = n - pivot
    if denom_second <= 0:
        return False

    freq_first = mid / pivot
    freq_second = (len(nz) - mid) / denom_second
    return bool(freq_second < 0.7 * freq_first)
