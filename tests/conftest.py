"""Shared fixtures for ts_triage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_index(n: int, freq: str = "M") -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=n, freq=freq)


@pytest.fixture
def monthly_seasonal():
    """Monthly series with clear annual seasonality and mild trend (n=120)."""
    rng = np.random.default_rng(42)
    n = 120
    t = np.arange(n)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    trend = 0.05 * t
    noise = rng.normal(0, 0.5, n)
    y = 10 + trend + seasonal + noise
    return pd.Series(y, index=_make_index(n, "ME"))


@pytest.fixture
def daily_series():
    """Daily series with weekly seasonality (n=365)."""
    rng = np.random.default_rng(7)
    n = 365
    t = np.arange(n)
    seasonal = 3 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 0.3, n)
    y = 20 + seasonal + noise
    return pd.Series(y, index=_make_index(n, "D"))


@pytest.fixture
def sparse_series():
    """Sparse intermittent demand series (sparsity > 0.7)."""
    rng = np.random.default_rng(99)
    n = 100
    mask = rng.random(n) < 0.25   # ~75% zeros
    y = np.where(mask, rng.exponential(5, n), 0.0)
    return pd.Series(y, index=_make_index(n, "W"))


@pytest.fixture
def short_series():
    """Very short series (n=10)."""
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0])
    return pd.Series(y, index=_make_index(10, "ME"))


@pytest.fixture
def random_walk():
    """Pure random walk — no structure (spectral_score should be low)."""
    rng = np.random.default_rng(13)
    n = 200
    y = np.cumsum(rng.normal(0, 1, n))
    return pd.Series(y, index=_make_index(n, "D"))


@pytest.fixture
def series_with_nans():
    """Monthly series with 15% NaN."""
    rng = np.random.default_rng(55)
    n = 96
    t = np.arange(n)
    y = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)
    nan_mask = rng.random(n) < 0.15
    y[nan_mask] = np.nan
    return pd.Series(y, index=_make_index(n, "ME"))
