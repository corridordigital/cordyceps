# ts_triage

**Lightweight, analytical time-series model selection for Python.**

`ts_triage` profiles a univariate series, picks the right forecasting model, and infers its hyperparameters — *without* grid search, cross-validation, or fitting anything unless you ask it to. One STL pass, one FFT, one decision tree, one numpy-backed compute path.

It is designed for the real bottleneck in forecasting pipelines: **not training, but deciding what to train**.

---

## Why

Most AutoML forecasting libraries brute-force the decision: fit N candidate models, compare them, keep the winner. That is expensive, noisy on short series, and often picks the wrong family for the wrong reason (e.g. ETS-MMM fits best on training data but extrapolates badly).

`ts_triage` takes the opposite stance:

- Extract the **statistical signature** of the series (seasonality strength, trend strength, intermittency, spectral entropy, variance structure, obsolescence, …).
- Apply a **transparent decision tree** to pick a model family and variant (naive / Croston / ETS / Prophet).
- **Derive** hyperparameters from the profile (analytically — from established forecasting literature), not by search.
- Fitting is opt-in and isolated.

The output is a structured `TriageResult` you can log, diff, unit-test, and feed into whatever training/serving stack you already have.

---

## Install

```bash
pip install -e .
# optional, only if you want Prophet to actually run when selected:
pip install -e ".[prophet]"
# dev:
pip install -e ".[dev]"
```

Requires Python 3.10+. Core deps: `numpy`, `scipy`, `statsmodels`, `pandas`. Prophet is **lazy-imported** — it is never loaded unless it is the chosen model.

---

## Quick start

```python
import pandas as pd
from ts_triage import triage_pipeline

y = pd.Series(..., index=pd.date_range("2020-01-01", periods=120, freq="M"))

result = triage_pipeline(y, horizon=12, fit=True)

print(result.recommendation.model)     # 'ets'
print(result.recommendation.variant)   # e.g. 'AAA'
print(result.params.to_dict())         # inferred hyperparameters
print(result.forecast.head())          # only populated when fit=True
```

Or, when you only want the *decision* (zero fitting cost):

```python
result = triage_pipeline(y, horizon=12)   # fit=False by default
result.recommendation  # ModelRecommendation(...)
result.params          # ModelParams(...)
result.profile         # TSProfile(...)
```

---

## How it works

```
             ┌──────────────┐     ┌──────────┐     ┌──────────────┐     ┌────────────┐
  pd.Series ─┤  profiler    ├────▶│  triage  ├────▶│ param_infer  ├────▶│  models    │
             │  (STL+FFT)   │     │ (tree)   │     │ (analytical) │     │ (optional) │
             └──────────────┘     └──────────┘     └──────────────┘     └────────────┘
                   TSProfile      ModelRecommendation   ModelParams      fit+forecast
```

### 1. Profiler — [ts_triage/profiler.py](ts_triage/profiler.py)

Computes a `TSProfile` in a single pass:

| Metric | What it captures |
|---|---|
| `length` / `effective_length` | total vs. non-missing, non-zero length |
| `missing_rate`, `sparsity` | data quality signal |
| `dominant_period` | FFT peak (Hann-windowed, DC-removed) |
| `spectral_score` | normalised spectral entropy — 1 = periodic, 0 = noise |
| `seasonality_strength`, `trend_strength` | from a **single** STL decomposition |
| `variance_type` | `'add'` vs `'mul'` via corr(rolling std, rolling mean) |
| `trend_instability` | sign-change ratio of the smoothed trend |
| `p_interdemand`, `cv2_demand` | Syntetos–Boylan–Croston inputs |
| `obsolescence_flag` | product-lifecycle decay heuristic |

STL, FFT, and rolling stats each run exactly once — all downstream metrics reuse cached arrays.

### 2. Triage — [ts_triage/triage.py](ts_triage/triage.py)

A priority-ordered decision tree:

1. `length < 24` → **naive (mean)**
2. `spectral_score < 0.15` → **naive (last)** — no detectable structure
3. `sparsity > 0.5` → **Croston / SBA / TSB** via the SBC matrix (may reclassify to ETS for smooth/lumpy corners)
4. `horizon > 2 × m` → **Prophet** — long-horizon complex seasonality
5. `missing_rate > 0.1` → **Prophet** — robust to irregular observations
6. default → **ETS**, with a 3-char code (e.g. `AAA`, `MAN`, `ANN`) derived from the profile

### 3. Param inference — [ts_triage/param_infer.py](ts_triage/param_infer.py)

Hyperparameters come from the profile, not from search:

- **ETS** — error/trend/seasonal types from `variance_type`, `trend_strength`, `seasonality_strength`; damped trend enabled unless horizon is short vs. `m`; `initialization_method` switches to `heuristic` beyond 200 obs.
- **Prophet** — `changepoint_prior_scale` binned from `trend_instability`; `seasonality_prior_scale` linear in `seasonality_strength`; `growth=flat` when no trend; yearly/weekly toggles from dominant period.
- **Croston** — `alpha = clip(2/(n_nonzero+1), 0.05, 0.30)`; TSB adds `beta = alpha/2`.

### 4. Models (optional) — [ts_triage/models.py](ts_triage/models.py)

Thin wrappers over `statsmodels` (ETS), Prophet (lazy), a hand-rolled Croston/SBA/TSB, and naive. Any fit failure is caught and falls back to naive — `TriageResult` is never partially populated.

---

## Unforecastability

```python
from ts_triage import TSInforecastableError

try:
    triage_pipeline(y, strict=True)
except TSInforecastableError as e:
    print(e.reason, e.value, e.threshold)
    # reason ∈ {'low_spectral_score', 'insufficient_length', 'too_sparse'}
```

Without `strict=True`, the pipeline logs a warning and continues with naive — useful in batch pipelines where you'd rather ship a baseline than raise.

---

## Public API

```python
from ts_triage import (
    triage_pipeline,        # the one entry-point
    TSProfile,              # statistical signature
    ModelRecommendation,    # model + variant + confidence + fallback
    ModelParams,            # inferred hyperparameters
    TriageResult,           # all of the above + optional fitted_model / forecast / aic
    TSInforecastableError,
)
```

Every schema exposes `.to_dict()` for logging / JSON serialisation.

---

## Testing

```bash
pytest                # runs the full suite
pytest --cov=ts_triage
```

Tests live under [tests/](tests/) and cover profiler invariants, the triage decision tree, param inference boundaries, the end-to-end pipeline, the custom exception, and model fit/forecast contracts.

---

## Design principles

- **One pass, no waste.** STL and FFT are each computed once and shared downstream.
- **Transparent over clever.** The triage rule that fires is inspectable, not a black box.
- **Analytical > search.** Every hyperparameter has a closed-form derivation from profile metrics.
- **Fail-soft by default, strict on request.** `strict=True` raises; otherwise degrade to naive and log.
- **Lazy heavyweight deps.** Prophet is only imported when the triage actually selects it.

---

## Status

Version `0.1.0` — early but stable surface. The public API listed above is the contract; internals may change.
