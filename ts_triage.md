---
title: "PRD — ts_triage v1.0"
created: 2026-03-20
updated: 2026-04-20
status: partial_v0.1
tags: [ml, forecasting, python, hipay, data-engineering]
---

# PRD — `ts_triage` v1.0
## Librairie Python de triage automatique de séries temporelles

> **État d'implémentation (2026-04-20)** — code livré en `0.1.0`. Le cœur profiler + triage + param_infer + models synchrones est en place ; les volets AMI, horizon optimal, refit policy et config par fréquence restent au backlog. Légende utilisée dans ce document :
>
> - ✅ **Implémenté** — conforme au PRD
> - ⚠️ **Partiel / divergent** — présent mais diffère du PRD ; la valeur PRD originale est conservée entre parenthèses
> - ❌ **TODO / Backlog** — non implémenté, contenu du PRD conservé tel quel comme spec cible
>
> Pour les contradictions techniques, le texte principal décrit le code actuel, suivi de `*(à tester : <solution PRD d'origine>)*` quand une alternative reste à évaluer.

---

## 1. Contexte & Motivation

En environnement payfac (HiPay), des centaines à milliers de séries temporelles doivent être prévues en continu : volumes de transactions par marchand, par pays, par canal. Les challenges opérationnels sont :

- **Hétérogénéité extrême** des séries : certaines sont lisses et régulières, d'autres éparses, intermittentes, ou chaotiques
- **Coût de compute** : appliquer Prophet ou un modèle lourd à toutes les séries indistinctement est prohibitif
- **Ré-entraînements inutiles** : sans signal de staleness, on ré-entraîne par calendrier, pas par nécessité

`ts_triage` résout ces trois problèmes via un pipeline de diagnostic pré-modélisation rapide, analytique, et sans GPU.

---

## 2. Vision & Objectifs

`ts_triage` est une librairie Python légère qui, pour toute série temporelle univariée :

1. ✅ Calcule un profil statistique minimal (**TSProfile**)
2. ✅ Sélectionne le meilleur algorithme parmi 4 candidats : Naïf, Croston, ETS, Prophet
3. ✅ Infère analytiquement les hyperparamètres optimaux — sans grid search ni cross-validation
4. ❌ **TODO** — Recommande un **horizon optimal de prévision** basé sur la décroissance de l'information mutuelle (AMI)
5. ❌ **TODO** — Détermine si un modèle existant doit être **ré-entraîné ou réutilisé**

### Principes non-négociables

| Principe | Règle |
|---|---|
| Zéro GPU | Aucun deep learning, aucune dépendance GPU |
| Inférence analytique | Hyperparamètres déduits des métriques, jamais cherchés |
| Performance | Profiling complet T=500 en < 50ms (hors fit modèle) |
| Une seule passe | STL exécuté une fois, FFT exécuté une fois — résultats partagés |
| Prophet optionnel | Lazy import, jamais chargé si non sélectionné |
| Déterminisme | Mêmes inputs → mêmes outputs, toujours |

---

## 3. Architecture du module

```
ts_triage/
├── schemas.py        # ✅ Dataclasses + .to_dict() JSON-ready
├── profiler.py       # ✅ TSProfile — numpy/scipy pur, vectorisé, zéro pandas dans les boucles
├── ami.py            # ❌ TODO — AMI profile via kNN (sklearn.mutual_info_regression)
├── triage.py         # ✅ Règles de décision → ModelRecommendation
├── param_infer.py    # ✅ TSProfile → ModelParams (100% analytique)
├── refit_policy.py   # ❌ TODO — Checks de staleness → RefitDecision
├── models.py         # ⚠️ Wrappers fonctionnels (ETS, Prophet, Croston, Naïf) — cf. §10
├── pipeline.py       # ✅ Points d'entrée publics
├── exceptions.py     # ✅ TSInforecastableError
└── config.py         # ❌ TODO — Seuils configurables par fréquence
```

---

## 4. Interface publique

### 4.1 Input — ⚠️ Partiel

Le code expose des kwargs directs plutôt qu'un dataclass `TimeSeriesInput`.

```python
def triage_pipeline(
    y:       pd.Series,        # index: DatetimeIndex, valeurs: float
    freq:    str | None = None,
    horizon: int | None = None,
    fit:     bool = False,
    strict:  bool = False,
) -> TriageResult: ...
```

*(PRD initial : dataclass `TimeSeriesInput` avec les mêmes champs — à réintroduire en ❌ TODO si on veut standardiser l'entrée pour batch/serving.)*

### 4.2 Points d'entrée

```python
# ✅ Implémenté
def triage_pipeline(y, freq=None, horizon=None, fit=False, strict=False) -> TriageResult: ...

# ❌ TODO
def check_refit(
    y_new:         pd.Series,    # nouvelles observations depuis dernier fit
    cached_result: TriageResult, # résultat du dernier triage
) -> RefitDecision: ...
```

### 4.3 `TriageResult` — ⚠️ Partiel

```python
@dataclass
class TriageResult:
    profile:           TSProfile              # ✅
    recommendation:    ModelRecommendation    # ✅
    params:            ModelParams            # ✅
    fitted_model:      Any | None             # ✅ si fit=True
    forecast:          pd.Series | None       # ✅ si fit=True et horizon fourni
    aic:               float | None           # ✅ ETS seulement
    # horizon_analysis: HorizonAnalysis       # ❌ TODO (cf. §6)
    # fit_timestamp:    datetime | None       # ❌ TODO (requis pour check_refit, cf. §9)
    def to_dict(self) -> dict: ...            # ✅ JSON-sérialisable
```

---

## 5. `TSProfile` — métriques brutes

STL et FFT sont exécutés **une seule fois** chacun. Tous les calculs dépendants réutilisent les mêmes objets.

### 5.1 Table des métriques

| Champ | Statut | Type | Calcul | Dépendances | Coût |
|---|---|---|---|---|---|
| `length` | ✅ | int | `len(y)` | — | O(1) |
| `effective_length` | ✅ | int | `(y.notna() & y!=0).sum()` | — | O(n) |
| `missing_rate` | ✅ | float | `y.isna().mean()` | — | O(n) |
| `sparsity` | ✅ | float | `(y==0).mean()` | — | O(n) |
| `dominant_period` | ✅ | int | Pic FFT post Hann window | FFT | O(n log n) |
| `spectral_score` | ✅ | float | Ω ∈ [0,1] entropie spectrale normalisée | FFT | gratuit |
| `seasonality_strength` | ✅ | float | `1 - Var(R)/Var(S+R)` via STL | STL | O(n) |
| `trend_strength` | ✅ | float | `1 - Var(R)/Var(T+R)` via STL | STL | O(n) |
| `variance_type` | ✅ | str | `'add'\|'mul'` via corr(rolling_std, rolling_mean) | dominant_period | O(n) |
| `trend_instability` | ✅ | float | Ratio sign-changes dans diff(rolling_mean) | dominant_period | O(n) |
| `p_interdemand` | ✅ | float | Moyenne inter-demand intervals | sparsity | O(n) |
| `cv2_demand` | ✅ | float | `Var(y_nz) / mean(y_nz)²` | sparsity | O(n) |
| `obsolescence_flag` | ✅ | bool | Fréquence nz décroissante en 2ème moitié | sparsity | O(n) |
| `ami_profile` | ❌ TODO | list[float] | AMI(h) h=1..h_max via kNN k=8 | — | O(n log n)×h_max |
| `ami_h1` | ❌ TODO | float | AMI(h=1) | ami_profile | gratuit |
| `ami_at_horizon` | ❌ TODO | float\|None | AMI(h=horizon) | ami_profile | gratuit |
| `ami_decay_rate` | ❌ TODO | float | Pente normalisée [0,1] sur [1, h_max] | ami_profile | O(h_max) |

### 5.2 Snippets de calcul clés

```python
# ✅ Unique appel STL — résultats partagés
# (code: STL(y_filled, period=stl_period, robust=True, seasonal_jump=2, trend_jump=2).fit())
stl = STL(y_filled, period=dominant_period, robust=True).fit()

# ❌ TODO — AMI profile
from sklearn.feature_selection import mutual_info_regression
h_max = min(2 * dominant_period, (horizon or 12) * 2, len(y) // 5)
ami_profile = [
    mutual_info_regression(y[:-h].reshape(-1, 1), y[h:], n_neighbors=8)[0]
    for h in range(1, h_max + 1)
]

# ✅ variance_type — O(n) (code: vectorisation numpy via sliding_window_view, pas rolling pandas)
corr = pearsonr(y.rolling(m).std().dropna(), y.rolling(m).mean().dropna())[0]
variance_type = 'mul' if corr > 0.7 else 'add'

# ✅ trend_instability — O(n)
rm = y.rolling(m).mean().dropna()
trend_instability = (np.diff(np.sign(np.diff(rm.values))) != 0).sum() / len(rm)

# ✅ obsolescence_flag — O(n) (code: seuil 0.7, garde-fous supplémentaires quand pivot=0 ou len(nz)<4)
nz = np.flatnonzero(y.values)
mid = len(nz) // 2
freq_first  = mid / nz[mid]
freq_second = (len(nz) - mid) / (len(y) - nz[mid])
obsolescence_flag = freq_second < 0.7 * freq_first
```

---

## 6. `HorizonAnalysis` — horizon optimal de prévision — ❌ TODO (section entière)

Fondement théorique : Catt (2026) — *The Knowable Future*.
AMI(h) décroît avec h. L'horizon optimal est le dernier h où le signal reste exploitable (> 20% du pic).

```python
@dataclass
class HorizonAnalysis:
    h_optimal:         int
    ami_profile:       list[float]
    horizon_requested: int | None
    horizon_feasible:  bool        # horizon_requested <= h_optimal
    horizon_warning:   str | None  # ex: "Requested h=24, reliable up to h=7"
    ami_reliable:      bool        # False si freq='D'
    modelling_effort:  str         # 'full' | 'standard' | 'baseline_only'
```

```python
peak = max(ami_profile)
threshold = peak * 0.20  # relatif à la série
h_optimal = max(h for h, v in enumerate(ami_profile, 1) if v > threshold)
```

| `ami_at_horizon` | `modelling_effort` | Signification |
|---|---|---|
| > 0.3 | `'full'` | Signal fort — investir en modélisation |
| 0.1 – 0.3 | `'standard'` | ETS/Prophet standard |
| < 0.1 | `'baseline_only'` | Naïf suffit |

**Fiabilité AMI par fréquence** (Catt 2026, Table 2, Spearman ρ ETS sur M4) :

| Fréquence | ρ AMI–sMAPE | `ami_reliable` |
|---|---|---|
| Weekly | −0.66 | True |
| Quarterly | −0.55 | True |
| Yearly | −0.55 | True |
| Monthly | −0.34 | True |
| Hourly | −0.29 | True |
| **Daily** | **−0.14** | **False** |

---

## 7. `ModelRecommendation` — ⚠️ Partiel

```python
@dataclass
class ModelRecommendation:
    model:            str        # ✅ 'naive' | 'croston' | 'ets' | 'prophet'
    variant:          str        # ✅ 'last'|'mean' / 'SBA'|'TSB'|'classic' / 'AAN'|'AAA'|...
    confidence:       float      # ✅ [0,1] distance aux seuils de triage
    fallback:         str        # ✅ modèle si fit échoue
    # modelling_effort: str      # ❌ TODO — relay de HorizonAnalysis
    # reasoning:        list[str]  # ❌ TODO — audit trail des règles déclenchées
```

### Arbre de décision — ordre effectif du code (⚠️ ordre divergent)

```
1. length < 24                          → naive(mean), confidence=1.0
2. spectral_score < 0.15                → naive(last), confidence=0.0
3. sparsity > 0.5                       → croston (variant via SBC matrix ; peut reclassifier ETS)
4. horizon > 2 × dominant_period        → prophet
5. missing_rate > 0.1                   → prophet
6. sinon                                → ets(variant dérivé axe par axe — cf. ci-dessous)
```

*(à tester : ordre PRD d'origine — `missing_rate > 0.1` **avant** `horizon > 2m`, avec en plus une règle `modelling_effort == 'baseline_only' → naive(last)` insérée après la sparsité, et un arbre ETS final explicite en 3 branches : `seasonality_strength > 0.4 → AAA|MAA`, `trend_strength > 0.1 → AAN|MAN`, sinon `ANN`.)*

### Génération du variant ETS (code actuel)

Les 3 axes sont évalués indépendamment puis concaténés (`error` + `trend` + `seas`) :

| Axe | Règle code | Valeurs |
|---|---|---|
| `error` | `'A'` si `variance_type == 'add'` (ou SBC force additif), sinon `'M'` | A / M |
| `trend` | `'A'` si `trend_strength > 0.1`, sinon `'N'` | A / N |
| `seas` | si `seasonality_strength > 0.2` → `'M'` si `variance_type == 'mul'` sinon `'A'` ; sinon `'N'` | M / A / N |

*(à tester : restreindre la sortie au sous-ensemble PRD `AAA | MAA | AAN | MAN | ANN` via un arbre explicite, et relever le seuil saisonnalité à `0.4` — le code combine actuellement les 3 axes librement, donc peut produire `MAM`, `ANM`, `MNM`, etc., et déclenche le composant saisonnier dès `> 0.2`.)*

### SBC Matrix — sélection variant Croston ✅

```
p < 1.32  & CV² < 0.49  → Smooth    → reclassé ETS (variant axe-par-axe)
p ≥ 1.32  & CV² < 0.49  → SBA
p < 1.32  & CV² ≥ 0.49  → Erratic   → reclassé ETS, error forcé 'add'
p ≥ 1.32  & CV² ≥ 0.49  → TSB si obsolescence_flag else SBA
```

---

## 8. `ModelParams` — hyperparamètres inférés analytiquement

Zéro grid search. Zéro CV. Tout dérivé du TSProfile.

### ETS (`statsmodels.tsa.holtwinters.ExponentialSmoothing`) — ⚠️ Partiel

Backend actuel : **Holt-Winters classique** (`tsa.holtwinters.ExponentialSmoothing`). *(à tester : backend state-space `statsmodels.tsa.exponential_smoothing.ets.ETSModel` — API et AIC différents, comportement sur erreur multiplicative plus robuste selon FPP3.)*

| Paramètre | Statut | Inférence | Source |
|---|---|---|---|
| `error` | ✅ | `'mul'` si variance_type='mul' else `'add'` (forcé `'add'` si SBC erratic) | variance_type |
| `trend` | ✅ | `'add'` si trend_strength > 0.1 else None | trend_strength |
| `damped_trend` | ✅ | True par défaut (FPP3) ; False si horizon ≤ m/2 | horizon |
| `seasonal` | ✅ | `'add'`\|`'mul'` si seasonality_strength > 0.2 else None | seasonality_strength |
| `seasonal_periods` | ✅ | dominant_period | FFT |
| `initialization_method` | ✅ | `'heuristic'` si T > 200 else `'estimated'` | length |

*α, β, γ : laissés à l'optimisation MLE interne de statsmodels.*

*(écart non documenté au PRD initial : au moment du fit, si `seasonal == 'mul'` et que la série contient des valeurs ≤ 0, le code rétrograde silencieusement vers `seasonal = 'add'` pour éviter une erreur statsmodels. PRD initial : pas de garde-fou — à documenter ou à remplacer par une exception explicite.)*

### Prophet

| Paramètre | Inférence | Source |
|---|---|---|
| `growth` | `'flat'` si trend_strength < 0.05 else `'linear'` | trend_strength |
| `seasonality_mode` | `'multiplicative'` si variance_type='mul' else `'additive'` | variance_type |
| `changepoint_prior_scale` | <0.05→0.01 / 0.05–0.15→0.05 / 0.15–0.30→0.15 / >0.30→0.30 | trend_instability |
| `seasonality_prior_scale` | `1 + 19 × seasonality_strength` → [1, 20] | seasonality_strength |
| `changepoint_range` | 0.85 si trend_instability > 0.20 else 0.80 | trend_instability |
| `n_changepoints` | T // 4 si T < 100 else 25 | length |
| `yearly_seasonality` | dominant_period ≈ 365(D) ou 12(M) | dominant_period, freq |
| `weekly_seasonality` | dominant_period ≈ 7(D) | dominant_period, freq |

### Croston

| Paramètre | Inférence | Source |
|---|---|---|
| `variant` | SBC matrix | p_interdemand, cv2_demand, obsolescence_flag |
| `alpha` | `2/(n_nonzero+1)` borné [0.05, 0.30] | effective_length |
| `beta` | `alpha/2` (TSB seulement) | alpha |

---

## 9. Politique de ré-entraînement — `refit_policy.py` — ❌ TODO (section entière)

Objectif : **ne ré-entraîner que quand c'est nécessaire**, pas par calendrier fixe.

### `RefitDecision`

```python
@dataclass
class RefitDecision:
    should_refit:  bool
    urgency:       str        # 'immediate' | 'scheduled' | 'none'
    reasons:       list[str]
    triggered_by:  list[str]
    refit_full:    bool       # True = triage complet / False = re-fit params inchangés
```

### Les 5 checks de staleness — tous O(n), aucun fit

| Check | Signal | Implémentation | `refit_full` |
|---|---|---|---|
| **Staleness temporelle** | Age > max_age[freq] | Comparaison timestamps | False |
| **Drift niveau** | CUSUM > k×σ | `np.cumsum(residuals - mean)` | False |
| **Drift saisonnalité** | ΔseasonalityStrength > 0.15 | STL léger sur y_new | True |
| **Drift AMI** | Δami_h1 > threshold[freq] | `mutual_info_regression` | True |
| **Drift sparsité** | Δsparsity > 0.10 | `(y_new==0).mean()` | True |

### Matrice de décision

| Triggers actifs | `should_refit` | `urgency` | `refit_full` |
|---|---|---|---|
| Aucun | False | `'none'` | — |
| age seul | True | `'scheduled'` | False |
| drift_niveau seul | True | `'scheduled'` | False |
| drift_saisonnalité | True | `'immediate'` | True |
| drift_ami | True | `'immediate'` | True |
| drift_sparsité | True | `'immediate'` | True |
| ≥ 2 triggers | True | `'immediate'` | True |

### `config.py`

```python
REFIT_CONFIG = {
    'D': {'max_age': 30,  'cusum_k': 4, 'ami_drift_threshold': 0.15},
    'W': {'max_age': 8,   'cusum_k': 4, 'ami_drift_threshold': 0.15},
    'M': {'max_age': 3,   'cusum_k': 3, 'ami_drift_threshold': 0.12},
    'Q': {'max_age': 2,   'cusum_k': 3, 'ami_drift_threshold': 0.12},
    'Y': {'max_age': 1,   'cusum_k': 2, 'ami_drift_threshold': 0.10},
}

TRIAGE_CONFIG = {
    'D': {'ami_weight': 0.3, 'spectral_weight': 0.7},  # Daily: AMI peu fiable (ρ≈-0.14)
    'W': {'ami_weight': 0.9, 'spectral_weight': 0.1},
    'M': {'ami_weight': 0.6, 'spectral_weight': 0.4},
    'Q': {'ami_weight': 0.8, 'spectral_weight': 0.2},
    'Y': {'ami_weight': 0.8, 'spectral_weight': 0.2},
}
```

---

## 10. `models.py` — wrappers — ⚠️ Partiel

Implémentation actuelle : **wrappers fonctionnels** (`fit_and_forecast` + helpers privés `_fit_naive`, `_fit_ets`, `_fit_croston`, `_fit_prophet`) plutôt qu'une hiérarchie `BaseModel`. Le fallback Naïf est centralisé dans `fit_and_forecast`.

```python
# ✅ Code actuel
def fit_and_forecast(
    y: pd.Series,
    rec: ModelRecommendation,
    params: ModelParams,
    horizon: Optional[int],
) -> tuple[Any, Optional[pd.Series], Optional[float]]:
    try:
        if rec.model == "naive":   return _fit_naive(y, rec.variant, horizon)
        if rec.model == "ets":     return _fit_ets(y, params, horizon)
        if rec.model == "croston": return _fit_croston(y, params, horizon)
        if rec.model == "prophet": return _fit_prophet(y, params, horizon)
    except Exception as exc:
        logger.warning("Model fit failed (%s). Falling back to naive.", exc)
        return _fit_naive(y, "last", horizon)
```

*(à tester : hiérarchie `BaseModel` de classes — `NaiveModel / CrostonModel / ETSModel / ProphetModel` avec contrat `fit/predict/aic`. Avantage : testabilité unitaire par modèle, introspection plus facile. Coût : plus de cérémonie pour un pipeline déjà linéaire.)*

*(PRD initial — log structuré sous forme de dict :*

```python
log.warning({"event": "model_fit_failed", "model": model.__class__.__name__, "error": str(e)})
```

*Code actuel : printf-style `logger.warning("Model fit failed (%s). ...", exc)`. À harmoniser avec §15.)*

---

## 11. `exceptions.py` — ✅ Implémenté

```python
class TSInforecastableError(Exception):
    def __init__(self, reason: str, value: float, threshold: float):
        self.reason    = reason     # 'low_spectral_score' | 'insufficient_length' | 'too_sparse'
        self.value     = value
        self.threshold = threshold
```

---

## 12. Flux d'exécution complet

```
triage_pipeline(y, freq, horizon, fit, strict)
  ├── 1. _infer_freq(y)                             O(1)           ✅
  ├── 2. profiler.compute(y, m)                     (profile)      ✅
  │     ├── basic_stats (length, missing, sparsity)  O(n)
  │     ├── fft_features()        ← unique FFT      O(n log n)
  │     ├── stl_features()        ← unique STL      O(n)
  │     ├── variance_features()                     O(n)
  │     └── intermittent_features()                 O(n)
  ├── 3. _check_inforecastable(profile) si strict                   ✅
  ├── 4. triage.decide(profile, m, horizon)         O(1)           ✅
  ├── 5. param_infer.infer(profile, rec, h, m)      O(1)           ✅
  └── 6. [si fit=True] models.fit_and_forecast      30ms – 5s      ✅

  ❌ TODO — étapes supplémentaires prévues au PRD d'origine :
  ├── ami.compute_profile(y, h_max)                 O(n log n) × h_max
  └── horizon_analysis(ami_profile, horizon)        O(h_max)

check_refit(y_new, cached_result)                   ❌ TODO (section entière)
  ├── check_age()                                   O(1)
  ├── check_level_drift()         CUSUM             O(n)
  ├── check_seasonality_drift()   STL léger         O(n)
  ├── check_ami_drift()           AMI(h=1)          O(n log n)
  ├── check_sparsity_drift()                        O(n)
  └── decide(triggered_checks)                      O(1)
```

---

## 13. Budget temps — ⚠️ cible, non benchmarké

Chiffres ci-dessous = objectifs PRD initiaux. Aucune mesure reproductible n'est actuellement incluse au repo ; à valider via `pytest-benchmark` ou équivalent. Note d'incohérence interne au PRD d'origine : §16 annonce *"Full pipeline < 20ms for T=500"*, cette table annonce *~30ms* — à réconcilier.


| Étape | T=100 | T=500 | T=2000 |
|---|---|---|---|
| Stats basiques | <0.1ms | <0.1ms | <0.5ms |
| FFT + Spectral | 0.2ms | 0.5ms | 2ms |
| STL | 2ms | 8ms | 30ms |
| Variance + trend | 0.1ms | 0.2ms | 0.5ms |
| Intermittent | 0.1ms | 0.1ms | 0.3ms |
| AMI profile (h_max=24) | 5ms | 20ms | 80ms |
| Triage + param infer | <0.1ms | <0.1ms | <0.1ms |
| **Total sans fit** | **~8ms** | **~30ms** | **~115ms** |
| check_refit | ~3ms | ~10ms | ~35ms |
| ETS fit | 20ms | 50ms | 200ms |
| Prophet fit | — | 2–5s | 5–10s |

---

## 14. Dépendances — ⚠️ Partiel

Extrait de `pyproject.toml` actuel :

```toml
[dependencies]
numpy = ">=1.23"         # (PRD initial : >=1.24)
scipy = ">=1.9"          # (PRD initial : >=1.10)
pandas = ">=1.5"         # (PRD initial : >=2.0)
statsmodels = ">=0.14"   # ✅ conforme
# scikit-learn            # ❌ absent — à ajouter (>=1.3) quand ami.py sera implémenté

[optional-dependencies]
prophet = ["prophet>=1.1"]   # ✅ lazy import effectif
dev     = ["pytest>=7.0", "pytest-cov"]   # ajout code, non spécifié au PRD
```

*(à tester : relever les versions minimales aux cibles PRD — peu coûteux si on n'a pas encore d'utilisateurs verrouillés sur pandas 1.x.)*

---

## 15. Contraintes non-fonctionnelles

| Contrainte | Statut | Règle |
|---|---|---|
| Pureté profiler | ✅ / ❌ | Zéro pandas dans `profiler.py` (✅) et `ami.py` (❌ module absent) — numpy pur |
| Idempotence | ✅ | Mêmes inputs → mêmes outputs, déterministe |
| Pas d'effets de bord | ✅ | Aucun état global mutable |
| Sérialisation | ✅ | Tout `TriageResult` JSON via `.to_dict()` |
| Logging | ⚠️ | Printf-style `logger.warning("... %s", exc)`, niveau WARNING *(à tester : dict structuré `{"event": ..., "model": ..., "error": ...}` comme spécifié au PRD initial)* |
| Prophet | ✅ | `from prophet import Prophet` uniquement dans `_fit_prophet()` |
| Tests | ✅ | Chaque module testable indépendamment, fixtures numpy pures |
| Fallback | ✅ | Tout échec de fit → Naïf automatique, jamais crash silencieux |

---

## 16. Intent

```
You are implementing `ts_triage`, a lightweight Python library for automatic
time series model selection and hyperparameter inference.

The goal is a frugal, fast, zero-GPU forecasting triage engine. Given a single
univariate time series (pandas Series with DatetimeIndex), the system must:
1. Compute a minimal set of statistical properties (TSProfile)
2. Select the best forecasting algorithm among 4 candidates only:
   Naive, Croston, ETS, Prophet
3. Infer near-optimal hyperparameters analytically — no grid search, no CV,
   no fitting at triage time
4. Recommend an optimal forecast horizon based on AMI decay
5. Determine whether an existing model needs retraining or can be reused

Philosophy:
- Frugal first. Metrics computed once, reused everywhere.
- STL run once, FFT run once. No redundant passes over the data.
- Hyperparameters inferred analytically. MLE in statsmodels ETS is the only
  search allowed.
- Fail explicitly: strict=True → raise TSInforecastableError.
- Prophet: lazy-import only, never loaded unless selected.

NOT: benchmarking framework, AutoML, backtesting, multivariate, deep learning.

Implementation rules:
- Pure numpy/scipy in profiler.py — no pandas in compute loops
- All public functions typed, return dataclasses with .to_dict()
- Full pipeline < 20ms for T=500 excluding model fit
- prophet never imported at module load time
- Each module independently testable with numpy-only fixtures
```

---

## 17. Références

- Catt, P. M. (2026). *The Knowable Future: Mapping the Decay of Past–Future Mutual Information Across Forecast Horizons*. UNITEC Institute of Technology.
- Wang, R., Klee, S., & Roos, A. (2025). *Time Series Forecastability Measures*. KDD '25, ACM.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.
- Syntetos, A. A., & Boylan, J. E. (2001). On the bias of intermittent demand estimates. *International Journal of Production Economics*.
- Teunter, R., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. *European Journal of Operational Research*.

---

## 18. Backlog consolidé (annexe, 2026-04-20)

Récap navigable des écarts identifiés entre PRD v1.0 et code v0.1.0. Les détails restent dans chaque section en amont ; aucune information n'est retirée.

### ❌ TODO / Backlog — features entières à livrer

| # | Feature | Section | Impact |
|---|---|---|---|
| 1 | Module `ami.py` (AMI via kNN, k=8) | §3, §5 | Bloque §6 |
| 2 | Champs `ami_*` dans `TSProfile` | §5.1 | Bloque §6 |
| 3 | `HorizonAnalysis` + `h_optimal` (Catt 2026) | §6 | Vision §2-4 |
| 4 | Module `refit_policy.py` + `check_refit()` + `RefitDecision` | §9 | Vision §2-5 |
| 5 | Module `config.py` — `REFIT_CONFIG`, `TRIAGE_CONFIG` par fréquence | §9 | Bloque §9 + pondération triage |
| 6 | Dataclass `TimeSeriesInput` (signature standardisée) | §4.1 | Ergonomie batch/serving |
| 7 | `ModelRecommendation.modelling_effort` (relay horizon) | §7 | Bloque règle triage PRD §7 étape 4 |
| 8 | `ModelRecommendation.reasoning: list[str]` (audit trail) | §7 | Observabilité |
| 9 | `TriageResult.fit_timestamp` | §4.3 | Pré-requis `check_refit` |
| 10 | Dépendance `scikit-learn>=1.3` | §14 | Conséquence de #1 |

### ⚠️ Contradictions techniques à rejouer (*à tester*)

| # | Sujet | Code actuel | PRD initial |
|---|---|---|---|
| A | Backend ETS | `tsa.holtwinters.ExponentialSmoothing` | `tsa.exponential_smoothing.ets.ETSModel` (state-space) |
| B | Ordre arbre triage | `horizon > 2m` avant `missing_rate > 0.1` | inverse + règle `baseline_only` insérée |
| C | Seuil saisonnalité variant ETS | `> 0.2` | `> 0.4` |
| D | Codes ETS générés | toutes combinaisons 3 axes (ex. `MAM`, `ANM`) | sous-ensemble `AAA\|MAA\|AAN\|MAN\|ANN` |
| E | Style logging | printf `logger.warning("... %s", exc)` | dict structuré `{"event": ..., "model": ..., "error": ...}` |
| F | Organisation `models.py` | wrappers fonctionnels | hiérarchie `BaseModel` |

### ⚠️ Écarts mineurs à surveiller

| # | Sujet | Code actuel | Note |
|---|---|---|---|
| a | Downgrade silencieux `seasonal='mul'`→`'add'` | présent dans `_fit_ets` si y ≤ 0 | PRD initial : non spécifié — décider : warning explicite ou exception |
| b | Alias fréquences `_freq_to_period` | legacy (`M`, `A`, `Q`, `Y`) + pandas modernes (`ME`, `MS`, `YE`, `YS`, `QE`, `QS`) | PRD initial : `D/W/M/Q/Y` uniquement |
| c | Versions deps min | `numpy>=1.23`, `scipy>=1.9`, `pandas>=1.5` | PRD initial : `>=1.24 / >=1.10 / >=2.0` |
| d | Budget temps §13 vs §16 | non benchmarké | incohérence interne PRD : 20ms (§16) vs 30ms (§13) pour T=500 |
| e | `seasonal_jump=2, trend_jump=2` dans STL | accélère LOESS avec `robust=True` | PRD initial : non spécifié — documenter ou restreindre |
