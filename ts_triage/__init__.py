"""
ts_triage — lightweight automatic time series model selection.

Public API
----------
triage_pipeline : main entry-point
TSProfile       : statistical profile of the input series
ModelRecommendation : selected model + variant
ModelParams     : inferred hyperparameters
TriageResult    : combined output (profile + recommendation + params + optional fit)
TSInforecastableError : raised when strict=True and series is unforecastable
"""

from .exceptions import TSInforecastableError
from .pipeline import triage_pipeline
from .schemas import ModelParams, ModelRecommendation, TriageResult, TSProfile

__all__ = [
    "triage_pipeline",
    "TSProfile",
    "ModelRecommendation",
    "ModelParams",
    "TriageResult",
    "TSInforecastableError",
]
