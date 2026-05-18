"""
Decision tree: TSProfile → ModelRecommendation.

Rules applied in priority order:
1. length < 24          → naive (variant='mean')
2. spectral_score < 0.15 → naive (variant='last'), confidence=0.0
3. sparsity > 0.5       → croston (variant via SBC matrix; may reclassify to ETS)
4. horizon > 2×m        → prophet
5. missing_rate > 0.1   → prophet
6. default              → ets
"""

from typing import Optional

from .schemas import ModelRecommendation, TSProfile


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def decide(
    profile: TSProfile,
    m: int,
    horizon: Optional[int] = None,
) -> ModelRecommendation:
    """Select the best model and variant for *profile*."""

    p = profile

    # --- Rule 1: too short for any seasonal model ---------------------------
    if p.length < 24:
        return ModelRecommendation(
            model="naive",
            variant="mean",
            confidence=1.0,
            fallback="naive",
        )

    # --- Rule 2: no detectable structure → random walk ----------------------
    if p.spectral_score < 0.15:
        return ModelRecommendation(
            model="naive",
            variant="last",
            confidence=0.0,
            fallback="naive",
        )

    # --- Rule 3: sparse / intermittent demand --------------------------------
    if p.sparsity > 0.5:
        model, variant, confidence = _sbc_variant(p)
        return ModelRecommendation(
            model=model,
            variant=variant,
            confidence=confidence,
            fallback="naive",
        )

    # --- Rule 4: long horizon → Prophet handles complex seasonality ----------
    if horizon is not None and horizon > 2 * m:
        return ModelRecommendation(
            model="prophet",
            variant="prophet",
            confidence=0.8,
            fallback="ets",
        )

    # --- Rule 5: heavy gaps → Prophet is robust to irregular observations ---
    if p.missing_rate > 0.1:
        return ModelRecommendation(
            model="prophet",
            variant="prophet",
            confidence=0.7,
            fallback="naive",
        )

    # --- Default: ETS --------------------------------------------------------
    variant = _ets_variant_code(p)
    return ModelRecommendation(
        model="ets",
        variant=variant,
        confidence=0.8,
        fallback="naive",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sbc_variant(profile: TSProfile) -> tuple[str, str, float]:
    """
    Syntetos-Boylan-Croston (SBC) matrix.

    Returns (model, variant, confidence).
    Cells that reclassify to ETS return model='ets'.
    """
    p_id = profile.p_interdemand   # mean inter-demand interval
    cv2 = profile.cv2_demand

    if p_id < 1.32 and cv2 < 0.49:
        # Smooth demand, short intervals → standard ETS works well
        return "ets", _ets_variant_code(profile), 0.65

    if p_id >= 1.32 and cv2 < 0.49:
        # Intermittent but smooth demand size → SBA
        return "croston", "SBA", 0.75

    if p_id < 1.32 and cv2 >= 0.49:
        # Short intervals but lumpy demand → ETS with additive error
        # (multiplicative error would amplify the high variability)
        return "ets", _ets_variant_code(profile, force_add_error=True), 0.65

    # p_id >= 1.32 and cv2 >= 0.49 → lumpy intermittent
    if profile.obsolescence_flag:
        return "croston", "TSB", 0.70
    return "croston", "SBA", 0.72


def _ets_variant_code(profile: TSProfile, force_add_error: bool = False) -> str:
    """
    Build the 3-character ETS code (error-trend-seasonal) from profile metrics.

    Returned as a string like 'ANN', 'AAN', 'MAN', 'AAA', etc.
    This is a lightweight preview; param_infer fills in the full ModelParams.
    """
    error_char = "A" if (force_add_error or profile.variance_type == "add") else "M"
    trend_char = "A" if profile.trend_strength > 0.1 else "N"
    seas_char = (
        ("M" if profile.variance_type == "mul" else "A")
        if profile.seasonality_strength > 0.2
        else "N"
    )
    return f"{error_char}{trend_char}{seas_char}"
