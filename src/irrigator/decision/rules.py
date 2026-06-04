"""Irrigation decision logic.

Takes the current water balance state + short-term forecast and produces
a concrete recommendation: irrigate yes/no, how many mm, when.

The decision is rule-based and transparent — no black box.  The rules
follow FAO-56 principles:

1. If depletion is approaching the stress threshold (Dr → RAW),
   and no significant rain is forecast, recommend irrigation.
2. The dose refills to a target level that accounts for expected rain
   (don't refill to FC if rain is likely — waste of water).
3. During the critical crop window (flowering), tighten all thresholds.
4. Respect irrigation system constraints (min/max dose, available days).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import yaml

from irrigator.config import ParcelConfig
from irrigator.crop.phenology import CropParams
from irrigator.water_balance.state import WaterBalanceState

logger = logging.getLogger(__name__)


@dataclass
class IrrigationAdvice:
    """Actionable irrigation recommendation for a parcel."""

    date: date
    irrigate: bool
    dose_mm: float
    recommended_date: date | None
    reason: str
    confidence: str  # "high", "medium", "low"
    # Context
    current_depletion: float
    current_taw: float
    current_raw: float
    current_ks: float
    crop_stage: str
    forecast_precip_mm: float
    days_until_stress: int | None

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "irrigate": self.irrigate,
            "dose_mm": round(self.dose_mm, 1),
            "recommended_date": self.recommended_date.isoformat()
            if self.recommended_date
            else None,
            "reason": self.reason,
            "confidence": self.confidence,
            "current_depletion_mm": round(self.current_depletion, 1),
            "current_taw_mm": round(self.current_taw, 1),
            "stress_coefficient": round(self.current_ks, 3),
            "crop_stage": self.crop_stage,
            "forecast_precip_mm": round(self.forecast_precip_mm, 1),
            "days_until_stress": self.days_until_stress,
        }


@dataclass
class DecisionParams:
    """Parameters controlling the irrigation decision.

    Loaded from configs/model_parameters.yaml decision section.
    """

    # Forecast rain discount factors (how much of forecast rain to count on)
    rain_confidence_0_24h: float = 0.85
    rain_confidence_24_48h: float = 0.70
    rain_confidence_48_96h: float = 0.50
    # Refill targets (fraction of TAW)
    refill_no_rain: float = 0.95
    refill_rain_possible: float = 0.80
    refill_rain_likely: float = 0.65
    # Rain thresholds
    rain_possible_mm: float = 5.0
    rain_likely_mm: float = 15.0
    # Critical window adjustments
    critical_trigger_reduction: float = 0.10
    critical_refill_increase: float = 0.05
    # System constraints (from parcel config, set at runtime)
    max_dose_mm: float = 40.0
    min_dose_mm: float = 15.0
    min_interval_days: int = 3

    @classmethod
    def from_config(
        cls,
        model_params_path: str = "configs/model_parameters.yaml",
        parcel: ParcelConfig | None = None,
    ) -> DecisionParams:
        """Load from config files."""
        with open(model_params_path) as f:
            cfg = yaml.safe_load(f)

        d = cfg.get("decision", {})
        rain_conf = d.get("rain_confidence", {})
        refill = d.get("refill_target", {})
        critical = d.get("critical_window_adjustment", {})

        params = cls(
            rain_confidence_0_24h=rain_conf.get("arome_0_24h", 0.85),
            rain_confidence_24_48h=rain_conf.get("arome_24_48h", 0.70),
            rain_confidence_48_96h=rain_conf.get("arpege_48_96h", 0.50),
            refill_no_rain=refill.get("rain_unlikely", 0.95),
            refill_rain_possible=refill.get("rain_possible", 0.80),
            refill_rain_likely=refill.get("rain_likely", 0.65),
            critical_trigger_reduction=critical.get("trigger_reduction", 0.10),
            critical_refill_increase=critical.get("refill_increase", 0.05),
        )

        if parcel:
            irrig = parcel.irrigation
            params.max_dose_mm = irrig.get("max_dose_mm", 40.0)
            params.min_dose_mm = irrig.get("min_dose_mm", 15.0)
            params.min_interval_days = irrig.get("min_interval_days", 3)

        return params


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------


def _discount_forecast_precip(
    forecast_states: list[WaterBalanceState],
    params: DecisionParams,
) -> float:
    """Compute effective forecast precipitation with confidence discounting.

    Nearby forecasts are trusted more than distant ones.
    """
    total = 0.0
    for i, state in enumerate(forecast_states):
        day_offset = i  # day 0, 1, 2, 3...
        if day_offset < 1:
            discount = params.rain_confidence_0_24h
        elif day_offset < 2:
            discount = params.rain_confidence_24_48h
        else:
            discount = params.rain_confidence_48_96h
        total += state.precip * discount
    return total


def _is_critical_window(crop_state: str, crop_params: CropParams, gdd: float) -> bool:
    """Check if the crop is in the flowering/early grain fill critical window."""
    if crop_state != "mid":
        return False
    # Critical window: from tasseling to ~300 GDD after
    gdd_tasseling = crop_params.gdd_end_development
    return gdd_tasseling <= gdd <= gdd_tasseling + 300


def _find_driest_day(forecast_states: list[WaterBalanceState]) -> date | None:
    """Find the day with least forecast precipitation (best time to irrigate)."""
    if not forecast_states:
        return None
    driest = min(forecast_states, key=lambda s: s.precip)
    return driest.date


def compute_recommendation(
    current_state: WaterBalanceState,
    forecast_states: list[WaterBalanceState],
    crop_params: CropParams,
    params: DecisionParams,
    last_irrigation_date: date | None = None,
) -> IrrigationAdvice:
    """Compute irrigation recommendation from current state + forecast.

    Decision tree:
    1. If crop isn't growing (pre-emergence or mature): don't irrigate
    2. If minimum interval hasn't elapsed since last irrigation: don't irrigate
    3. If current depletion is below trigger: don't irrigate
    4. If forecast rain will prevent stress: don't irrigate
    5. Otherwise: irrigate, compute dose, find best day

    Parameters
    ----------
    current_state : today's WaterBalanceState
    forecast_states : forward balance under NWP forecast (from Block 5)
    crop_params : CropParams
    params : DecisionParams
    last_irrigation_date : date of most recent irrigation (or None)

    Returns
    -------
    IrrigationAdvice with the recommendation.
    """
    today = current_state.date
    dr = current_state.depletion
    taw = current_state.taw
    raw = current_state.raw
    ks = current_state.stress_coeff
    stage = current_state.crop_stage

    # Effective forecast precipitation
    effective_precip = _discount_forecast_precip(forecast_states, params)

    # Check if in critical window
    in_critical = _is_critical_window(stage, crop_params, current_state.gdd)

    # Adjust thresholds for critical window
    if in_critical:
        trigger_p = crop_params.p - params.critical_trigger_reduction
        refill_bonus = params.critical_refill_increase
    else:
        trigger_p = crop_params.p
        refill_bonus = 0.0

    trigger_dr = trigger_p * taw  # depletion level that triggers irrigation

    # --- Decision tree ---

    # 1. Not growing
    if stage in ("not_planted", "pre_emergence", "mature"):
        return IrrigationAdvice(
            date=today,
            irrigate=False,
            dose_mm=0,
            recommended_date=None,
            reason=f"Crop stage '{stage}' — no irrigation needed",
            confidence="high",
            current_depletion=dr,
            current_taw=taw,
            current_raw=raw,
            current_ks=ks,
            crop_stage=stage,
            forecast_precip_mm=effective_precip,
            days_until_stress=None,
        )

    # 2. Minimum interval
    if last_irrigation_date and (today - last_irrigation_date).days < params.min_interval_days:
        days_since = (today - last_irrigation_date).days
        return IrrigationAdvice(
            date=today,
            irrigate=False,
            dose_mm=0,
            recommended_date=None,
            reason=f"Irrigated {days_since}d ago — minimum interval is {params.min_interval_days}d",
            confidence="high",
            current_depletion=dr,
            current_taw=taw,
            current_raw=raw,
            current_ks=ks,
            crop_stage=stage,
            forecast_precip_mm=effective_precip,
            days_until_stress=None,
        )

    # 3. Depletion below trigger — no immediate need
    if dr < trigger_dr * 0.8:
        return IrrigationAdvice(
            date=today,
            irrigate=False,
            dose_mm=0,
            recommended_date=None,
            reason=f"Soil water adequate — depletion {dr:.0f} mm < trigger {trigger_dr:.0f} mm",
            confidence="high",
            current_depletion=dr,
            current_taw=taw,
            current_raw=raw,
            current_ks=ks,
            crop_stage=stage,
            forecast_precip_mm=effective_precip,
            days_until_stress=None,
        )

    # 4. Will forecast rain prevent stress?
    # Project: will depletion exceed trigger within forecast horizon?
    stress_in_forecast = any(s.stress_coeff < 0.9 for s in forecast_states)
    max_depletion_forecast = max((s.depletion for s in forecast_states), default=dr)

    if not stress_in_forecast and max_depletion_forecast < trigger_dr:
        return IrrigationAdvice(
            date=today,
            irrigate=False,
            dose_mm=0,
            recommended_date=None,
            reason=f"Forecast rain ({effective_precip:.0f} mm discounted) should prevent stress",
            confidence="medium",
            current_depletion=dr,
            current_taw=taw,
            current_raw=raw,
            current_ks=ks,
            crop_stage=stage,
            forecast_precip_mm=effective_precip,
            days_until_stress=None,
        )

    # 5. Irrigation needed — compute dose
    # Determine refill target based on expected rain
    if effective_precip >= params.rain_likely_mm:
        refill_frac = params.refill_rain_likely + refill_bonus
    elif effective_precip >= params.rain_possible_mm:
        refill_frac = params.refill_rain_possible + refill_bonus
    else:
        refill_frac = params.refill_no_rain + refill_bonus

    refill_frac = min(1.0, refill_frac)

    # Target depletion after irrigation
    target_dr = (1.0 - refill_frac) * taw
    dose = dr - target_dr - effective_precip
    dose = max(0, dose)

    # Apply system constraints
    dose = min(dose, params.max_dose_mm)
    if dose < params.min_dose_mm:
        # Not worth irrigating for such a small amount
        return IrrigationAdvice(
            date=today,
            irrigate=False,
            dose_mm=0,
            recommended_date=None,
            reason=f"Computed dose {dose:.0f} mm below minimum {params.min_dose_mm:.0f} mm",
            confidence="medium",
            current_depletion=dr,
            current_taw=taw,
            current_raw=raw,
            current_ks=ks,
            crop_stage=stage,
            forecast_precip_mm=effective_precip,
            days_until_stress=None,
        )

    # Find best day to irrigate
    recommended_day = _find_driest_day(forecast_states) or today

    # Days until stress
    stress_days = [s for s in forecast_states if s.stress_coeff < 0.9]
    days_to_stress = (stress_days[0].date - today).days if stress_days else None

    # Confidence
    if forecast_states and len(forecast_states) <= 2:
        confidence = "high"
    elif forecast_states and len(forecast_states) <= 4:
        confidence = "medium"
    else:
        confidence = "low"

    critical_note = " (CRITICAL: flowering window)" if in_critical else ""

    return IrrigationAdvice(
        date=today,
        irrigate=True,
        dose_mm=round(dose, 1),
        recommended_date=recommended_day,
        reason=(
            f"Depletion {dr:.0f}/{taw:.0f} mm approaching stress threshold. "
            f"Forecast rain: {effective_precip:.0f} mm (discounted). "
            f"Recommend {dose:.0f} mm on {recommended_day}{critical_note}"
        ),
        confidence=confidence,
        current_depletion=dr,
        current_taw=taw,
        current_raw=raw,
        current_ks=ks,
        crop_stage=stage,
        forecast_precip_mm=effective_precip,
        days_until_stress=days_to_stress,
    )
