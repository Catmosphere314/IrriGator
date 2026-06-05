"""Irrigation decision logic — ensemble-aware.

Accepts either:
- Legacy: list[WaterBalanceState] from single-path forward balance
- Ensemble: EnsembleStressReport from blended AROME + IFS ENS

When using ensemble input, the decision uses probability of stress
rather than deterministic yes/no:
- >70% members show stress → irrigate (high confidence)
- 40-70% → irrigate (medium confidence)
- <40% → don't irrigate, but flag uncertainty
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import yaml

from irrigator.config import ParcelConfig
from irrigator.crop.phenology import CropParams
from irrigator.forecasts.short_term import EnsembleStressReport
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
    confidence: str
    # Context
    current_depletion: float
    current_taw: float
    current_raw: float
    current_ks: float
    crop_stage: str
    forecast_precip_mm: float
    days_until_stress: int | None
    # Ensemble info (if available)
    stress_probability: float | None = None
    ensemble_ks_median: float | None = None

    def to_dict(self) -> dict:
        d = {
            "date": self.date.isoformat(),
            "irrigate": self.irrigate,
            "dose_mm": round(self.dose_mm, 1),
            "recommended_date": self.recommended_date.isoformat() if self.recommended_date else None,
            "reason": self.reason,
            "confidence": self.confidence,
            "current_depletion_mm": round(self.current_depletion, 1),
            "current_taw_mm": round(self.current_taw, 1),
            "stress_coefficient": round(self.current_ks, 3),
            "crop_stage": self.crop_stage,
            "forecast_precip_mm": round(self.forecast_precip_mm, 1),
            "days_until_stress": self.days_until_stress,
        }
        if self.stress_probability is not None:
            d["stress_probability"] = round(self.stress_probability, 2)
        if self.ensemble_ks_median is not None:
            d["ensemble_ks_median_day5"] = round(self.ensemble_ks_median, 3)
        return d


@dataclass
class DecisionParams:
    """Parameters controlling the irrigation decision."""
    rain_confidence_0_24h: float = 0.85
    rain_confidence_24_48h: float = 0.70
    rain_confidence_48_96h: float = 0.50
    refill_no_rain: float = 0.95
    refill_rain_possible: float = 0.80
    refill_rain_likely: float = 0.65
    rain_possible_mm: float = 5.0
    rain_likely_mm: float = 15.0
    critical_trigger_reduction: float = 0.10
    critical_refill_increase: float = 0.05
    max_dose_mm: float = 40.0
    min_dose_mm: float = 15.0
    min_interval_days: int = 3
    # Ensemble thresholds
    stress_prob_irrigate: float = 0.70   # irrigate if >70% members stress
    stress_prob_watch: float = 0.40      # flag if >40%

    @classmethod
    def from_config(
        cls,
        model_params_path: str = "configs/model_parameters.yaml",
        parcel: ParcelConfig | None = None,
    ) -> DecisionParams:
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


def _is_critical_window(crop_stage: str, crop_params: CropParams, gdd: float) -> bool:
    if crop_stage != "mid":
        return False
    return crop_params.gdd_end_development <= gdd <= crop_params.gdd_end_development + 300


# ---------------------------------------------------------------------------
# Ensemble-aware recommendation
# ---------------------------------------------------------------------------

def compute_recommendation(
    current_state: WaterBalanceState,
    stress_report: EnsembleStressReport | list[WaterBalanceState],
    crop_params: CropParams,
    params: DecisionParams,
    last_irrigation_date: date | None = None,
) -> IrrigationAdvice:
    """Compute irrigation recommendation from current state + forecast.

    Accepts either an EnsembleStressReport (from IFS ENS) or a legacy
    list of WaterBalanceState (from single-path forward balance).
    """
    today = current_state.date
    dr = current_state.depletion
    taw = current_state.taw
    raw = current_state.raw
    ks = current_state.stress_coeff
    stage = current_state.crop_stage

    in_critical = _is_critical_window(stage, crop_params, current_state.gdd)

    if in_critical:
        trigger_p = crop_params.p - params.critical_trigger_reduction
    else:
        trigger_p = crop_params.p
    trigger_dr = trigger_p * taw

    # --- Extract forecast info depending on input type ---
    if isinstance(stress_report, EnsembleStressReport):
        return _recommend_from_ensemble(
            current_state, stress_report, crop_params, params,
            last_irrigation_date, in_critical, trigger_dr,
        )
    else:
        return _recommend_from_single_path(
            current_state, stress_report, crop_params, params,
            last_irrigation_date, in_critical, trigger_dr,
        )


def _recommend_from_ensemble(
    state: WaterBalanceState,
    report: EnsembleStressReport,
    crop_params: CropParams,
    params: DecisionParams,
    last_irrigation_date: date | None,
    in_critical: bool,
    trigger_dr: float,
) -> IrrigationAdvice:
    """Decision logic using ensemble stress probabilities."""
    today = state.date
    dr, taw, raw, ks, stage = state.depletion, state.taw, state.raw, state.stress_coeff, state.crop_stage

    # Ensemble forecast precipitation (mean across members, discounted)
    ens_precip = 0.0
    for d in report.daily_stats:
        if d.day_offset < 2:
            ens_precip += d.precip_mean * params.rain_confidence_0_24h
        elif d.day_offset < 4:
            ens_precip += d.precip_mean * params.rain_confidence_24_48h
        else:
            ens_precip += d.precip_mean * params.rain_confidence_48_96h

    # Peak stress probability in forecast horizon
    max_stress_prob = max((d.stress_probability for d in report.daily_stats), default=0)
    worst = report.worst_day

    # Median Ks at day 5 (if available)
    ens_ks_day5 = None
    for d in report.daily_stats:
        if d.day_offset == 5:
            ens_ks_day5 = d.ks_median
            break

    # --- Decision tree ---

    # Not growing
    if stage in ("not_planted", "pre_emergence", "mature"):
        return IrrigationAdvice(
            date=today, irrigate=False, dose_mm=0, recommended_date=None,
            reason=f"Crop stage '{stage}' — no irrigation needed",
            confidence="high", current_depletion=dr, current_taw=taw,
            current_raw=raw, current_ks=ks, crop_stage=stage,
            forecast_precip_mm=ens_precip, days_until_stress=None,
            stress_probability=max_stress_prob, ensemble_ks_median=ens_ks_day5,
        )

    # Minimum interval
    if last_irrigation_date and (today - last_irrigation_date).days < params.min_interval_days:
        return IrrigationAdvice(
            date=today, irrigate=False, dose_mm=0, recommended_date=None,
            reason=f"Irrigated {(today - last_irrigation_date).days}d ago",
            confidence="high", current_depletion=dr, current_taw=taw,
            current_raw=raw, current_ks=ks, crop_stage=stage,
            forecast_precip_mm=ens_precip, days_until_stress=None,
            stress_probability=max_stress_prob, ensemble_ks_median=ens_ks_day5,
        )

    # Soil currently well above trigger
    if dr < trigger_dr * 0.7:
        return IrrigationAdvice(
            date=today, irrigate=False, dose_mm=0, recommended_date=None,
            reason=f"Soil water adequate (depletion {dr:.0f}/{taw:.0f} mm). "
                   f"Stress probability: {max_stress_prob:.0%} over forecast.",
            confidence="high", current_depletion=dr, current_taw=taw,
            current_raw=raw, current_ks=ks, crop_stage=stage,
            forecast_precip_mm=ens_precip, days_until_stress=report.first_stress_day,
            stress_probability=max_stress_prob, ensemble_ks_median=ens_ks_day5,
        )

    # Ensemble says stress is unlikely (<40% probability)
    if max_stress_prob < params.stress_prob_watch:
        return IrrigationAdvice(
            date=today, irrigate=False, dose_mm=0, recommended_date=None,
            reason=f"Low stress probability ({max_stress_prob:.0%}). "
                   f"Ensemble expects sufficient rain ({ens_precip:.0f} mm discounted).",
            confidence="medium", current_depletion=dr, current_taw=taw,
            current_raw=raw, current_ks=ks, crop_stage=stage,
            forecast_precip_mm=ens_precip, days_until_stress=report.first_stress_day,
            stress_probability=max_stress_prob, ensemble_ks_median=ens_ks_day5,
        )

    # --- Irrigation needed ---
    # Determine confidence from stress probability
    if max_stress_prob > params.stress_prob_irrigate:
        confidence = "high"
        prob_note = f"{max_stress_prob:.0%} of ensemble members show stress"
    elif max_stress_prob > params.stress_prob_watch:
        confidence = "medium"
        prob_note = f"{max_stress_prob:.0%} of members show stress — uncertain"
    else:
        confidence = "low"
        prob_note = f"Marginal: {max_stress_prob:.0%} stress probability"

    # Compute dose
    refill_bonus = params.critical_refill_increase if in_critical else 0
    if ens_precip >= params.rain_likely_mm:
        refill_frac = params.refill_rain_likely + refill_bonus
    elif ens_precip >= params.rain_possible_mm:
        refill_frac = params.refill_rain_possible + refill_bonus
    else:
        refill_frac = params.refill_no_rain + refill_bonus

    target_dr = (1.0 - min(1.0, refill_frac)) * taw
    dose = max(0, dr - target_dr - ens_precip)
    dose = min(dose, params.max_dose_mm)

    if dose < params.min_dose_mm:
        return IrrigationAdvice(
            date=today, irrigate=False, dose_mm=0, recommended_date=None,
            reason=f"Computed dose {dose:.0f} mm below minimum. {prob_note}.",
            confidence=confidence, current_depletion=dr, current_taw=taw,
            current_raw=raw, current_ks=ks, crop_stage=stage,
            forecast_precip_mm=ens_precip, days_until_stress=report.first_stress_day,
            stress_probability=max_stress_prob, ensemble_ks_median=ens_ks_day5,
        )

    # Find driest day from ensemble mean precip
    driest_day = min(report.daily_stats[:4], key=lambda d: d.precip_mean) if report.daily_stats else None
    rec_date = driest_day.date.date() if driest_day and hasattr(driest_day.date, "date") else today

    critical_note = " (CRITICAL: flowering window)" if in_critical else ""

    return IrrigationAdvice(
        date=today, irrigate=True, dose_mm=round(dose, 1),
        recommended_date=rec_date,
        reason=f"{prob_note}. Depletion {dr:.0f}/{taw:.0f} mm. "
               f"Forecast rain: {ens_precip:.0f} mm. "
               f"Recommend {dose:.0f} mm on {rec_date}{critical_note}",
        confidence=confidence, current_depletion=dr, current_taw=taw,
        current_raw=raw, current_ks=ks, crop_stage=stage,
        forecast_precip_mm=ens_precip, days_until_stress=report.first_stress_day,
        stress_probability=max_stress_prob, ensemble_ks_median=ens_ks_day5,
    )


def _recommend_from_single_path(
    state: WaterBalanceState,
    forecast_states: list[WaterBalanceState],
    crop_params: CropParams,
    params: DecisionParams,
    last_irrigation_date: date | None,
    in_critical: bool,
    trigger_dr: float,
) -> IrrigationAdvice:
    """Legacy single-path decision (backward compatible)."""
    today = state.date
    dr, taw, raw, ks, stage = state.depletion, state.taw, state.raw, state.stress_coeff, state.crop_stage

    eff_precip = sum(
        s.precip * (params.rain_confidence_0_24h if i < 1
                    else params.rain_confidence_24_48h if i < 2
                    else params.rain_confidence_48_96h)
        for i, s in enumerate(forecast_states)
    )

    if stage in ("not_planted", "pre_emergence", "mature"):
        return IrrigationAdvice(
            date=today, irrigate=False, dose_mm=0, recommended_date=None,
            reason=f"Crop stage '{stage}'", confidence="high",
            current_depletion=dr, current_taw=taw, current_raw=raw,
            current_ks=ks, crop_stage=stage, forecast_precip_mm=eff_precip,
            days_until_stress=None,
        )

    if last_irrigation_date and (today - last_irrigation_date).days < params.min_interval_days:
        return IrrigationAdvice(
            date=today, irrigate=False, dose_mm=0, recommended_date=None,
            reason=f"Min interval not elapsed", confidence="high",
            current_depletion=dr, current_taw=taw, current_raw=raw,
            current_ks=ks, crop_stage=stage, forecast_precip_mm=eff_precip,
            days_until_stress=None,
        )

    stress_in_forecast = any(s.stress_coeff < 0.9 for s in forecast_states)
    if not stress_in_forecast and dr < trigger_dr:
        return IrrigationAdvice(
            date=today, irrigate=False, dose_mm=0, recommended_date=None,
            reason=f"No stress expected in forecast",
            confidence="medium", current_depletion=dr, current_taw=taw,
            current_raw=raw, current_ks=ks, crop_stage=stage,
            forecast_precip_mm=eff_precip, days_until_stress=None,
        )

    refill_bonus = params.critical_refill_increase if in_critical else 0
    if eff_precip >= params.rain_likely_mm:
        refill_frac = params.refill_rain_likely + refill_bonus
    elif eff_precip >= params.rain_possible_mm:
        refill_frac = params.refill_rain_possible + refill_bonus
    else:
        refill_frac = params.refill_no_rain + refill_bonus

    target_dr = (1.0 - min(1.0, refill_frac)) * taw
    dose = max(0, min(dr - target_dr - eff_precip, params.max_dose_mm))

    if dose < params.min_dose_mm:
        return IrrigationAdvice(
            date=today, irrigate=False, dose_mm=0, recommended_date=None,
            reason=f"Dose {dose:.0f} mm below minimum",
            confidence="medium", current_depletion=dr, current_taw=taw,
            current_raw=raw, current_ks=ks, crop_stage=stage,
            forecast_precip_mm=eff_precip, days_until_stress=None,
        )

    driest = min(forecast_states[:4], key=lambda s: s.precip) if forecast_states else None
    rec_date = driest.date if driest else today

    return IrrigationAdvice(
        date=today, irrigate=True, dose_mm=round(dose, 1),
        recommended_date=rec_date,
        reason=f"Stress expected. Recommend {dose:.0f} mm on {rec_date}",
        confidence="high" if len(forecast_states) <= 2 else "medium",
        current_depletion=dr, current_taw=taw, current_raw=raw,
        current_ks=ks, crop_stage=stage, forecast_precip_mm=eff_precip,
        days_until_stress=None,
    )
