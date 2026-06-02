"""Rule-based irrigation logic."""

import datetime as dt
import pandas as pd

from irrigator.decision import assess_confidence


class IrrigationAdvice:
    irrigate : bool
    reason : str
    dose_mm : float | int | None = None
    recommended_date : dt.datetime | None = None
    confidence : pd.DataFrame | None = None



def find_driest_window(forecast_states, window_days):
    """Find the driest window."""
    raise NotImplementedError("Not yet implemented !")



def irrigation_recommendation(state, forecast_states, params):
    """
    state: current WaterBalanceState
    forecast_states: list of WaterBalanceState for next N days
    params: crop and management parameters
    """
    # 1. Current depletion check
    if state.fraction_awc > (1 - params.p_threshold):
        return IrrigationAdvice(irrigate=False, reason="Soil water adequate")

    # 2. Will rain refill within forecast horizon?
    forecast_precip_total = sum(f.precip for f in forecast_states)
    projected_depletion = (
        state.depletion + sum(f.etc for f in forecast_states) - forecast_precip_total
    )

    if projected_depletion < params.p_threshold * params.total_awc:
        return IrrigationAdvice(
            irrigate=False,
            reason=f"Forecast rain ({forecast_precip_total:.0f} mm) should be sufficient",
        )

    # 3. Irrigation needed — compute amount
    # Target: refill to field capacity minus a safety margin
    # (don't fully refill if rain is possible)
    target_refill = state.depletion - forecast_precip_total * params.rain_confidence
    dose_mm = max(0, min(target_refill, params.max_dose_mm))

    # 4. When? Find the driest window in forecast
    best_day = find_driest_window(forecast_states, window_days=1)

    return IrrigationAdvice(
        irrigate=True,
        dose_mm=dose_mm,
        recommended_date=best_day,
        reason=f"Depletion will reach {projected_depletion:.0f} mm without irrigation",
        confidence=assess_confidence(forecast_states),
    )