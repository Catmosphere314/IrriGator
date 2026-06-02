"""IrriGator Block 6 — Irrigation decision layer.

Combines current water balance state, short-term forecasts, and seasonal
outlook to produce actionable irrigation recommendations.

Typical usage::

    from irrigator.decision.rules import compute_recommendation, DecisionParams
    from irrigator.decision.seasonal_outlook import compute_seasonal_outlook
    from irrigator.decision.output import build_parcel_report, save_report

    params = DecisionParams.from_config(parcel=parcel)
    advice = compute_recommendation(current_state, forecast_states, crop_params, params)
    outlook = compute_seasonal_outlook(scenarios, parcel, terrain, soil, crop_params)
    report = build_parcel_report(parcel.id, today, advice, outlook)
    save_report(report, Path("output/"))
"""

from irrigator.decision.rules import IrrigationAdvice, compute_recommendation

__all__ = ["IrrigationAdvice", "compute_recommendation"]
