"""IrriGator Block 5 — Forecast integration.

Short-term (AROME/ARPEGE, 2-4 days):
    Forward water balance to detect imminent stress.

Seasonal (SEAS5, 1-6 months):
    Analog-based disaggregation of ensemble members into daily scenarios.
    Probabilistic irrigation demand over the season.
"""

from irrigator.forecasts.short_term import run_forward_balance, will_stress_occur

__all__ = ["run_forward_balance", "will_stress_occur"]
