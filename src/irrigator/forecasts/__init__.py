"""IrriGator Block 5 — Forecast integration.

Short-term:
    AROME deterministic (0-48h) + IFS ENS probabilistic (0-15 days)
    → Blended ensemble stress report

Seasonal:
    SEAS5 → PCA analog matching → daily scenarios → seasonal outlook
"""

from irrigator.forecasts.short_term import (
    DailyEnsembleStats,
    EnsembleStressReport,
    run_ensemble_forward_balance,
    run_forward_balance,
    will_stress_occur,
)
from irrigator.forecasts.arome_processing import load_arome_daily_cache

__all__ = [
    "DailyEnsembleStats",
    "EnsembleStressReport",
    "run_ensemble_forward_balance",
    "run_forward_balance",
    "will_stress_occur",
    "load_arome_daily_cache",
]
