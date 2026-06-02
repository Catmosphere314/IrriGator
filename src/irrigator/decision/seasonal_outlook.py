"""Seasonal irrigation outlook from SEAS5 ensemble scenarios.

Runs the water balance under each of the 51 SEAS5 ensemble scenarios
(from analog disaggregation) to produce a probabilistic estimate of
total irrigation demand for the rest of the season.

This is the strategic planning component — not "irrigate tomorrow"
but "expect 120-200 mm of irrigation this season."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from irrigator.config import ParcelConfig
from irrigator.crop.phenology import CropParams
from irrigator.forecasts.analog_disaggregation import SeasonalScenario
from irrigator.static_layers.soil import SoilProfile
from irrigator.static_layers.terrain import TerrainParams
from irrigator.water_balance.bucket_model import run_simulation

logger = logging.getLogger(__name__)


@dataclass
class SeasonalOutlook:
    """Probabilistic seasonal irrigation demand."""

    n_members: int
    total_irrigation_p25: float  # mm
    total_irrigation_median: float
    total_irrigation_p75: float
    total_irrigation_mean: float
    stress_days_median: float
    stress_days_p75: float
    max_single_deficit: float  # worst single-day depletion across members
    season_description: str

    def to_dict(self) -> dict:
        return {
            "n_ensemble_members": self.n_members,
            "total_irrigation_mm": {
                "p25": round(self.total_irrigation_p25),
                "median": round(self.total_irrigation_median),
                "p75": round(self.total_irrigation_p75),
                "mean": round(self.total_irrigation_mean),
            },
            "stress_days": {
                "median": round(self.stress_days_median),
                "p75": round(self.stress_days_p75),
            },
            "description": self.season_description,
        }


def compute_seasonal_outlook(
    scenarios: list[SeasonalScenario],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil: SoilProfile,
    crop_params: CropParams,
    initial_depletion_frac: float = 0.0,
    irrigation_trigger_mm: float | None = None,
) -> SeasonalOutlook:
    """Run the water balance under each SEAS5 scenario.

    Each scenario contains a gridded xr.Dataset at ERA5-Land resolution.
    We first downscale to parcel level via ``extract_parcel_forcing()``
    (same pipeline as Block 2 — no duplication), then run the simulation.

    Parameters
    ----------
    scenarios : list of SeasonalScenario from analog disaggregation
    parcel, terrain, soil, crop_params : the usual
    initial_depletion_frac : starting soil state
    irrigation_trigger_mm : override trigger (default: use RAW from soil/crop)

    Returns
    -------
    SeasonalOutlook with ensemble statistics.
    """
    from irrigator.atmospheric.forcing import extract_parcel_forcing

    total_irrigations = []
    stress_day_counts = []
    max_depletions = []

    for scenario in scenarios:
        # Downscale gridded scenario to parcel level
        # Uses the SAME pipeline as ERA5-Land historical (Block 2)
        forcing = extract_parcel_forcing(
            era5_daily=scenario.daily_ds,
            parcel=parcel,
            terrain=terrain,
            precip_correction=None,  # SEAS5 is already bias-corrected
        )

        # Run simulation
        states = run_simulation(
            forcing=forcing,
            parcel=parcel,
            terrain=terrain,
            soil=soil,
            crop_params=crop_params,
            irrigation_schedule={},  # rainfed first
            initial_depletion_frac=initial_depletion_frac,
        )

        # Count stress days and compute irrigation that would have been needed
        stress_days = sum(1 for s in states if s.stress_coeff < 0.9)
        max_depletion = max(s.depletion for s in states) if states else 0

        # Estimate needed irrigation: sum of depletion exceedance above RAW
        total_needed = 0.0
        for s in states:
            if s.depletion > s.raw:
                total_needed += s.depletion - s.raw * 0.5  # refill halfway

        total_irrigations.append(total_needed)
        stress_day_counts.append(stress_days)
        max_depletions.append(max_depletion)

    # Ensemble statistics
    irrig_arr = np.array(total_irrigations)
    stress_arr = np.array(stress_day_counts)

    p25 = float(np.percentile(irrig_arr, 25))
    median = float(np.percentile(irrig_arr, 50))
    p75 = float(np.percentile(irrig_arr, 75))
    mean = float(irrig_arr.mean())

    # Season description
    if median < 50:
        desc = "Wet season expected — minimal irrigation likely needed"
    elif median < 120:
        desc = "Near-normal season — moderate irrigation expected"
    elif median < 200:
        desc = "Drier than average — plan for significant irrigation"
    else:
        desc = "Dry season expected — heavy irrigation demand"

    outlook = SeasonalOutlook(
        n_members=len(scenarios),
        total_irrigation_p25=p25,
        total_irrigation_median=median,
        total_irrigation_p75=p75,
        total_irrigation_mean=mean,
        stress_days_median=float(np.median(stress_arr)),
        stress_days_p75=float(np.percentile(stress_arr, 75)),
        max_single_deficit=float(max(max_depletions)) if max_depletions else 0,
        season_description=desc,
    )

    logger.info(
        "Seasonal outlook (%d members): irrigation %d–%d mm (p25–p75), stress days median=%d, %s",
        len(scenarios),
        int(p25),
        int(p75),
        int(np.median(stress_arr)),
        desc,
    )

    return outlook
