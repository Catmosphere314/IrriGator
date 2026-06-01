"""FAO-56 soil water balance with coupled crop growth.

This is the core simulation loop.  It runs daily, advancing both the
crop state (GDD, Kc, Z_r) and the soil water balance simultaneously.

Temporal structure
------------------
For each day t from planting to harvest:

    1. Advance crop:  GDD(t) → stage(t) → Kc(t), Z_r(t)
    2. Compute TAW:   TAW(t) = soil.awc_for_root_depth(Z_r(t))
    3. Compute Ks:    water stress from yesterday's depletion
    4. Compute ETc:   ETc(t) = Ks(t) × Kc(t) × ET0(t)
    5. Water balance:  Dr(t) = Dr(t-1) - P(t) - I(t) + ETc(t) + R(t)
    6. Drainage:      if Dr(t) < 0, excess drains → Dr(t) = 0
    7. Store state

The key insight: Z_r grows over time, which changes TAW.  When roots
reach a new soil layer, the plant can access more water.  FAO-56
handles this by adjusting Dr when TAW changes (Section 8.5).

References
----------
- FAO-56 Chapter 8: ETc under soil water stress
- FAO-56 Equation 83: daily water balance
- FAO-56 Equation 84: stress coefficient Ks
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from irrigator.atmospheric.forcing import DailyForcing
from irrigator.config import ParcelConfig
from irrigator.crop.phenology import CropParams, CropState, advance_crop
from irrigator.static_layers.soil import SoilProfile
from irrigator.static_layers.terrain import TerrainParams
from irrigator.water_balance.et0 import compute_et0
from irrigator.water_balance.state import WaterBalanceState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single daily step
# ---------------------------------------------------------------------------


def _compute_ks(depletion: float, taw: float, raw: float) -> float:
    """Water stress coefficient Ks (FAO-56 eq. 84).

    Ks = 1.0                              when Dr ≤ RAW (no stress)
    Ks = (TAW - Dr) / (TAW - RAW)         when RAW < Dr < TAW
    Ks = 0.0                              when Dr ≥ TAW (wilting)

    Parameters
    ----------
    depletion : current root zone depletion Dr [mm]
    taw : total available water [mm]
    raw : readily available water = p × TAW [mm]
    """
    if taw <= 0:
        return 1.0
    if depletion <= raw:
        return 1.0
    if depletion >= taw:
        return 0.0
    return (taw - depletion) / (taw - raw)


def _simple_runoff(precip: float, depletion: float, taw: float) -> float:
    """Simplified runoff: fraction of rainfall that runs off.

    When soil is near field capacity (low depletion), more rain runs off.
    This is a crude approximation — replace with SCS curve number if needed.
    """
    if precip <= 0:
        return 0.0
    # Fraction of TAW that's already filled
    if taw <= 0:
        return 0.0
    saturation = 1.0 - (depletion / taw)
    # Simple linear: 0% runoff when empty, 20% when at FC
    runoff_frac = max(0.0, min(0.3, 0.3 * saturation))
    return precip * runoff_frac


def daily_step(
    current_date: date,
    et0: float,
    precip: float,
    irrigation: float,
    crop: CropState,
    soil: SoilProfile,
    dr_prev: float,
    taw_prev: float,
    p: float,
) -> WaterBalanceState:
    """Advance the water balance by one day.

    This is the core FAO-56 daily computation (Chapter 8).

    Parameters
    ----------
    current_date : today
    et0 : reference evapotranspiration [mm]
    precip : precipitation [mm]
    irrigation : irrigation applied [mm]
    crop : today's CropState (from advance_crop)
    soil : SoilProfile for the parcel
    dr_prev : yesterday's depletion [mm]
    taw_prev : yesterday's TAW [mm]
    p : depletion fraction (0.55 for maize)

    Returns
    -------
    WaterBalanceState for today.
    """
    # --- TAW from current root depth ---
    # TAW = available water capacity integrated over root zone
    z_r = crop.z_root
    if z_r <= 0:
        # No roots yet — use a minimal soil layer
        z_r = 0.10
    taw = soil.awc_for_root_depth(z_r)
    taw = max(taw, 1.0)  # avoid division by zero

    # Adjust depletion when TAW increases (roots growing into new soil)
    # FAO-56 Section 8.5: new soil layer assumed at field capacity
    # So when TAW increases, depletion stays the same (new water available)
    # When TAW decreases (shouldn't happen), cap depletion at new TAW
    dr = min(dr_prev, taw)

    # --- Readily available water ---
    # Adjust p for high ET0 days (FAO-56 eq. 86)
    p_adj = p + 0.04 * (5.0 - et0)
    p_adj = max(0.1, min(0.8, p_adj))
    raw = p_adj * taw

    # --- Stress coefficient ---
    ks = _compute_ks(dr, taw, raw)

    # --- Crop ET ---
    kc = crop.kc
    etc_pot = kc * et0  # potential crop ET (no stress)
    etc_act = ks * kc * et0  # actual crop ET (with stress)

    # --- Runoff ---
    runoff = _simple_runoff(precip, dr, taw)
    effective_precip = precip - runoff

    # --- Water balance (FAO-56 eq. 83) ---
    # Dr(t) = Dr(t-1) - (P - RO) - I + ETc + DP
    # Rearranging to solve for Dr(t) before drainage:
    dr_new = dr - effective_precip - irrigation + etc_act

    # --- Drainage ---
    # If Dr < 0, soil exceeded field capacity → excess drains
    drainage = 0.0
    if dr_new < 0:
        drainage = -dr_new  # water that drains away
        dr_new = 0.0

    # Cap at TAW (can't deplete beyond wilting point)
    dr_new = min(dr_new, taw)

    return WaterBalanceState(
        date=current_date,
        et0=et0,
        kc=kc,
        etc_pot=etc_pot,
        etc_act=etc_act,
        precip=precip,
        irrigation=irrigation,
        runoff=runoff,
        drainage=drainage,
        depletion=dr_new,
        taw=taw,
        raw=raw,
        stress_coeff=ks,
        z_root=z_r,
        gdd=crop.gdd,
        crop_stage=crop.stage,
    )


# ---------------------------------------------------------------------------
# Full simulation loop
# ---------------------------------------------------------------------------


def run_simulation(
    forcing: DailyForcing,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil: SoilProfile,
    crop_params: CropParams,
    irrigation_schedule: dict[date, float] | None = None,
    initial_depletion_frac: float = 0.0,
) -> list[WaterBalanceState]:
    """Run the coupled crop growth + water balance simulation.

    This is the main entry point.  It loops through each day:
    1. Compute ET0 from forcing
    2. Advance crop state (GDD → stage → Kc, Z_r)
    3. Step the water balance
    4. Record the state

    Parameters
    ----------
    forcing : DailyForcing from Block 2
    parcel : ParcelConfig
    terrain : TerrainParams
    soil : SoilProfile from Block 1
    crop_params : CropParams (GDD thresholds, Kc values, etc.)
    irrigation_schedule : dict mapping dates to irrigation amounts [mm]
                         (None or {} for rainfed simulation)
    initial_depletion_frac : starting depletion as fraction of TAW
                             0.0 = at field capacity, 1.0 = at wilting point

    Returns
    -------
    List of WaterBalanceState, one per day.
    """
    if irrigation_schedule is None:
        irrigation_schedule = {}

    n = forcing.n_days
    logger.info(
        "Starting simulation: %d days, parcel %s, planting %s",
        n,
        parcel.id,
        crop_params.planting_date,
    )

    # --- Compute ET0 for the full period ---
    et0_series = compute_et0(forcing, terrain, parcel.lat)
    # Handle xarray/pandas output → numpy
    if hasattr(et0_series, "values"):
        et0_series = et0_series.values
    et0_series = np.asarray(et0_series, dtype=np.float64)

    # --- Initial conditions ---
    # Start with a minimal root zone and scale depletion
    z_r_init = max(crop_params.z_root_initial, 0.10)
    taw_init = soil.awc_for_root_depth(z_r_init)
    dr = initial_depletion_frac * taw_init
    taw_prev = taw_init
    gdd_prev = 0.0
    print(taw_prev)

    states: list[WaterBalanceState] = []

    # --- Daily loop ---
    for i in range(n):
        current_date = pd.Timestamp(forcing.dates[i]).date()
        t_min = float(forcing.t_min[i])
        t_max = float(forcing.t_max[i])
        et0 = float(et0_series[i])
        precip = float(forcing.precip_mm[i])
        irrigation = irrigation_schedule.get(current_date, 0.0)

        # Step 1: advance crop
        print(gdd_prev)
        crop = advance_crop(current_date, t_min, t_max, gdd_prev, crop_params)

        # Step 2: water balance
        state = daily_step(
            current_date=current_date,
            et0=et0,
            precip=precip,
            irrigation=irrigation,
            crop=crop,
            soil=soil,
            dr_prev=dr,
            taw_prev=taw_prev,
            p=crop_params.p,
        )

        # Update for next day
        dr = state.depletion
        taw_prev = state.taw
        gdd_prev = crop.gdd

        states.append(state)

    # Log summary
    if states:
        total_precip = sum(s.precip for s in states)
        total_etc = sum(s.etc_act for s in states)
        total_irrig = sum(s.irrigation for s in states)
        total_drainage = sum(s.drainage for s in states)
        stress_days = sum(1 for s in states if s.is_stressed)
        max_gdd = states[-1].gdd

        logger.info(
            "Simulation complete: %d days, GDD=%.0f, "
            "precip=%.0f mm, ETc=%.0f mm, irrigation=%.0f mm, "
            "drainage=%.0f mm, stress days=%d",
            n,
            max_gdd,
            total_precip,
            total_etc,
            total_irrig,
            total_drainage,
            stress_days,
        )

    return states


def states_to_dataframe(states: list[WaterBalanceState]) -> pd.DataFrame:
    """Convert simulation output to a DataFrame for analysis."""
    records = []
    for s in states:
        records.append(
            {
                "date": s.date,
                "et0": s.et0,
                "kc": s.kc,
                "etc_pot": s.etc_pot,
                "etc_act": s.etc_act,
                "precip": s.precip,
                "irrigation": s.irrigation,
                "runoff": s.runoff,
                "drainage": s.drainage,
                "depletion": s.depletion,
                "taw": s.taw,
                "raw": s.raw,
                "stress_coeff": s.stress_coeff,
                "fraction_available": s.fraction_available,
                "z_root": s.z_root,
                "gdd": s.gdd,
                "crop_stage": s.crop_stage,
            }
        )
    return pd.DataFrame(records).set_index("date")
