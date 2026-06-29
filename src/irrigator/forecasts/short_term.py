"""Short-term forecast integration with ensemble probabilistic stress.

Architecture:
- AROME (-5days-0h, deterministic): single forward water balance path
- AROME (0-48h, deterministic): single forward water balance path
- IFS ENS (0-15d, 51 members): ensemble forward water balance
- Output: per-day statistics (mean/median/p25/p75/min/max Ks, depletion, precip)

The caller blends AROME (days 0-2) with IFS ENS (days 3-15):
- Days 0-2: AROME deterministic Ks (high confidence)
- Days 3-15: IFS ENS distribution (mean, quantiles, spread)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from datetime import date, timedelta
from pathlib import Path


from irrigator.atmospheric.forcing import DailyForcing
from irrigator.config import ParcelConfig
from irrigator.crop.phenology import CropParams, advance_crop
from irrigator.static_layers.soil import SoilProfile
from irrigator.static_layers.terrain import TerrainParams
from irrigator.water_balance.bucket_model import daily_step
from irrigator.water_balance.et0 import compute_et0
from irrigator.water_balance.state import WaterBalanceState

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Forecast standardization (unchanged)
# ---------------------------------------------------------------------------


def standardize_forecast_to_era5_format(
    forecast_ds: xr.Dataset,
    source: str = "arpege",
    shift_utc: int = 0,
) -> xr.Dataset:
    """Rename/convert a forecast dataset to match ERA5-Land daily format.

    After this, the dataset can be passed directly to
    ``extract_parcel_forcing()`` from Block 2.

    Parameters
    ----------
    forecast_ds : raw forecast dataset (from meteofrance_client)
    source : "arome" or "arpege"

    Returns
    -------
    xr.Dataset with variable names and units matching ERA5-Land daily.
    """
    # Variable name mapping — adjust based on actual Météo-France output
    # These are common NWP output names; the exact names depend on the
    # API version and request format.
    rename_candidates = {
        "2t": "t_mean",
        "t2m": "t_mean",
        "mn2t": "t_min",
        "mx2t": "t_max",
        "10u": "u10",
        "10v": "v10",
        "ssrd": "rs_mj",
        "ssr": "rs_mj",
        "sp": "pressure_kpa",
        "tp": "precip_mm",
        "2d": "dewpoint",
        "d2m": "dewpoint",
    }

    result = forecast_ds.copy()
    for old, new in rename_candidates.items():
        if old in result.data_vars and new not in result.data_vars:
            result = result.rename({old: new})

    for tvar in ("t_mean", "t_min", "t_max", "dewpoint"):
        if tvar in result and float(result[tvar].mean()) > 100:
            result[tvar] = result[tvar] - 273.15

    if "u10" in result and "v10" in result and "wind_speed_10m" not in result:
        result["wind_speed_10m"] = np.sqrt(result["u10"] ** 2 + result["v10"] ** 2)

    if "pressure_kpa" in result and float(result["pressure_kpa"].mean()) > 10000:
        result["pressure_kpa"] = result["pressure_kpa"] / 1000.0

    for old_time in ("time", "step", "forecast_time"):
        if old_time in result.dims and "valid_time" not in result.dims:
            result = result.rename({old_time: "valid_time"})

    result = result.assign_coords(valid_time=result.valid_time + pd.Timedelta(f"{shift_utc}h"))

    return result


# ---------------------------------------------------------------------------
# Single-path forward balance (AROME deterministic)
# ---------------------------------------------------------------------------


def run_forward_balance(
    current_state: WaterBalanceState,
    forecast_forcing: DailyForcing,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil: SoilProfile,
    crop_params: CropParams,
) -> list[WaterBalanceState]:
    """Single deterministic forward balance (for AROME)."""
    et0_series = compute_et0(forecast_forcing, terrain, parcel.lat)
    if hasattr(et0_series, "values"):
        et0_series = et0_series.values
    et0_series = np.asarray(et0_series, dtype=np.float64)

    dr = current_state.depletion
    gdd = current_state.gdd
    taw_prev = current_state.taw
    states = []

    for i in range(forecast_forcing.n_days):
        current_date = pd.Timestamp(forecast_forcing.dates[i]).date()
        crop = advance_crop(
            current_date,
            float(forecast_forcing.t_min[i]),
            float(forecast_forcing.t_max[i]),
            gdd,
            crop_params,
        )
        state = daily_step(
            current_date=current_date,
            et0=float(et0_series[i]),
            precip=float(forecast_forcing.precip_mm[i]),
            irrigation=0.0,
            crop=crop,
            soil=soil,
            dr_prev=dr,
            taw_prev=taw_prev,
            p=crop_params.p,
        )
        dr, taw_prev, gdd = state.depletion, state.taw, crop.gdd
        states.append(state)

    return states


# ---------------------------------------------------------------------------
# Ensemble stress report
# ---------------------------------------------------------------------------


@dataclass
class DailyEnsembleStats:
    """Ensemble statistics for one forecast day."""

    date: pd.Timestamp
    day_offset: int  # 0 = today
    source: str  # "arome" or "ifs_ens"
    # Stress coefficient Ks
    ks_mean: float
    ks_median: float
    ks_p25: float
    ks_p75: float
    ks_min: float
    ks_max: float
    # Depletion
    depletion_mean: float
    depletion_p25: float
    depletion_p75: float
    # Precipitation
    precip_mean: float
    precip_p25: float
    precip_p75: float
    # ET
    etc_mean: float
    # Number of members showing stress
    n_members_stressed: int
    n_members_total: int

    @property
    def stress_probability(self) -> float:
        """Fraction of ensemble members showing stress (Ks < 0.9)."""
        if self.n_members_total == 0:
            return 0.0
        return self.n_members_stressed / self.n_members_total

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat() if hasattr(self.date, "isoformat") else str(self.date),
            "day_offset": self.day_offset,
            "source": self.source,
            "ks": {
                "mean": round(self.ks_mean, 3),
                "median": round(self.ks_median, 3),
                "p25": round(self.ks_p25, 3),
                "p75": round(self.ks_p75, 3),
                "min": round(self.ks_min, 3),
                "max": round(self.ks_max, 3),
            },
            "depletion_mm": {
                "mean": round(self.depletion_mean, 1),
                "p25": round(self.depletion_p25, 1),
                "p75": round(self.depletion_p75, 1),
            },
            "precip_mm": {
                "mean": round(self.precip_mean, 1),
                "p25": round(self.precip_p25, 1),
                "p75": round(self.precip_p75, 1),
            },
            "etc_mean_mm": round(self.etc_mean, 1),
            "stress_probability": round(self.stress_probability, 2),
        }


@dataclass
class EnsembleStressReport:
    """Full probabilistic stress report across the forecast horizon."""

    daily_stats: list[DailyEnsembleStats]
    n_members: int
    arome_days: int  # days covered by AROME (deterministic)
    ens_days: int  # days covered by IFS ENS

    @property
    def stress_expected(self) -> bool:
        """Any day with >50% of members showing stress."""
        return any(d.stress_probability > 0.5 for d in self.daily_stats)

    @property
    def first_stress_day(self) -> int | None:
        """Day offset when stress probability first exceeds 50%."""
        for d in self.daily_stats:
            if d.stress_probability > 0.5:
                return d.day_offset
        return None

    @property
    def worst_day(self) -> DailyEnsembleStats | None:
        """Day with lowest median Ks."""
        if not self.daily_stats:
            return None
        return min(self.daily_stats, key=lambda d: d.ks_median)

    def to_dict(self) -> dict:
        return {
            "n_members": self.n_members,
            "arome_days": self.arome_days,
            "ens_days": self.ens_days,
            "stress_expected": self.stress_expected,
            "first_stress_day": self.first_stress_day,
            "worst_day": self.worst_day.to_dict() if self.worst_day else None,
            "daily": [d.to_dict() for d in self.daily_stats],
        }


# ---------------------------------------------------------------------------
# Ensemble forward balance (IFS ENS)
# ---------------------------------------------------------------------------


def run_ensemble_forward_balance(
    current_state: WaterBalanceState,
    member_forcings: dict[int, DailyForcing],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil: SoilProfile,
    crop_params: CropParams,
    arome_states: list[WaterBalanceState] | None = None,
    arome_days: int = 2,
    stress_threshold: float = 0.9,
) -> EnsembleStressReport:
    """Run forward water balance for each IFS ENS member.

    Produces per-day ensemble statistics: mean/median/p25/p75/min/max Ks,
    depletion, precipitation, and stress probability.

    Parameters
    ----------
    current_state : today's WaterBalanceState
    member_forcings : {member_id: DailyForcing} from IFS ENS processing
    parcel, terrain, soil, crop_params : the usual
    arome_states : deterministic AROME forward balance (days 0-2)
    arome_days : how many days AROME covers (default 2)
    stress_threshold : Ks below this counts as "stressed"

    Returns
    -------
    EnsembleStressReport with blended AROME + IFS ENS statistics.
    """
    # Run forward balance for each member
    all_member_states: dict[int, list[WaterBalanceState]] = {}

    # IFS ENS members branch from the AROME endpoint so there's no
    # discontinuity at the handoff.  They skip the first arome_days
    # of their own forcing (AROME covers those days deterministically).
    if arome_states and len(arome_states) >= arome_days:
        ens_start = arome_states[arome_days - 1]
        ens_skip = arome_days
    else:
        ens_start = current_state
        ens_skip = 0

    for member_id, forcing in member_forcings.items():
        et0 = compute_et0(forcing, terrain, parcel.lat)
        if hasattr(et0, "values"):
            et0 = et0.values
        et0 = np.asarray(et0, dtype=np.float64)

        dr = ens_start.depletion
        gdd = ens_start.gdd
        taw_prev = ens_start.taw
        member_states = []

        for i in range(ens_skip,forcing.n_days):
            current_date = pd.Timestamp(forcing.dates[i]).date()
            crop = advance_crop(
                current_date,
                float(forcing.t_min[i]),
                float(forcing.t_max[i]),
                gdd,
                crop_params,
            )
            state = daily_step(
                current_date=current_date,
                et0=float(et0[i]),
                precip=float(forcing.precip_mm[i]),
                irrigation=0.0,
                crop=crop,
                soil=soil,
                dr_prev=dr,
                taw_prev=taw_prev,
                p=crop_params.p,
            )
            dr, taw_prev, gdd = state.depletion, state.taw, crop.gdd
            member_states.append(state)

        all_member_states[member_id] = member_states

    # Determine number of forecast days from first member
    first_member = next(iter(all_member_states.values()))
    n_ens_days = len(first_member)
    n_arome = min(arome_days, len(arome_states)) if arome_states else 0
    n_days = n_arome + n_ens_days
    n_members = len(all_member_states)

    # Build per-day ensemble statistics
    daily_stats: list[DailyEnsembleStats] = []

    for day_idx in range(n_days):
        # For days within AROME range: use AROME deterministic
        if arome_states and day_idx < arome_days and day_idx < len(arome_states):
            arome = arome_states[day_idx]
            daily_stats.append(
                DailyEnsembleStats(
                    date=pd.Timestamp(arome.date),
                    day_offset=day_idx,
                    source="arome",
                    ks_mean=arome.stress_coeff,
                    ks_median=arome.stress_coeff,
                    ks_p25=arome.stress_coeff,
                    ks_p75=arome.stress_coeff,
                    ks_min=arome.stress_coeff,
                    ks_max=arome.stress_coeff,
                    depletion_mean=arome.depletion,
                    depletion_p25=arome.depletion,
                    depletion_p75=arome.depletion,
                    precip_mean=arome.precip,
                    precip_p25=arome.precip,
                    precip_p75=arome.precip,
                    etc_mean=arome.etc_act,
                    n_members_stressed=1 if arome.stress_coeff < stress_threshold else 0,
                    n_members_total=1,
                )
            )
            continue

        # For days beyond AROME: aggregate across IFS ENS members.
        # member_states[0] = first day after AROME handoff, so offset.
        ens_idx = day_idx - n_arome
        day_ks = []
        day_depletion = []
        day_precip = []
        day_etc = []
        day_date = None

        for member_id, states in all_member_states.items():
            if ens_idx < len(states):
                s = states[ens_idx]
                day_ks.append(s.stress_coeff)
                day_depletion.append(s.depletion)
                day_precip.append(s.precip)
                day_etc.append(s.etc_act)
                if day_date is None:
                    day_date = s.date

        if not day_ks:
            continue

        ks_arr = np.array(day_ks)
        dr_arr = np.array(day_depletion)
        pr_arr = np.array(day_precip)
        etc_arr = np.array(day_etc)

        daily_stats.append(
            DailyEnsembleStats(
                date=pd.Timestamp(day_date),
                day_offset=day_idx,
                source="ifs_ens",
                ks_mean=float(ks_arr.mean()),
                ks_median=float(np.median(ks_arr)),
                ks_p25=float(np.percentile(ks_arr, 25)),
                ks_p75=float(np.percentile(ks_arr, 75)),
                ks_min=float(ks_arr.min()),
                ks_max=float(ks_arr.max()),
                depletion_mean=float(dr_arr.mean()),
                depletion_p25=float(np.percentile(dr_arr, 25)),
                depletion_p75=float(np.percentile(dr_arr, 75)),
                precip_mean=float(pr_arr.mean()),
                precip_p25=float(np.percentile(pr_arr, 25)),
                precip_p75=float(np.percentile(pr_arr, 75)),
                etc_mean=float(etc_arr.mean()),
                n_members_stressed=int((ks_arr < stress_threshold).sum()),
                n_members_total=len(ks_arr),
            )
        )

    report = EnsembleStressReport(
        daily_stats=daily_stats,
        n_members=n_members,
        arome_days=n_arome,
        ens_days=n_ens_days,
    )

    logger.info(
        "Ensemble stress report: %d days (%d AROME + %d IFS ENS), "
        "%d members, stress_expected=%s, first_stress_day=%s",
        len(daily_stats),
        report.arome_days,
        report.ens_days,
        n_members,
        report.stress_expected,
        report.first_stress_day,
    )
    return report


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------


def will_stress_occur(
    forecast_states: list[WaterBalanceState],
    threshold: float = 0.9,
) -> dict:
    """Check if stress develops (single-path, for backward compat)."""
    if not forecast_states:
        return {"stress_expected": False}
    stress_days = [s for s in forecast_states if s.stress_coeff < threshold]
    if not stress_days:
        return {
            "stress_expected": False,
            "min_ks": min(s.stress_coeff for s in forecast_states),
            "forecast_days": len(forecast_states),
        }
    first = stress_days[0]
    worst = min(stress_days, key=lambda s: s.stress_coeff)
    return {
        "stress_expected": True,
        "days_until_stress": (first.date - forecast_states[0].date).days,
        "first_stress_date": first.date,
        "worst_ks": worst.stress_coeff,
        "worst_date": worst.date,
        "forecast_days": len(forecast_states),
        "total_forecast_precip": sum(s.precip for s in forecast_states),
    }
