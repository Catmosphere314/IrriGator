"""Short-term forecast integration: AROME/ARPEGE → daily xr.Dataset.

Processes NWP forecast data into the same daily xr.Dataset format as
ERA5-Land, so it can be passed through ``extract_parcel_forcing()``
for parcel-level downscaling.  This avoids duplicating the downscaling
logic from Block 2.

Resolution notes
----------------
- ARPEGE (~10 km, 96h): gets lapse rate + radiation correction through
  extract_parcel_forcing, same as ERA5-Land.
- AROME (~1.3 km, 48h): already fine-scale, but still passes through
  extract_parcel_forcing for wind conversion and consistency.

After downscaling, the forward water balance runs on the resulting
DailyForcing to predict stress within the forecast horizon.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from irrigator.atmospheric.forcing import DailyForcing, extract_parcel_forcing
from irrigator.config import ParcelConfig
from irrigator.crop.phenology import CropParams, advance_crop
from irrigator.static_layers.soil import SoilProfile
from irrigator.static_layers.terrain import TerrainParams
from irrigator.water_balance.bucket_model import daily_step
from irrigator.water_balance.et0 import compute_et0
from irrigator.water_balance.state import WaterBalanceState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forecast standardization
# ---------------------------------------------------------------------------


def standardize_forecast_to_era5_format(
    forecast_ds: xr.Dataset,
    source: str = "arpege",
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
        # Temperature
        "2t": "t_mean",
        "t2m": "t_mean",
        "mn2t": "t_min",
        "mx2t": "t_max",
        # Wind
        "10u": "u10",
        "10v": "v10",
        # Radiation, pressure, precip
        "ssrd": "rs_mj",
        "ssr": "rs_mj",
        "sp": "pressure_kpa",
        "tp": "precip_mm",
        "2d": "dewpoint",
        "d2m": "dewpoint",
    }

    result = forecast_ds.copy()

    # Rename variables that exist
    for old, new in rename_candidates.items():
        if old in result.data_vars and new not in result.data_vars:
            result = result.rename({old: new})

    # Unit conversions if needed
    # Temperature: K → °C (if values suggest Kelvin)
    for tvar in ("t_mean", "t_min", "t_max", "dewpoint"):
        if tvar in result and float(result[tvar].mean()) > 100:
            result[tvar] = result[tvar] - 273.15

    # Wind: combine u,v if separate
    if "u10" in result and "v10" in result and "wind_speed_10m" not in result:
        result["wind_speed_10m"] = np.sqrt(result["u10"] ** 2 + result["v10"] ** 2)

    # Pressure: Pa → kPa
    if "pressure_kpa" in result and float(result["pressure_kpa"].mean()) > 10000:
        result["pressure_kpa"] = result["pressure_kpa"] / 1000.0

    # Ensure time dimension is named consistently
    for old_time in ("time", "step", "forecast_time"):
        if old_time in result.dims and "valid_time" not in result.dims:
            result = result.rename({old_time: "valid_time"})

    logger.info(
        "Standardized %s forecast: %d variables, %d timesteps",
        source,
        len(result.data_vars),
        result.sizes.get("valid_time", 0),
    )
    return result


# ---------------------------------------------------------------------------
# Forward water balance under forecast
# ---------------------------------------------------------------------------


def run_forward_balance(
    current_state: WaterBalanceState,
    forecast_forcing: DailyForcing,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil: SoilProfile,
    crop_params: CropParams,
) -> list[WaterBalanceState]:
    """Run water balance forward under forecast forcing.

    Takes today's state and extends through the forecast horizon
    (2-4 days) with NO irrigation, to see if stress will develop.

    Parameters
    ----------
    current_state : today's WaterBalanceState (from historical simulation)
    forecast_forcing : DailyForcing (already downscaled to parcel via
                       extract_parcel_forcing)
    parcel, terrain, soil, crop_params : the usual

    Returns
    -------
    List of WaterBalanceState for each forecast day.
    """
    et0_series = compute_et0(forecast_forcing, terrain, parcel.lat)
    if hasattr(et0_series, "values"):
        et0_series = et0_series.values
    et0_series = np.asarray(et0_series, dtype=np.float64)

    dr = current_state.depletion
    gdd = current_state.gdd
    taw_prev = current_state.taw
    forecast_states = []

    for i in range(forecast_forcing.n_days):
        current_date = pd.Timestamp(forecast_forcing.dates[i]).date()
        t_min = float(forecast_forcing.t_min[i])
        t_max = float(forecast_forcing.t_max[i])
        et0 = float(et0_series[i])
        precip = float(forecast_forcing.precip_mm[i])

        crop = advance_crop(current_date, t_min, t_max, gdd, crop_params)

        state = daily_step(
            current_date=current_date,
            et0=et0,
            precip=precip,
            irrigation=0.0,  # no irrigation in forecast — we want to see what happens
            crop=crop,
            soil=soil,
            dr_prev=dr,
            taw_prev=taw_prev,
            p=crop_params.p,
        )

        dr = state.depletion
        taw_prev = state.taw
        gdd = crop.gdd
        forecast_states.append(state)

    return forecast_states


def will_stress_occur(
    forecast_states: list[WaterBalanceState],
    threshold: float = 0.9,
) -> dict:
    """Check if stress develops within the forecast horizon."""
    if not forecast_states:
        return {"stress_expected": False}

    stress_days = [s for s in forecast_states if s.stress_coeff < threshold]

    if not stress_days:
        return {
            "stress_expected": False,
            "min_ks": min(s.stress_coeff for s in forecast_states),
            "forecast_days": len(forecast_states),
        }

    first_stress = stress_days[0]
    worst_stress = min(stress_days, key=lambda s: s.stress_coeff)

    return {
        "stress_expected": True,
        "days_until_stress": (first_stress.date - forecast_states[0].date).days,
        "first_stress_date": first_stress.date,
        "worst_ks": worst_stress.stress_coeff,
        "worst_date": worst_stress.date,
        "forecast_days": len(forecast_states),
        "total_forecast_precip": sum(s.precip for s in forecast_states),
    }
