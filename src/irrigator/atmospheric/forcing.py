"""Daily meteorological forcing at parcel level.

This module ties together all Block 2 components:
1. Extract ERA5-Land daily data at the parcel grid cell
2. Apply temperature lapse-rate correction (terrain)
3. Apply precipitation bias correction (COMEPHORE)
4. Apply radiation terrain correction (slope/aspect)
5. Convert wind speed from 10m to 2m
6. Package into DailyForcing time series

The DailyForcing dataclass contains everything needed to compute
ET0 (Penman-Monteith) and run the soil water balance (Block 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from irrigator.atmospheric.precip_bias_correction import BiasCorrection
from irrigator.atmospheric.radiation_correction import correct_radiation
from irrigator.atmospheric.temperature_downscale import correct_temperature
from irrigator.config import ParcelConfig
from irrigator.static_layers.terrain import TerrainParams

logger = logging.getLogger(__name__)

_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)


@dataclass
class DailyForcing:
    """Daily meteorological forcing for one parcel.

    All values are arrays of length n_days.

    Attributes
    ----------
    dates : array of dates
    t_min : minimum temperature [°C]
    t_max : maximum temperature [°C]
    t_mean : mean temperature [°C]
    dewpoint : mean dewpoint temperature [°C]
    wind_speed_2m : wind speed at 2m height [m/s]
    pressure_kpa : surface pressure [kPa]
    rs_mj : incoming solar radiation [MJ/m²/day]
    precip_mm : precipitation [mm/day]
    """

    dates: np.ndarray
    t_min: np.ndarray
    t_max: np.ndarray
    t_mean: np.ndarray
    dewpoint: np.ndarray
    wind_speed_2m: np.ndarray
    pressure_kpa: np.ndarray
    rs_mj: np.ndarray
    precip_mm: np.ndarray

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for easy inspection."""
        return pd.DataFrame(
            {
                "date": self.dates,
                "t_min": self.t_min,
                "t_max": self.t_max,
                "t_mean": self.t_mean,
                "dewpoint": self.dewpoint,
                "wind_speed_2m": self.wind_speed_2m,
                "pressure_kpa": self.pressure_kpa,
                "rs_mj": self.rs_mj,
                "precip_mm": self.precip_mm,
            }
        ).set_index("date")

    def slice(self, start: date, end: date) -> DailyForcing:
        """Return a DailyForcing for a date sub-range."""
        mask = (self.dates >= np.datetime64(start)) & (self.dates <= np.datetime64(end))
        return DailyForcing(
            dates=self.dates[mask],
            t_min=self.t_min[mask],
            t_max=self.t_max[mask],
            t_mean=self.t_mean[mask],
            dewpoint=self.dewpoint[mask],
            wind_speed_2m=self.wind_speed_2m[mask],
            pressure_kpa=self.pressure_kpa[mask],
            rs_mj=self.rs_mj[mask],
            precip_mm=self.precip_mm[mask],
        )


# ---------------------------------------------------------------------------
# Wind height conversion
# ---------------------------------------------------------------------------


def wind_10m_to_2m(wind_10m: float | np.ndarray) -> float | np.ndarray:
    """Convert wind speed from 10m to 2m height.

    Uses the logarithmic wind profile (FAO-56 eq. 47):
        u2 = uz × 4.87 / ln(67.8 × z − 5.42)

    For z=10m: u2 = u10 × 4.87 / ln(672.58) = u10 × 0.748
    """
    return wind_10m * (4.87 / np.log(67.8 * 10 - 5.42))


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_parcel_forcing(
    era5_daily: xr.Dataset,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    precip_correction: BiasCorrection | None = None,
) -> DailyForcing:
    """Extract and downscale daily forcing at a parcel location.

    This is the main entry point for Block 2.  It:
    1. Extracts ERA5-Land daily values at the nearest grid cell
    2. Applies temperature lapse-rate correction
    3. Applies precipitation bias correction (if provided)
    4. Corrects radiation for slope/aspect
    5. Converts wind from 10m to 2m

    Parameters
    ----------
    era5_daily : xr.Dataset
        Daily ERA5-Land data (from era5_processor.process_era5_to_daily).
    parcel : ParcelConfig
    terrain : TerrainParams (with era5_elevation_m set)
    precip_correction : fitted BiasCorrection (optional)

    Returns
    -------
    DailyForcing time series for the parcel.
    """

    # Step 1: extract at nearest grid cell
    cell = era5_daily.sel(longitude=parcel.lon, latitude=parcel.lat, method="nearest")
    dates = cell.valid_time.values
    n = len(dates)

    t_min = cell["t_min"].values.astype(np.float64)
    t_max = cell["t_max"].values.astype(np.float64)
    t_mean = cell["t_mean"].values.astype(np.float64)
    dewpoint = cell["dewpoint"].values.astype(np.float64)
    wind_10m = cell["wind_speed_10m"].values.astype(np.float64)
    pressure = cell["pressure_kpa"].values.astype(np.float64)
    rs = cell["rs_mj"].values.astype(np.float64)
    precip = cell["precip_mm"].values.astype(np.float64)

    logger.info(
        "Parcel %s: extracted %d days from ERA5-Land at grid cell (%.0f, %.0f)",
        parcel.id,
        n,
        parcel.lon,
        parcel.lat,
    )

    # Step 2: temperature lapse-rate correction
    if terrain.era5_elevation_m is not None:
        z_era5 = terrain.era5_elevation_m
        z_local = terrain.elevation_m
        t_min = correct_temperature(t_min, z_era5, z_local)
        t_max = correct_temperature(t_max, z_era5, z_local)
        t_mean = correct_temperature(t_mean, z_era5, z_local)
        dewpoint = correct_temperature(dewpoint, z_era5, z_local)
        dz = z_local - z_era5
        logger.info(
            "Lapse correction: Δz=%.0f m, ΔT=%.2f°C",
            dz,
            -0.0065 * dz,
        )

    # Step 3: precipitation bias correction
    if precip_correction is not None and precip_correction.method != "none":
        precip_raw = precip.copy()
        precip = precip_correction.correct(precip)
        logger.info(
            "Precip correction (%s): raw total=%.0f mm → corrected=%.0f mm",
            precip_correction.method,
            precip_raw.sum(),
            precip.sum(),
        )

    # Step 4: radiation terrain correction
    for i in range(n):
        dt = pd.Timestamp(dates[i])
        doy = dt.timetuple().tm_yday
        rs[i] = correct_radiation(rs[i], doy, parcel.lat, terrain)

    # Step 5: wind height conversion (10m → 2m)
    wind_2m = wind_10m_to_2m(wind_10m)

    forcing = DailyForcing(
        dates=dates,
        t_min=t_min,
        t_max=t_max,
        t_mean=t_mean,
        dewpoint=dewpoint,
        wind_speed_2m=wind_2m,
        pressure_kpa=pressure,
        rs_mj=rs,
        precip_mm=precip,
    )

    logger.info(
        "Parcel %s forcing: %d days, T=[%.1f, %.1f]°C, "
        "total precip=%.0f mm, mean Rs=%.1f MJ/m²/day",
        parcel.id,
        n,
        float(t_min.min()),
        float(t_max.max()),
        float(precip.sum()),
        float(rs.mean()),
    )

    return forcing
