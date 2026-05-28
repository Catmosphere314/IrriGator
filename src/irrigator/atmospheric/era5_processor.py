"""ERA5-Land hourly-to-daily processor.

Converts raw ERA5-Land hourly data into daily aggregates suitable for
FAO-56 Penman-Monteith ET0 computation.

Variable conversions
--------------------
- Temperature: hourly K → daily Tmin, Tmax, Tmean in °C
- Dewpoint: hourly K → daily mean in °C
- Wind: u10, v10 → daily mean wind speed at 10m [m/s]
- Pressure: hourly Pa → daily mean [kPa]
- Radiation: hourly accumulated J/m² → daily total [MJ/m²/day]
- Precipitation: hourly accumulated m → daily total [mm/day]

ERA5-Land accumulation convention
---------------------------------
Surface solar radiation and precipitation in ERA5-Land are accumulated
from the start of the forecast step.  For hourly data, each timestep's
value represents the accumulation over that hour.  Daily totals are
simply the sum of all 24 hourly values.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from irrigator.config import RegionConfig

logger = logging.getLogger(__name__)

# ERA5-Land variable names (may differ between CDS versions)
_VAR_MAP = {
    "t2m": "2m_temperature",  # can also be "t2m"
    "d2m": "2m_dewpoint_temperature",  # can also be "d2m"
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "sp": "surface_pressure",
    "ssrd": "surface_solar_radiation_downwards",
    "tp": "total_precipitation",
}


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str:
    """Find which variable name exists in the dataset."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of {candidates} found in dataset. Available: {list(ds.data_vars)}")


def process_era5_to_daily(ds: xr.Dataset) -> xr.Dataset:
    """Convert ERA5-Land hourly dataset to daily aggregates.

    Parameters
    ----------
    ds : xr.Dataset
        Hourly ERA5-Land data (from open_era5_land or open_mfdataset).

    Returns
    -------
    xr.Dataset with daily variables:
        t_min, t_max, t_mean, dewpoint, wind_speed_10m,
        pressure_kpa, rs_mj, precip_mm
    """
    # Identify variable names (ERA5 naming varies between CDS versions)
    t2m = _find_var(ds, ["t2m", "2m_temperature", "VAR_2T"])
    d2m = _find_var(ds, ["d2m", "2m_dewpoint_temperature", "VAR_2D"])
    u10 = _find_var(ds, ["u10", "10m_u_component_of_wind"])
    v10 = _find_var(ds, ["v10", "10m_v_component_of_wind"])
    sp = _find_var(ds, ["sp", "surface_pressure"])
    ssrd = _find_var(ds, ["ssrd", "surface_solar_radiation_downwards"])
    tp = _find_var(ds, ["tp", "total_precipitation"])

    daily = ds.resample(time="1D")

    # Temperature: K → °C
    t_min = daily[t2m].min() - 273.15
    t_max = daily[t2m].max() - 273.15
    t_mean = daily[t2m].mean() - 273.15

    # Dewpoint: K → °C
    dewpoint = daily[d2m].mean() - 273.15

    # Wind speed at 10m: combine u and v components
    wind_speed = np.sqrt(ds[u10] ** 2 + ds[v10] ** 2)
    wind_10m = wind_speed.resample(time="1D").mean()

    # Pressure: Pa → kPa
    pressure = daily[sp].mean() / 1000.0

    # Solar radiation: J/m² (accumulated per hour) → MJ/m²/day
    # Sum hourly values, convert J → MJ
    rs = daily[ssrd].sum() / 1e6

    # Precipitation: m (accumulated per hour) → mm/day
    # Sum hourly values, convert m → mm
    precip = daily[tp].sum() * 1000.0
    # ERA5-Land can have tiny negative values from numerical noise
    precip = precip.clip(min=0)

    result = xr.Dataset(
        {
            "t_min": t_min.rename("t_min"),
            "t_max": t_max.rename("t_max"),
            "t_mean": t_mean.rename("t_mean"),
            "dewpoint": dewpoint.rename("dewpoint"),
            "wind_speed_10m": wind_10m.rename("wind_speed_10m"),
            "pressure_kpa": pressure.rename("pressure_kpa"),
            "rs_mj": rs.rename("rs_mj"),
            "precip_mm": precip.rename("precip_mm"),
        }
    )

    result.attrs["source"] = "ERA5-Land daily aggregates"
    result.attrs["temperature_units"] = "degC"
    result.attrs["pressure_units"] = "kPa"
    result.attrs["radiation_units"] = "MJ/m2/day"
    result.attrs["precipitation_units"] = "mm/day"

    logger.info(
        "Daily aggregation: %d days, T range [%.1f, %.1f] °C",
        len(result.time),
        float(result.t_min.min()),
        float(result.t_max.max()),
    )

    return result


def save_daily(ds: xr.Dataset, cfg: RegionConfig) -> None:
    """Save daily aggregated ERA5-Land data."""
    out_dir = cfg.processed_dir / "atmospheric"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "era5_daily.nc"
    ds.to_netcdf(out_path)
    logger.info("Saved daily ERA5-Land: %s", out_path)


def load_daily(cfg: RegionConfig) -> xr.Dataset:
    """Load previously saved daily ERA5-Land data."""
    path = cfg.processed_dir / "atmospheric" / "era5_daily.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"Daily ERA5-Land not found: {path}\nRun process_era5_to_daily first."
        )
    return xr.open_dataset(path)
