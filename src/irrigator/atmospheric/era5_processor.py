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
import pandas as pd
import xarray as xr

from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths — relative to repo root (assumes cwd = repo root)
# ---------------------------------------------------------------------------

DEFAULT_PROCESSED_DIR = Path("data/processed")

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


def process_era5_to_daily(ds: xr.Dataset, shift_utc: int = 0) -> xr.Dataset:
    """Convert ERA5-Land hourly dataset to daily aggregates.

    Parameters
    ----------
    ds : xr.Dataset
        Hourly ERA5-Land data (from open_era5_land or open_mfdataset).
    shift_utc : shift to apply compared to UTC, 0 default,
        can be set to 1 for French time (omitting the time change)

    Returns
    -------
    xr.Dataset with daily variables:
        t_min, t_max, t_mean, dewpoint, wind_speed_10m,
        pressure_kpa, rs_mj, precip_mm
    """
    # Shift UTC → CET before daily aggregation
    ds = ds.assign_coords(valid_time=ds.valid_time + pd.Timedelta(f"{shift_utc}h"))

    # Identify variable names (ERA5 naming varies between CDS versions)
    t2m = _find_var(ds, ["t2m", "2m_temperature", "VAR_2T"])
    d2m = _find_var(ds, ["d2m", "2m_dewpoint_temperature", "VAR_2D"])
    u10 = _find_var(ds, ["u10", "10m_u_component_of_wind"])
    v10 = _find_var(ds, ["v10", "10m_v_component_of_wind"])
    sp = _find_var(ds, ["sp", "surface_pressure"])
    ssrd = _find_var(ds, ["ssrd", "surface_solar_radiation_downwards"])
    tp = _find_var(ds, ["tp", "total_precipitation"])

    # Temperature: K → °C
    temp_ds = ds[t2m].resample(valid_time="1D")
    t_min = temp_ds.min() - 273.15
    t_max = temp_ds.max() - 273.15
    t_mean = temp_ds.mean() - 273.15
    del temp_ds

    # Dewpoint: K → °C
    dew_ds = ds[d2m].resample(valid_time="1D")
    dewpoint = dew_ds.mean() - 273.15
    del dew_ds

    # Wind speed at 10m: combine u and v components
    wind_speed = np.sqrt(ds[u10] ** 2 + ds[v10] ** 2)
    wind_10m = wind_speed.resample(valid_time="1D").mean()

    # Pressure: Pa → kPa
    pres_ds = ds[sp].resample(valid_time="1D")
    pressure = pres_ds.mean() / 1000.0
    del pres_ds

    # Solar radiation: J/m² (accumulated per hour) → MJ/m²/day
    # Daily total = value at 00UTC of d+1, which holds the previous day's accumulation
    rad_00utc = ds[ssrd].sel(valid_time=ds.valid_time.dt.hour == 0)
    # Shift back by one day so it aligns with the correct date
    rs = rad_00utc.assign_coords(valid_time=rad_00utc.valid_time - pd.Timedelta("1D")) / 1e6
    del rad_00utc

    # Precipitation: m (accumulated per hour) → mm/day
    # Daily total = value at 00UTC of d+1, which holds the previous day's accumulation
    tp_00utc = ds[tp].sel(valid_time=ds.valid_time.dt.hour == 0)
    # Shift back by one day so it aligns with the correct date
    precip = tp_00utc.assign_coords(valid_time=tp_00utc.valid_time - pd.Timedelta("1D")) * 1000.0
    del tp_00utc
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

    # Trim edges: the 00UTC shift creates a spurious first date (day before data start)
    # and leaves NaN on the last date (no next-day 00UTC available)
    valid_start = ds.valid_time.values[0].astype("datetime64[D]")
    valid_end = ds.valid_time.values[-1].astype("datetime64[D]") - np.timedelta64(1, "D")
    result = result.sel(valid_time=slice(str(valid_start), str(valid_end)))

    logger.info(
        "Daily aggregation: %d days, T range [%.1f, %.1f] °C",
        len(result.valid_time),
        float(result.t_min.min()),
        float(result.t_max.max()),
    )

    return result


def save_daily(ds: xr.Dataset, processed_dir: str | Path = DEFAULT_PROCESSED_DIR) -> Path:
    """Save daily aggregated ERA5-Land data.

    Parameters
    ----------
    ds : daily ERA5-Land dataset (from process_era5_to_daily)
    processed_dir : output directory (default: data/processed)

    Returns
    -------
    Path to saved file.
    """
    processed_dir = Path(processed_dir)
    out_dir = processed_dir / "atmospheric"
    out_dir.mkdir(parents=True, exist_ok=True)
    year = str(ds.valid_time[0].dt.year.values)
    out_path = out_dir / f"era5_daily_{year}.nc"

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    size_mb = out_path.stat().st_size / 1e6
    logger.info("Saved daily ERA5-Land: %s (%.1f MB)", out_path, size_mb)
    return out_path


def load_daily(processed_dir: str | Path = DEFAULT_PROCESSED_DIR, year: int = 2025) -> xr.Dataset:
    """Load previously saved daily ERA5-Land data.

    Parameters
    ----------
    processed_dir : directory containing atmospheric/era5_daily.nc
    year : year for which to load data
    """
    processed_dir = Path(processed_dir)
    path = processed_dir / "atmospheric" / f"era5_daily_{year}.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"Daily ERA5-Land not found: {path}\nRun process_era5_to_daily first."
        )
    return xr.open_dataset(path)
