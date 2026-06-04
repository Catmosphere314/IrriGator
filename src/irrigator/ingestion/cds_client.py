"""CDS API client for ERA5-Land and SEAS5 data retrieval.

Prerequisites
-------------
- Install ``cdsapi``: ``pip install cdsapi``
- Create ``~/.cdsapirc`` with your CDS API key:
      url: https://cds.climate.copernicus.eu/api
      key: <your-uid>:<your-api-key>

Notes
-----
ERA5-Land is hourly at ~9 km.  We request full days and aggregate to daily
in the atmospheric processing step (Block 2), not here.  This module only
handles download and raw storage.

SEAS5 is monthly, ~36 km, 51-member ensemble.  We download monthly means
for temperature and total precipitation.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import cdsapi
import xarray as xr

from irrigator.config import RegionConfig

logger = logging.getLogger(__name__)

# ERA5-Land variables needed for Penman-Monteith ET0
ERA5_LAND_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "total_precipitation",
]

# Additional ERA5-Land variables for soil moisture validation
ERA5_LAND_VALIDATION_VARIABLES = [
    "volumetric_soil_water_layer_1",  # 0-7 cm
    "volumetric_soil_water_layer_2",  # 7-28 cm
    "volumetric_soil_water_layer_3",  # 28-100 cm
    "volumetric_soil_water_layer_4",  # 100-289 cm
]

# SEAS5 variables
SEAS5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
]


def _init_cds_client() -> cdsapi.Client:
    """Initialise CDS API client with quiet logging."""
    try:
        client = cdsapi.Client(quiet=False)
    except Exception as exc:
        raise RuntimeError(
            "CDS API client failed to initialise. "
            "Check that ~/.cdsapirc exists and contains valid credentials. "
            "Register at https://cds.climate.copernicus.eu/"
        ) from exc
    return client


# ---------------------------------------------------------------------------
# ERA5-Land
# ---------------------------------------------------------------------------


def _era5_output_path(raw_dir: Path, year: int, month: int, *, validation: bool = False) -> Path:
    """Consistent file naming: era5land_YYYY_MM.nc"""
    out_dir = raw_dir / "era5_land"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        out_dir / f"era5land_{year:04d}_{month:02d}.nc"
        if not validation
        else out_dir / f"era5land_validation_{year:04d}_{month:02d}.nc"
    )
    return output_path


def fetch_era5_land_month(
    cfg: RegionConfig,
    year: int,
    month: int,
    *,
    include_validation: bool = False,
    overwrite: bool = False,
) -> Path:
    """Download one month of hourly ERA5-Land data for the region.

    Parameters
    ----------
    cfg : RegionConfig
    year, month : target period
    include_validation : if True, also download soil moisture layers
    overwrite : re-download even if file exists

    Returns
    -------
    Path to the downloaded NetCDF file.
    """
    out_path = _era5_output_path(cfg.raw_dir, year, month)
    out_path_validation = (
        _era5_output_path(cfg.raw_dir, year, month, validation=True) if include_validation else None
    )

    if (
        out_path.exists()
        and (out_path_validation is None or out_path_validation.exists())
        and not overwrite
    ):
        logger.info("ERA5-Land %04d-%02d already exists: %s", year, month, out_path)
        return out_path

    variables = list(ERA5_LAND_VARIABLES)

    # Build day list for the month
    first_day = date(year, month, 1)
    if month == 12:
        last_day = date(year, 12, 31)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    days = [f"{d:02d}" for d in range(1, last_day.day + 1)]

    request = {
        "variable": variables,
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": cfg.bbox_wgs84.as_cds_area(),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(request)

    logger.info("Requesting ERA5-Land %04d-%02d from CDS...", year, month)
    client = _init_cds_client()
    client.retrieve("reanalysis-era5-land", request, str(out_path))
    logger.info("Downloaded: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    if include_validation and out_path_validation is not None:
        variables = ERA5_LAND_VALIDATION_VARIABLES

        request = {
            "variable": variables,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": cfg.bbox_wgs84.as_cds_area(),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        logger.info("Requesting ERA5-Land %04d-%02d from CDS...", year, month)
        client = _init_cds_client()
        client.retrieve("reanalysis-era5-land", request, str(out_path_validation))
        logger.info(
            "Downloaded: %s (%.1f MB)",
            out_path_validation,
            out_path_validation.stat().st_size / 1e6,
        )

    return out_path


def fetch_era5_land_range(
    cfg: RegionConfig,
    start: date,
    end: date,
    **kwargs,
) -> list[Path]:
    """Download ERA5-Land for a date range, month by month.

    Returns list of paths to downloaded files.
    """
    paths = []
    current = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)

    while current <= end_month:
        path = fetch_era5_land_month(cfg, current.year, current.month, **kwargs)
        paths.append(path)
        # Advance to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return paths


def open_era5_land(
    cfg: RegionConfig,
    start: date | None = None,
    end: date | None = None,
    parent_path: str | None = None,
) -> xr.Dataset:
    """Open downloaded ERA5-Land files as a single lazy xarray Dataset.

    Uses dask for out-of-core access.  Call ``.load()`` or ``.compute()``
    to materialise into memory.
    """
    era5_dir = cfg.raw_dir / "era5_land"
    if not era5_dir.exists():
        raise FileNotFoundError(
            f"No ERA5-Land data found at {era5_dir}. Run fetch_era5_land_range first."
        )

    files = sorted(era5_dir.glob("era5land_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5-Land NetCDF files in {era5_dir}")

    # Filter by date range if specified
    if start or end:
        filtered = []
        for f in files:
            # Parse year/month from filename: era5land_YYYY_MM.nc
            parts = f.stem.split("_")
            file_year, file_month = int(parts[1]), int(parts[2])
            file_date = date(file_year, file_month, 1)
            if start and file_date < date(start.year, start.month, 1):
                continue
            if end and file_date > date(end.year, end.month, 1):
                continue
            filtered.append(f)
        files = filtered

    logger.info("Opening %d ERA5-Land files with dask", len(files))
    ds = xr.open_mfdataset(files, chunks={"time": 24}, combine="by_coords")
    return ds


# ---------------------------------------------------------------------------
# SEAS5
# ---------------------------------------------------------------------------


def _seas5_output_path(raw_dir: Path, year: int, month: int) -> Path:
    out_dir = raw_dir / "seas5"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"seas5_{year:04d}_{month:02d}.nc"


def fetch_seas5(
    cfg: RegionConfig,
    year: int,
    month: int,
    *,
    overwrite: bool = False,
) -> Path:
    """Download SEAS5 seasonal forecast initialised at year/month.

    Downloads all lead times (1-6 months) and all ensemble members.

    Parameters
    ----------
    cfg : RegionConfig
    year, month : initialisation date of the forecast
    overwrite : re-download even if file exists

    Returns
    -------
    Path to downloaded NetCDF.
    """
    out_path = _seas5_output_path(cfg.raw_dir, year, month)

    if out_path.exists() and not overwrite:
        logger.info("SEAS5 %04d-%02d already exists: %s", year, month, out_path)
        return out_path

    seas5_cfg = cfg.data.get("seas5", {})
    leadtime_months = seas5_cfg.get("leadtime_months", [1, 2, 3, 4, 5, 6])

    request = {
        "originating_centre": "ecmwf",
        "system": str(seas5_cfg.get("system", 51)),
        "variable": SEAS5_VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "leadtime_month": [str(m) for m in leadtime_months],
        "area": cfg.bbox_wgs84.as_cds_area(),
        "data_format": "netcdf",
    }

    logger.info("Requesting SEAS5 %04d-%02d from CDS...", year, month)
    client = _init_cds_client()
    client.retrieve("seasonal-monthly-single-levels", request, str(out_path))
    logger.info("Downloaded: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    return out_path


def open_seas5(cfg: RegionConfig, year: int, month: int) -> xr.Dataset:
    """Open a downloaded SEAS5 forecast file."""
    path = _seas5_output_path(cfg.raw_dir, year, month)
    if not path.exists():
        raise FileNotFoundError(f"SEAS5 file not found: {path}. Run fetch_seas5 first.")
    return xr.open_dataset(path)
