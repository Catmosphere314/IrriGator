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

All ERA5-Land downloads use the France metropolitan bounding box by default
(41°N–51.5°N, 6°W–10°E).  Any new parcel is immediately served from the
same cache — no per-region download needed.

SEAS5 is monthly, ~36 km, 51-member ensemble.  We download monthly means
for temperature and total precipitation.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from calendar import monthrange

import cdsapi
import uuid
import xarray as xr

from irrigator.config import BBoxWGS84

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths — relative to repo root (assumes cwd = repo root)
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# France metropolitan bounding box (used for all downloads)
# ---------------------------------------------------------------------------

FRANCE_BBOX = BBoxWGS84(north=51.5, south=41.0, west=-6.0, east=10.0)

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

ERA5_MONTH_VARIABLES=[
    "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "total_precipitation",
        "10m_wind_speed",
        "surface_solar_radiation_downwards",
        "evaporation"]

# SEAS5 variables
SEAS5_VARIABLES = [
    "2m_temperature",
    "minimum_2m_temperature_in_the_last_24_hours",
    "maximum_2m_temperature_in_the_last_24_hours",
    "2m_dewpoint_temperature",
    "10m_wind_speed",
    "surface_solar_radiation_downwards",
    "total_precipitation",
    "evaporation",
    "mean_sea_level_pressure",
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

    if validation:
        return out_dir / f"era5land_validation_{year:04d}_{month:02d}.nc"
    return out_dir / f"era5land_{year:04d}_{month:02d}.nc"


def fetch_era5_land_month(
    year: int,
    month: int,
    *,
    bounding_box: BBoxWGS84 = FRANCE_BBOX,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    include_validation: bool = False,
    overwrite: bool = False,
    force_cds_refresh: bool = False
) -> Path:
    """Download one month of hourly ERA5-Land data.

    Parameters
    ----------
    year, month : target period
    bounding_box : spatial extent (default: France metropolitan)
    raw_dir : output directory for ERA5 downloads
    include_validation : if True, also download soil moisture layers
    overwrite : re-download even if file exists
    force_cds_refresh : to avoid cache issues when overwriting

    Returns
    -------
    Path to the downloaded NetCDF file.
    """
    raw_dir = Path(raw_dir)
    out_path = _era5_output_path(raw_dir, year, month)
    out_path_validation = (
        _era5_output_path(raw_dir, year, month, validation=True) if include_validation else None
    )

    if (
        out_path.exists()
        and (out_path_validation is None or out_path_validation.exists())
        and not overwrite
    ):
        with xr.open_dataset(
            out_path,
            chunks=None,  # no dask
            decode_cf=False,  # skip CF decoding
            mask_and_scale=False,  # skip scale/masking setup
            create_default_indexes=False,  # avoid loading dim coords into pandas indexes
            cache=False,
        ) as ds:
            n_time = ds.sizes["valid_time"]
        if n_time // 24 == monthrange(year, month)[1] and n_time % 24 == 0:
            logger.info("ERA5-Land %04d-%02d already exists: %s", year, month, out_path)
            return out_path
        else:
            logger.info(
                "ERA5-Land %04d-%02d already exists, BUT OVERWRITTEN AS THE MONTH IS NOT FULL: %s",
                year,
                month,
                out_path,
            )
            ds.close()

    # Build day list for the month
    first_day = date(year, month, 1)
    if month == 12:
        last_day = date(year, 12, 31)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    days = [f"{d:02d}" for d in range(1, last_day.day + 1)]
    print(days)


    request = {
        "variable": list(ERA5_LAND_VARIABLES),
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": bounding_box.as_cds_area(),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    if force_cds_refresh:
        request["nocache"] = '123'

    logger.info("Requesting ERA5-Land %04d-%02d from CDS...", year, month)
    client = _init_cds_client()
    client.retrieve("reanalysis-era5-land", request, str(out_path))
    logger.info("Downloaded: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    if include_validation and out_path_validation is not None:
        val_request = {
            "variable": list(ERA5_LAND_VALIDATION_VARIABLES),
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": bounding_box.as_cds_area(),
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        logger.info("Requesting ERA5-Land validation %04d-%02d from CDS...", year, month)
        client = _init_cds_client()
        client.retrieve("reanalysis-era5-land", val_request, str(out_path_validation))
        logger.info(
            "Downloaded: %s (%.1f MB)",
            out_path_validation,
            out_path_validation.stat().st_size / 1e6,
        )

    return out_path


def fetch_era5_land_range(
    start: date,
    end: date,
    **kwargs,
) -> list[Path]:
    """Download ERA5-Land for a date range, month by month.

    Parameters
    ----------
    start, end : date range
    **kwargs : forwarded to fetch_era5_land_month
        (bounding_box, raw_dir, include_validation, overwrite)

    Returns list of paths to downloaded files.
    """
    paths = []
    current = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)

    while current <= end_month:
        path = fetch_era5_land_month(current.year, current.month, **kwargs)
        paths.append(path)
        # Advance to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return paths


def open_era5_land(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    start: date | None = None,
    end: date | None = None,
) -> xr.Dataset:
    """Open downloaded ERA5-Land files as a single lazy xarray Dataset.

    Uses dask for out-of-core access.  Call ``.load()`` or ``.compute()``
    to materialise into memory.

    Parameters
    ----------
    raw_dir : root raw data directory (contains era5_land/ subdirectory)
    start, end : optional date range filter
    """
    raw_dir = Path(raw_dir)
    era5_dir = raw_dir / "era5_land"
    if not era5_dir.exists():
        raise FileNotFoundError(
            f"No ERA5-Land data found at {era5_dir}. Run fetch_era5_land_range first."
        )

    # Match only forcing files, exclude validation files
    files = sorted(f for f in era5_dir.glob("era5land_*.nc") if "validation" not in f.name)
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
# ERA5 - Month 0.25
# ---------------------------------------------------------------------------


def _era5_month_output_path(raw_dir: Path, year_min: int, year_max: int) -> Path:
    """Consistent file naming: era5land_YYYY_MM.nc"""
    out_dir = raw_dir / "era5_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / f"era5land_{year_min:04d}_{year_max:04d}.nc"


def fetch_era5_month(
    year_min: int,
    year_max: int,
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> Path:
    """Download the history of ERA5 monthly averages.

    Parameters
    ----------
    year_min, year_max : target period
    raw_dir : output directory for ERA5 downloads

    Returns
    -------
    Path to the downloaded NetCDF file.
    """
    raw_dir = Path(raw_dir)
    out_path = _era5_month_output_path(raw_dir, year_min, year_max)


    request = {
        "variable": list(ERA5_MONTH_VARIABLES),
        "year": list(range(year_min, year_max+1)),
        "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
        "time": ["00:00"],
        "area": [60, -20, 40, 20],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

   

    logger.info("Requesting ERA5-Land %04d-%02d from CDS...", year_min, year_max)
    client = _init_cds_client()
    client.retrieve("reanalysis-era5-single-levels-monthly-means", request, str(out_path))
    logger.info("Downloaded: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    return out_path



# ---------------------------------------------------------------------------
# SEAS5
# ---------------------------------------------------------------------------


def _seas5_output_path(raw_dir: Path, year: int, month: int) -> Path:
    out_dir = raw_dir / "seas5"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"seas5_{year:04d}_{month:02d}.nc"


def fetch_seas5(
    year: int,
    month: int,
    *,
    bounding_box: BBoxWGS84 = FRANCE_BBOX,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> Path:
    """Download SEAS5 seasonal forecast initialised at year/month.

    Downloads all lead times (1-6 months) and all ensemble members.

    Parameters
    ----------
    year, month : initialisation date of the forecast
    bounding_box : spatial extent (default: France metropolitan)
    raw_dir : output directory
    overwrite : re-download even if file exists

    Returns
    -------
    Path to downloaded NetCDF.
    """
    raw_dir = Path(raw_dir)
    out_path = _seas5_output_path(raw_dir, year, month)

    if out_path.exists() and not overwrite:
        logger.info("SEAS5 %04d-%02d already exists: %s", year, month, out_path)
        return out_path

    leadtime_months = [1, 2, 3, 4, 5, 6]

    request = {
        "originating_centre": "ecmwf",
        "system": "51",
        "variable": SEAS5_VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "leadtime_month": [str(m) for m in leadtime_months],
        "area": bounding_box.as_cds_area(),
        "data_format": "netcdf",
    }

    logger.info("Requesting SEAS5 %04d-%02d from CDS...", year, month)
    client = _init_cds_client()
    client.retrieve("seasonal-monthly-single-levels", request, str(out_path))
    logger.info("Downloaded: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    return out_path





def fetch_seas5_hindcasts(
    init_month: int,
    *,
    start_year: int = 1993,
    end_year: int = 2016,
    bounding_box: BBoxWGS84 = FRANCE_BBOX,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Download the SEAS5 retrospective initializations for one calendar month.

    C3S serves 1993-2016 as hindcasts for the seasonal monthly dataset.  The
    resulting local archive is used to estimate the lead-dependent SEAS5 model
    climatology required for first-order bias correction.

    This is intentionally explicit rather than hidden inside the correction
    routine because it may download a substantial amount of data.
    """
    paths: list[Path] = []
    for year in range(start_year, end_year + 1):
        paths.append(
            fetch_seas5(
                year=year,
                month=init_month,
                bounding_box=bounding_box,
                raw_dir=raw_dir,
                overwrite=overwrite,
            )
        )
    return paths


def open_seas5(raw_dir: str | Path = DEFAULT_RAW_DIR, *, year: int, month: int) -> xr.Dataset:
    """Open a downloaded SEAS5 forecast file."""
    raw_dir = Path(raw_dir)
    path = _seas5_output_path(raw_dir, year, month)
    if not path.exists():
        raise FileNotFoundError(f"SEAS5 file not found: {path}. Run fetch_seas5 first.")
    return xr.open_dataset(path)
