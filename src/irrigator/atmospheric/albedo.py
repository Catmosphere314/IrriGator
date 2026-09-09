"""Tools to download and store the Forecast Albedo from ERA5.

It is used for backtesting, because TIGGE IFS is not providing SSRD (Surface Solar Radiation Downwards)
but rather radiation and heat with already processed albedo."""
from __future__ import annotations

from pathlib import Path
import logging
from datetime import date, timedelta
from calendar import monthrange

import cdsapi
import xarray as xr

from irrigator.config import BBoxWGS84
from irrigator.ingestion.cds_client import (
    _init_cds_client,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_STATIC_DIR,
    FRANCE_BBOX,
)

logger = logging.getLogger(__name__)

ALBEDO_VAR = ["forecast_albedo"]


def _albedo_output_path(
    raw_dir: Path = DEFAULT_RAW_DIR, year_min: int = 2007, year_max: int = 2026,
) -> Path:
    """Consistent file naming: albedo_era5_YYYY_YYYY.nc"""
    out_dir = raw_dir / "era5_albedo"
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / f"albedo_era5_{year_min}_{year_max}.nc"


def fetch_era5_albedo(
    year_min: int = 2007,
    year_max: int = 2026,
    *,
    bounding_box: BBoxWGS84 = FRANCE_BBOX,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
    force_cds_refresh: bool = False,
) -> Path:
    """Download one month of hourly ERA5 albedo 00:00 data.

    Parameters
    ----------
    year_min, year_max : target period
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
    out_path = _albedo_output_path(raw_dir, year_min, year_max)

    if out_path.exists() and not overwrite:
        logger.info("ERA5 Albedo %04d-%04d already exists: %s", year_min, year_max, out_path)
        return out_path

    request = {
        "product_type": ["reanalysis"],
        "variable": list(ALBEDO_VAR),
        "year": [str(year) for year in range(year_min, year_max + 1)],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": ["00:00"],
        "area": bounding_box.as_cds_area(),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    if force_cds_refresh:
        request["nocache"] = "123"

    logger.info("Requesting ERA5-Land %04d-%04d from CDS...", year_min, year_max)
    client = _init_cds_client()
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
    logger.info("Downloaded: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    return out_path


def process_era5_forecast_albedo(
    year_range: tuple[int, int],
    raw_dir : Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    bounding_box: BBoxWGS84 = FRANCE_BBOX,
    overwrite: bool = False,
    force_refresh: bool = False,
) -> Path:
    """Create daily ERA5 forecast albedo."""

    input_path = _albedo_output_path(
        raw_dir=raw_dir, year_min=year_range[0], year_max=year_range[1]
    )
    output_path = _albedo_output_path(
        raw_dir=processed_dir, year_min=year_range[0], year_max=year_range[1]
    )
    if not input_path.exists() or force_refresh:
        fetch_era5_albedo(
            year_min=year_range[0],
            year_max=year_range[1],
            bounding_box=bounding_box,
            raw_dir=raw_dir,
            overwrite=overwrite or force_refresh,
            force_cds_refresh=force_refresh,
        )

    ds = xr.open_dataset(input_path)

    if "fal" not in ds:
        raise ValueError(
            f"'fal' not found in {input_path}. Available variables: {list(ds.data_vars)}"
        )

    fal = ds["fal"].rename("forecast_albedo").clip(0.0, 1.0)

    out = fal.to_dataset()

    out["forecast_albedo"].attrs.update(
        {
            "long_name": "ERA5-Land daily mean forecast albedo",
            "units": "1",
            "description": (
                "Daily ERA5 forecast albedo sampled at 00 UTC, "
                "used to reconstruct downward shortwave radiation "
                "from TIGGE surface net solar radiation."
            ),
        }
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_netcdf(
        output_path,
        encoding={
            "forecast_albedo": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            }
        },
    )

    ds.close()

    return output_path


