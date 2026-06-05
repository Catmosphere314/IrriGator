"""Météo-France public API client for forecast and radar-gauge data.

Prerequisites
-------------
- Register at https://portail-api.meteofrance.fr/ and obtain an API key.
- Set the environment variable ``METEOFRANCE_API_KEY`` or pass it explicitly.

Data products
-------------
- **COMEPHORE**: radar-gauge merged precipitation, 1 km, hourly.
  Used for precipitation bias correction of ERA5-Land (Block 2).
- **AROME**: high-resolution NWP, 1.3 km, up to 48 h.
  Used for short-term irrigation scheduling (Block 5).
- **ARPEGE**: global NWP, ~10 km, up to 96 h.
  Used for medium-range forecast extension (Block 5).

Notes
-----
The Météo-France API reorganised in 2024-2025.  The endpoints and
authentication flow below follow the "Données Publiques" portal.
If the API structure changes, update ``_API_ENDPOINTS`` and the
request-building logic — the rest of the module is stable.

COMEPHORE historical archives may require a separate bulk-download
procedure (HTTPS or FTP).  This module handles the API-based access
for recent/rolling data.  For the full 2010-2023 calibration archive,
check https://donneespubliques.meteofrance.fr/ for direct file access.

Note that COMEPHORE data has been downloaded directly, so no need to use API.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path

import requests
import xarray as xr

from irrigator.config import RegionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

_API_BASE = "https://public-api.meteofrance.fr/public"

# Endpoint patterns — these may change; centralise them here.
_API_ENDPOINTS = {
    "arome": {
        # AROME forecast packages (GRIB2 via WCS or direct download)
        "base": f"{_API_BASE}/arome/1.0",
        "coverage": "MF-NWP-HIGHRES-AROME-001-FRANCE-WCS",
    },
    "arpege": {
        "base": f"{_API_BASE}/arpege/1.0",
        "coverage": "MF-NWP-GLOBAL-ARPEGE-01-FRANCE-WCS",
    },
    "comephore": {
        "base": f"{_API_BASE}/DPClim/v1",
    },
}

# Variables we need, mapped to Météo-France parameter names
# These may differ between AROME and ARPEGE; adjust as needed.
_FORECAST_PARAMS = {
    "temperature_2m": "Temperature at 2m",
    "dewpoint_2m": "Dew point temperature at 2m",
    "wind_u_10m": "U component of wind at 10m",
    "wind_v_10m": "V component of wind at 10m",
    "precipitation": "Total precipitation",
    "solar_radiation": "Surface downwelling shortwave flux",
}


def _get_api_key() -> str:
    """Retrieve Météo-France API key from environment."""
    key = os.environ.get("METEOFRANCE_API_KEY")
    if not key:
        raise RuntimeError(
            "METEOFRANCE_API_KEY environment variable not set. "
            "Register at https://portail-api.meteofrance.fr/"
        )
    return key


def _api_session(api_key: str | None = None) -> requests.Session:
    """Build a requests session with auth headers."""
    session = requests.Session()
    if api_key is None:
        session.headers.update(
            {
                "apikey": _get_api_key(),
                "Accept": "application/json",
            }
        )
    else:
        session.headers.update(
            {
                "apikey": api_key,
                "Accept": "application/json",
            }
        )
    return session


# ---------------------------------------------------------------------------
# COMEPHORE — radar-gauge precipitation
# ---------------------------------------------------------------------------


def _comephore_output_dir(raw_dir: Path) -> Path:
    out = raw_dir / "comephore"
    out.mkdir(parents=True, exist_ok=True)
    return out


def fetch_comephore_daily(
    cfg: RegionConfig,
    target_date: date,
    *,
    overwrite: bool = False,
) -> Path:
    """Download COMEPHORE hourly precipitation for one day.

    Stores as NetCDF clipped to the region bounding box.

    Parameters
    ----------
    cfg : RegionConfig
    target_date : the day to fetch
    overwrite : re-download if file exists

    Returns
    -------
    Path to the daily NetCDF file.

    Notes
    -----
    COMEPHORE access via the public API may be limited to recent months.
    For the full historical archive (2010-2023), use the bulk download
    from donneespubliques.meteofrance.fr — this function handles the
    API-based rolling access.
    """
    out_dir = _comephore_output_dir(cfg.raw_dir)
    out_path = out_dir / f"comephore_{target_date.isoformat()}.nc"

    if out_path.exists() and not overwrite:
        logger.debug("COMEPHORE %s exists: %s", target_date, out_path)
        return out_path

    session = _api_session()
    bbox = cfg.bbox_wgs84

    # The exact endpoint and parameters depend on the current API version.
    # This is a template — adapt once you have API access and can inspect
    # the actual response format.
    params = {
        "product": "COMEPHORE",
        "date": target_date.isoformat(),
        "bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}",
        "format": "netcdf",
    }

    url = f"{_API_ENDPOINTS['comephore']['base']}/gridded/comephore"
    logger.info("Fetching COMEPHORE for %s", target_date)

    resp = session.get(url, params=params, timeout=120)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    logger.info("Saved COMEPHORE: %s (%.1f KB)", out_path, len(resp.content) / 1e3)
    return out_path


def fetch_comephore_range(
    cfg: RegionConfig,
    start: date,
    end: date,
    **kwargs,
) -> list[Path]:
    """Download COMEPHORE for a date range, day by day."""
    paths = []
    current = start
    while current <= end:
        try:
            path = fetch_comephore_daily(cfg, current, **kwargs)
            paths.append(path)
        except requests.HTTPError as exc:
            logger.warning("COMEPHORE %s failed: %s", current, exc)
        current += timedelta(days=1)
    return paths


def open_comephore(cfg: RegionConfig, start: date, end: date) -> xr.Dataset:
    """Open downloaded COMEPHORE files as a single Dataset."""
    com_dir = _comephore_output_dir(cfg.raw_dir)
    files = sorted(com_dir.glob("comephore_*.nc"))

    # Filter to date range
    filtered = []
    for f in files:
        file_date = date.fromisoformat(f.stem.replace("comephore_", ""))
        if start <= file_date <= end:
            filtered.append(f)

    if not filtered:
        raise FileNotFoundError(f"No COMEPHORE files found in {com_dir} for {start} to {end}")

    return xr.open_mfdataset(filtered, combine="by_coords")


# ---------------------------------------------------------------------------
# AROME / ARPEGE — NWP forecasts
# ---------------------------------------------------------------------------


def _forecast_output_dir(raw_dir: Path, model: str) -> Path:
    out = raw_dir / model.lower()
    out.mkdir(parents=True, exist_ok=True)
    return out


def fetch_forecast(
    cfg: RegionConfig,
    model: str,
    run_date: date,
    run_hour: int = 0,
    *,
    overwrite: bool = False,
) -> Path:
    """Download AROME or ARPEGE forecast for a given model run.

    Parameters
    ----------
    cfg : RegionConfig
    model : "arome" or "arpege"
    run_date : date of the model run
    run_hour : initialisation hour (0, 6, 12, 18 for ARPEGE; 0, 3, 6, ... for AROME)
    overwrite : re-download if exists

    Returns
    -------
    Path to downloaded file (GRIB2 or NetCDF depending on API response).
    """
    model = model.lower()
    if model not in ("arome", "arpege"):
        raise ValueError(f"Unknown model: {model}. Use 'arome' or 'arpege'.")

    out_dir = _forecast_output_dir(cfg.raw_dir, model)
    out_path = out_dir / f"{model}_{run_date.isoformat()}_{run_hour:02d}z.grib2"

    if out_path.exists() and not overwrite:
        logger.debug("%s %s %02dZ exists", model.upper(), run_date, run_hour)
        return out_path

    model_cfg = cfg.data.get(model, {})
    horizon_h = model_cfg.get("forecast_horizon_h", 48 if model == "arome" else 96)
    bbox = cfg.bbox_wgs84

    session = _api_session()
    endpoint_cfg = _API_ENDPOINTS[model]

    # Build WCS-style request
    # NOTE: The exact parameter names and endpoint paths depend on the
    # Météo-France API version at time of use. This is a template.
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": endpoint_cfg.get("coverage", ""),
        "subset": [
            f"time({run_date.isoformat()}T{run_hour:02d}:00:00Z)",
            f"lat({bbox.south},{bbox.north})",
            f"long({bbox.west},{bbox.east})",
        ],
        "format": "application/grib2",
    }

    url = f"{endpoint_cfg['base']}/wcs/{endpoint_cfg.get('coverage', '')}"
    logger.info("Fetching %s %s %02dZ (horizon %dh)", model.upper(), run_date, run_hour, horizon_h)

    resp = session.get(url, params=params, timeout=300)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    logger.info("Saved %s: %s (%.1f MB)", model.upper(), out_path, len(resp.content) / 1e6)
    return out_path


def fetch_latest_forecasts(cfg: RegionConfig) -> dict[str, Path]:
    """Fetch latest available AROME and ARPEGE runs.

    Tries the most recent 00Z run for today.  Falls back to yesterday's
    12Z if today's isn't available yet (AROME typically available ~3h after
    init time).

    Returns
    -------
    Dict with keys "arome" and "arpege", values are file paths.
    """
    today = date.today()
    yesterday = today - timedelta(days=1)
    results = {}

    for model in ("arome", "arpege"):
        for run_date, run_hour in [(today, 0), (yesterday, 12), (yesterday, 0)]:
            try:
                path = fetch_forecast(cfg, model, run_date, run_hour)
                results[model] = path
                break
            except requests.HTTPError:
                continue
        else:
            logger.warning("Could not fetch any recent %s run", model.upper())

    return results


def fetch_arome_analysis_daily(cfg, target_date, run_hour=0):
    """Fetch AROME 00Z run for one day (steps 0-23h).

    Used to bridge the ERA5-Land ~5-day latency gap.
    Returns path to downloaded GRIB, ready for daily processing.

    NEED TO CHECK HOW TO JUST SELECT THE FIRST 00-23H!!!!!!!!!!!
    """
    return fetch_forecast(cfg, "arome", target_date, run_hour)


def open_forecast(path: Path) -> xr.Dataset:
    """Open a downloaded forecast file.

    GRIB2 files require ``cfgrib`` engine.  Install via:
        pip install cfgrib eccodes
    """
    suffix = path.suffix.lower()
    if suffix in (".grib", ".grib2", ".grb", ".grb2"):
        return xr.open_dataset(path, engine="cfgrib")
    else:
        return xr.open_dataset(path)
