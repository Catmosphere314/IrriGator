"""Météo-France public API client — GRIB opener and ARPEGE stubs.

This module provides:
- ``open_forecast()`` for opening downloaded GRIB2/NetCDF files
- ARPEGE forecast stubs (kept for future development)

Authentication uses OAuth via ``irrigator.utils.auth_meteofrance``.

AROME operations have moved to ``irrigator.forecasts.arome_processing``.
COMEPHORE operations have moved to ``irrigator.ingestion.comephore_client``.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import requests
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

_API_BASE = "https://public-api.meteofrance.fr/public"

_API_ENDPOINTS = {
    "arpege": {
        "base": f"{_API_BASE}/arpege/1.0",
        "coverage": "MF-NWP-GLOBAL-ARPEGE-01-FRANCE-WCS",
    },
}


# ---------------------------------------------------------------------------
# GRIB / NetCDF opener (used by arome_processing and ARPEGE)
# ---------------------------------------------------------------------------


def open_forecast(path: Path) -> xr.Dataset:
    """Open a downloaded forecast file.

    GRIB2 files require ``cfgrib`` engine.  Install via:
        pip install cfgrib eccodes
    """
    suffix = path.suffix.lower()
    if suffix in (".grib", ".grib2", ".grb", ".grb2"):
        return xr.open_dataset(
            path,
            engine="cfgrib",
            decode_timedelta=True,
            backend_kwargs={"indexpath": ""},
        )
    else:
        return xr.open_dataset(path)


# ---------------------------------------------------------------------------
# ARPEGE — global NWP (kept for future development)
# ---------------------------------------------------------------------------


def fetch_arpege(
    run_date: date,
    run_hour: int = 0,
    *,
    raw_dir: Path = Path("data/raw"),
    overwrite: bool = False,
) -> Path:
    """Download ARPEGE forecast for a given model run.

    Parameters
    ----------
    run_date : date of the model run
    run_hour : initialisation hour (0, 6, 12, 18)
    raw_dir : output directory
    overwrite : re-download if exists

    Returns
    -------
    Path to downloaded GRIB2 file.

    Notes
    -----
    ARPEGE is kept as a fallback for when IFS ENS is unavailable.
    For routine operation, IFS ENS (51 members, higher European skill)
    is preferred — see ``ingestion.ifs_ens_client``.
    """
    out_dir = raw_dir / "arpege"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"arpege_{run_date.isoformat()}_{run_hour:02d}z.grib2"

    if out_path.exists() and not overwrite:
        logger.debug("ARPEGE %s %02dZ exists", run_date, run_hour)
        return out_path

    from irrigator.config import BBoxWGS84
    from irrigator.utils.auth_meteofrance import meteo_headers

    bbox = BBoxWGS84(north=51.5, south=41.0, west=-6.0, east=10.0)

    session = requests.Session()
    session.headers.update(meteo_headers())
    endpoint_cfg = _API_ENDPOINTS["arpege"]

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
    logger.info("Fetching ARPEGE %s %02dZ", run_date, run_hour)

    resp = session.get(url, params=params, timeout=300)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)
    logger.info("Saved ARPEGE: %s (%.1f MB)", out_path, len(resp.content) / 1e6)
    return out_path


def fetch_latest_arpege(raw_dir: Path = Path("data/raw")) -> Path | None:
    """Fetch the most recent available ARPEGE run.

    Tries today 00Z, then yesterday 12Z, then yesterday 00Z.
    """
    today = date.today()
    yesterday = today - timedelta(days=1)

    for run_date, run_hour in [(today, 0), (yesterday, 12), (yesterday, 0)]:
        try:
            return fetch_arpege(run_date, run_hour, raw_dir=raw_dir)
        except Exception as exc:
            logger.debug("ARPEGE %s %02dZ not available: %s", run_date, run_hour, exc)
            continue

    logger.warning("Could not fetch any recent ARPEGE run")
    return None
