"""COMEPHORE radar-gauge precipitation client.

Downloads monthly COMEPHORE archives from data.gouv.fr. These are
France-wide 1 km hourly precipitation grids — no per-region
subsetting needed at download time.

The archives contain hourly NetCDF or GRIB files inside a TAR.
After download, extract and aggregate to daily totals for use in
precipitation bias correction (Block 2).

Data source
-----------
https://www.data.gouv.fr/fr/datasets/donnees-de-lames-deau-comephore/
Coverage: metropolitan France, 1997–present, hourly, 1 km resolution.
Access: open, no API key needed.

Notes
-----
COMEPHORE is the reference for precipitation bias correction of
ERA5-Land (CDF-t per grid cell). It is NOT used as direct forcing
for the water balance — ERA5-Land (corrected) provides the forcing.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from datetime import date, timedelta
from pathlib import Path

import requests
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR = Path("data/raw")

# ---------------------------------------------------------------------------
# data.gouv.fr API
# ---------------------------------------------------------------------------

COMEPHORE_DATASET_ID = "669e23a7ce052a9e8521b75e"
COMEPHORE_DATASET_API = f"https://www.data.gouv.fr/api/1/datasets/{COMEPHORE_DATASET_ID}/"


def _comephore_dir(raw_dir: Path) -> Path:
    out = raw_dir / "comephore"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _find_comephore_month_resource(
    session: requests.Session,
    target_date: date,
) -> dict:
    """Return the data.gouv.fr resource metadata for a COMEPHORE month."""
    response = session.get(COMEPHORE_DATASET_API, timeout=60)
    response.raise_for_status()
    metadata = response.json()

    expected_name = f"H_COMEPHORE_{target_date:%Y%m}"
    resources = metadata.get("resources", [])

    matches = [
        resource
        for resource in resources
        if resource.get("type") == "main"
        and resource.get("format", "").lower() == "tar"
        and expected_name in (resource.get("title", "") + " " + resource.get("url", ""))
    ]

    if not matches:
        raise FileNotFoundError(
            f"No COMEPHORE resource found for {target_date:%Y-%m}. "
            "The requested month may predate 1997 or may not have "
            "been published yet."
        )

    if len(matches) > 1:
        logger.warning(
            "Multiple COMEPHORE resources found for %s; using the first",
            target_date.strftime("%Y-%m"),
        )

    return matches[0]


def fetch_comephore_month(
    target_date: date,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> Path:
    """Download the monthly COMEPHORE TAR archive containing target_date.

    Parameters
    ----------
    target_date : any date within the target month
    raw_dir : root raw directory (default: data/raw)
    overwrite : re-download if archive exists

    Returns
    -------
    Path to the downloaded TAR archive.
    """
    out_dir = _comephore_dir(raw_dir)
    archive_dir = out_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    out_path = archive_dir / f"H_COMEPHORE_{target_date:%Y%m}.tar"

    if out_path.exists() and not overwrite:
        logger.debug("COMEPHORE archive exists: %s", out_path)
        return out_path

    session = requests.Session()
    resource = _find_comephore_month_resource(session, target_date)
    download_url = resource.get("latest") or resource["url"]

    temporary_path = out_path.with_suffix(".tar.part")
    temporary_path.unlink(missing_ok=True)

    logger.info(
        "Downloading COMEPHORE %s to %s",
        target_date.strftime("%Y-%m"),
        out_path,
    )

    try:
        with session.get(
            download_url,
            stream=True,
            timeout=(30, 1800),
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type or "application/json" in content_type:
                raise RuntimeError(f"Expected a TAR archive but received {content_type!r}")

            with temporary_path.open("wb") as destination:
                shutil.copyfileobj(response.raw, destination)

        if temporary_path.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Downloaded archive is unexpectedly small: {temporary_path.stat().st_size} bytes"
            )

        temporary_path.replace(out_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.info(
        "Saved COMEPHORE archive: %s (%.1f MB)",
        out_path,
        out_path.stat().st_size / 1e6,
    )
    return out_path


def fetch_comephore_range(
    start: date,
    end: date,
    **kwargs,
) -> list[Path]:
    """Download COMEPHORE archives for all months in a date range.

    Parameters
    ----------
    start, end : date range
    **kwargs : forwarded to fetch_comephore_month
        (raw_dir, overwrite)

    Returns list of paths to downloaded TAR archives.
    """
    paths = []
    current = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)

    while current <= end_month:
        try:
            path = fetch_comephore_month(current, **kwargs)
            paths.append(path)
        except FileNotFoundError as exc:
            logger.warning("COMEPHORE %s not available: %s", current.strftime("%Y-%m"), exc)
        except requests.HTTPError as exc:
            logger.warning("COMEPHORE %s failed: %s", current.strftime("%Y-%m"), exc)

        # Advance to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return paths


def extract_comephore_archive(
    archive_path: Path,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> Path:
    """Extract a COMEPHORE TAR archive to the raw directory.

    Parameters
    ----------
    archive_path : path to the .tar file
    raw_dir : root raw directory
    overwrite : re-extract if already done

    Returns
    -------
    Path to the extraction directory.
    """
    out_dir = _comephore_dir(raw_dir) / "hourly"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted (look for any .nc files for this month)
    month_tag = archive_path.stem  # e.g. H_COMEPHORE_202301
    marker = out_dir / f".{month_tag}_extracted"

    if marker.exists() and not overwrite:
        logger.debug("COMEPHORE %s already extracted", month_tag)
        return out_dir

    logger.info("Extracting %s", archive_path.name)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=out_dir, filter="data")

    marker.touch()
    logger.info("Extracted COMEPHORE to %s", out_dir)
    return out_dir


def open_comephore(
    raw_dir: Path = DEFAULT_RAW_DIR,
    start: date | None = None,
    end: date | None = None,
) -> xr.Dataset:
    """Open extracted COMEPHORE files as a single lazy Dataset.

    Parameters
    ----------
    raw_dir : root raw directory containing comephore/hourly/
    start, end : optional date range filter

    Returns
    -------
    xr.Dataset with hourly precipitation grids.
    """
    hourly_dir = _comephore_dir(raw_dir) / "hourly"
    if not hourly_dir.exists():
        raise FileNotFoundError(
            f"No extracted COMEPHORE data at {hourly_dir}. "
            "Run fetch_comephore_month + extract_comephore_archive first."
        )

    files = sorted(hourly_dir.glob("**/*.nc"))
    if not files:
        # Try GRIB files
        files = sorted(hourly_dir.glob("**/*.grib2")) + sorted(hourly_dir.glob("**/*.grib"))

    if not files:
        raise FileNotFoundError(f"No COMEPHORE data files in {hourly_dir}")

    logger.info("Opening %d COMEPHORE files with dask", len(files))
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 24})
    return ds
