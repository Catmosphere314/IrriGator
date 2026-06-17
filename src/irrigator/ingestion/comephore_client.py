"""COMEPHORE radar-gauge precipitation client.

Downloads monthly COMEPHORE archives from data.gouv.fr. These are
France-wide 1 km hourly precipitation grids — no per-region
subsetting needed at download time.

The archives contain hourly GeoTIFF files inside a TAR.
After download, extract and aggregate to daily totals for use in
precipitation bias correction (Block 2).

Data source
-----------
https://www.data.gouv.fr/fr/datasets/donnees-de-lames-deau-comephore/
Coverage: metropolitan France, 1997–present, hourly, 1 km resolution.
Access: open, no API key needed.

File naming
-----------
Each TAR contains files named ``YYYYMMDDHH_{RR,ERR,QUALIF}.gtif``:
- RR:     precipitation accumulation [1/100 mm]
- ERR:    estimation error [1/100 mm]
- QUALIF: quality index [0–100, higher is better]

All stored as uint16.  No-data sentinels: 65535 (RR, ERR), 255 (QUALIF).

Daily aggregation
-----------------
- precip_mm     = sum(RR_hourly) / 100              [mm/day]
- error_mm      = sqrt(sum(ERR_hourly²)) / 100      [mm/day, independence assumption]
- quality_frac  = mean(QUALIF_hourly) / 100          [0–1, fraction of max quality]
- n_hours_valid = count of hours with valid RR        [0–24]

The independence assumption for error propagation is a lower bound.
Systematic radar biases (beam geometry, ground clutter) persist across
hours, so the true daily error is larger.  quality_frac and n_hours_valid
let downstream code (CDF-t) downweight days with poor coverage.

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
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# COMEPHORE file conventions
# ---------------------------------------------------------------------------

PATTERNS = {
    "accum": "RR",
    "error": "ERR",
    "qualif": "QUALIF",
}

PATTERNS_NA = {"accum": 65535, "error": 65535, "qualif": 255}

# RR and ERR are stored in 1/100 mm; divide by this to get mm
UNIT_SCALE = 100.0

# QUALIF is 0–100; divide by this to get a 0–1 fraction
QUALIF_SCALE = 100.0

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


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


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
                f"Downloaded archive is unexpectedly small: "
                f"{temporary_path.stat().st_size} bytes"
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
    start, end : date range (inclusive, month granularity)
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

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return paths


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Open hourly GeoTIFFs
# ---------------------------------------------------------------------------


def open_comephore_hourly(
    raw_dir: Path = DEFAULT_RAW_DIR,
    var_type: str = "accum",
    start: date | None = None,
    end: date | None = None,
) -> xr.DataArray:
    """Open extracted COMEPHORE hourly GeoTIFF files as a DataArray.

    Parameters
    ----------
    raw_dir : root raw directory containing comephore/hourly/
    var_type : one of "accum" (RR), "error" (ERR), "qualif" (QUALIF)
    start, end : date range filter (inclusive)

    Returns
    -------
    xr.DataArray with dims (time, y, x), raw integer values masked
    for no-data. No unit conversion applied — use the processing
    functions for that.
    """
    import rioxarray  # noqa: F401 — registers the rio accessor

    hourly_dir = _comephore_dir(raw_dir) / "hourly"
    if not hourly_dir.exists():
        raise FileNotFoundError(
            f"No extracted COMEPHORE data at {hourly_dir}. "
            "Run fetch_comephore_month + extract_comephore_archive first."
        )

    try:
        pattern = PATTERNS[var_type]
    except KeyError:
        raise ValueError(f"var_type must be one of {tuple(PATTERNS)}, got {var_type!r}") from None

    # Build file list for the requested date range
    if start and end:
        list_dates = [
            (start + timedelta(days=i)).strftime("%Y%m%d")
            for i in range((end - start).days + 1)
        ]
        files = sorted(
            f
            for date_str in list_dates
            for f in hourly_dir.glob(f"**/{date_str}*_{pattern}.gtif")
        )
    else:
        files = sorted(hourly_dir.glob(f"**/*_{pattern}.gtif"))

    if not files:
        raise FileNotFoundError(
            f"No COMEPHORE {pattern} GeoTIFF files in {hourly_dir} "
            f"between {start} and {end}"
        )

    logger.info("Opening %d COMEPHORE %s hourly files", len(files), pattern)

    arrays = []
    for f in files:
        timestamp = datetime.strptime(f.name[:10], "%Y%m%d%H")
        arr = rioxarray.open_rasterio(
            f,
            chunks={"x": 512, "y": 512},
            masked=True,
            cache=False,
        )
        if arr.sizes["band"] != 1:
            raise ValueError(f"Expected one band in {f}, found {arr.sizes['band']}")
        arr = arr.squeeze("band", drop=True).expand_dims(time=[timestamp])
        arrays.append(arr)

    data = xr.concat(
        arrays,
        dim="time",
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="override",
    )

    na_val = PATTERNS_NA[var_type]
    data = (
        data.astype(np.float32)
        .where(data != na_val)
        .sortby("time")
        .chunk({"time": 24})
    )

    return data


# ---------------------------------------------------------------------------
# Daily aggregation with error propagation
# ---------------------------------------------------------------------------


def process_comephore_month_to_daily(
    year: int,
    month: int,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    overwrite: bool = False,
) -> Path:
    """Aggregate hourly COMEPHORE to daily for one month.

    Reads all three variable types (RR, ERR, QUALIF), aggregates to
    daily resolution, and saves a single compressed NetCDF.

    Error propagation
    -----------------
    Daily error = sqrt(sum(hourly_error²)) / UNIT_SCALE, assuming
    independence between hourly errors. This is a lower bound — the
    true daily error is larger due to persistent systematic biases
    (beam geometry, ground clutter, gauge undercatch).

    The quality_frac and n_hours_valid fields let downstream code
    (CDF-t) downweight or exclude days with poor radar coverage.

    Parameters
    ----------
    year, month : target month
    raw_dir : root raw directory containing comephore/hourly/
    processed_dir : output directory (default: data/processed)
    overwrite : overwrite existing output

    Returns
    -------
    Path to the saved NetCDF file.
    """
    out_dir = Path(processed_dir) / "comephore"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comephore_daily_{year:04d}{month:02d}.nc"

    if out_path.exists() and not overwrite:
        logger.info("COMEPHORE daily %04d-%02d already exists: %s", year, month, out_path)
        return out_path

    # Date range for this month
    first = date(year, month, 1)
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)

    logger.info("Processing COMEPHORE %04d-%02d to daily (%s → %s)", year, month, first, last)

    # --- Load hourly data for all three types ---
    accum = open_comephore_hourly(raw_dir, var_type="accum", start=first, end=last)
    error = open_comephore_hourly(raw_dir, var_type="error", start=first, end=last)

    try:
        qualif = open_comephore_hourly(raw_dir, var_type="qualif", start=first, end=last)
        has_qualif = True
    except FileNotFoundError:
        logger.warning("No QUALIF files for %04d-%02d, quality_frac will be based on valid hours only", year, month)
        has_qualif = False

    # --- Daily precipitation: sum(hourly) / 100 → mm/day ---
    precip_daily = (
        accum
        .resample(time="1D")
        .sum(skipna=True)
        / UNIT_SCALE
    ).clip(min=0)

    # --- Number of valid hours per day ---
    n_valid = (
        accum.notnull()
        .astype(np.int8)
        .resample(time="1D")
        .sum()
    )

    # --- Error propagation: sqrt(sum(σ²)) / 100 → mm/day ---
    # Under hourly independence assumption (lower bound on true error)
    error_sq_sum = (
        (error ** 2)
        .resample(time="1D")
        .sum(skipna=True)
    )
    error_daily = np.sqrt(error_sq_sum) / UNIT_SCALE

    # --- Quality: mean(qualif) / 100 → fraction [0-1] ---
    if has_qualif:
        quality_daily = (
            qualif
            .resample(time="1D")
            .mean(skipna=True)
            / QUALIF_SCALE
        )
    else:
        # Fall back to fraction of valid hours
        quality_daily = n_valid.astype(np.float32) / 24.0

    # --- Assemble dataset ---
    result = xr.Dataset(
        {
            "precip_mm": precip_daily.rename("precip_mm"),
            "precip_error_mm": error_daily.rename("precip_error_mm"),
            "quality_frac": quality_daily.rename("quality_frac"),
            "n_hours_valid": n_valid.rename("n_hours_valid"),
        }
    )

    result.attrs.update({
        "source": "COMEPHORE (Météo-France) — daily aggregation",
        "precip_units": "mm/day",
        "error_units": "mm/day (sqrt of sum of squared hourly errors)",
        "error_note": (
            "Lower bound assuming hourly independence. "
            "Systematic radar biases make the true daily error larger."
        ),
        "quality_note": (
            "Mean hourly quality index scaled to [0, 1]. "
            "Values below ~0.5 indicate poor radar coverage."
        ),
        "spatial_resolution": "1 km",
        "year": year,
        "month": month,
    })

    # --- Save with compression ---
    encoding = {
        "precip_mm": {"dtype": "float32", "zlib": True, "complevel": 4},
        "precip_error_mm": {"dtype": "float32", "zlib": True, "complevel": 4},
        "quality_frac": {"dtype": "float32", "zlib": True, "complevel": 4},
        "n_hours_valid": {"dtype": "int8", "zlib": True, "complevel": 4},
    }

    result.compute().to_netcdf(out_path, encoding=encoding)

    size_mb = out_path.stat().st_size / 1e6
    n_days = len(result.time)
    logger.info(
        "Saved COMEPHORE daily %04d-%02d: %s (%.1f MB, %d days)",
        year, month, out_path, size_mb, n_days,
    )
    return out_path


# ---------------------------------------------------------------------------
# Batch pipeline: fetch → extract → process for a date range
# ---------------------------------------------------------------------------


def run_comephore_pipeline(
    start: date,
    end: date,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    overwrite_download: bool = False,
    overwrite_daily: bool = False,
    cleanup_hourly: bool = False,
) -> list[Path]:
    """Full COMEPHORE pipeline: download → extract → daily aggregation.

    Processes month by month so intermediate hourly files can be cleaned
    up progressively (set cleanup_hourly=True to delete extracted GeoTIFFs
    after each month is aggregated — the TAR archives are kept).

    Parameters
    ----------
    start, end : date range (inclusive, month granularity)
    raw_dir : root raw directory for downloads and extraction
    processed_dir : output directory for daily NetCDF files
    overwrite_download : re-download TAR archives
    overwrite_daily : re-process even if daily file exists
    cleanup_hourly : delete extracted hourly GeoTIFFs after daily
        aggregation (TARs are kept on disk for reproducibility)

    Returns
    -------
    List of paths to daily NetCDF files.
    """
    daily_paths = []
    current = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)
    hourly_dir = _comephore_dir(raw_dir) / "hourly"

    while current <= end_month:
        year, month = current.year, current.month
        tag = f"{year:04d}-{month:02d}"

        try:
            # Step 1: Download
            logger.info("[%s] Step 1/3: Downloading...", tag)
            archive_path = fetch_comephore_month(
                current, raw_dir=raw_dir, overwrite=overwrite_download,
            )

            # Step 2: Extract
            logger.info("[%s] Step 2/3: Extracting...", tag)
            extract_comephore_archive(
                archive_path, raw_dir=raw_dir, overwrite=overwrite_download,
            )

            # Step 3: Aggregate to daily
            logger.info("[%s] Step 3/3: Aggregating to daily...", tag)
            daily_path = process_comephore_month_to_daily(
                year, month,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                overwrite=overwrite_daily,
            )
            daily_paths.append(daily_path)

            # Optional: clean up hourly GeoTIFFs for this month
            if cleanup_hourly and hourly_dir.exists():
                month_str = f"{year:04d}{month:02d}"
                hourly_files = list(hourly_dir.glob(f"**/{month_str}*"))
                if hourly_files:
                    for f in hourly_files:
                        f.unlink(missing_ok=True)
                    logger.info(
                        "[%s] Cleaned up %d hourly files",
                        tag, len(hourly_files),
                    )

        except Exception as exc:
            logger.error("[%s] Failed: %s", tag, exc)

        # Advance to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    logger.info(
        "COMEPHORE pipeline complete: %d monthly files produced",
        len(daily_paths),
    )
    return daily_paths


# ---------------------------------------------------------------------------
# Load daily COMEPHORE (for downstream use)
# ---------------------------------------------------------------------------


def load_comephore_daily(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    start: date | None = None,
    end: date | None = None,
) -> xr.Dataset:
    """Open daily COMEPHORE files as a single lazy Dataset.

    Parameters
    ----------
    processed_dir : directory containing comephore/comephore_daily_YYYYMM.nc
    start, end : optional date range filter

    Returns
    -------
    xr.Dataset with variables: precip_mm, precip_error_mm,
    quality_frac, n_hours_valid.
    """
    comephore_dir = Path(processed_dir) / "comephore"
    if not comephore_dir.exists():
        raise FileNotFoundError(
            f"No daily COMEPHORE data at {comephore_dir}. "
            "Run run_comephore_pipeline first."
        )

    files = sorted(comephore_dir.glob("comephore_daily_*.nc"))
    if not files:
        raise FileNotFoundError(f"No daily COMEPHORE NetCDF files in {comephore_dir}")

    # Filter by date range
    if start or end:
        filtered = []
        for f in files:
            # Parse YYYYMM from filename
            ym = f.stem.replace("comephore_daily_", "")
            file_year, file_month = int(ym[:4]), int(ym[4:6])
            file_date = date(file_year, file_month, 1)
            if start and file_date < date(start.year, start.month, 1):
                continue
            if end and file_date > date(end.year, end.month, 1):
                continue
            filtered.append(f)
        files = filtered

    if not files:
        raise FileNotFoundError(
            f"No daily COMEPHORE files match {start} → {end}"
        )

    logger.info("Opening %d daily COMEPHORE files", len(files))
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 31})
    return ds
