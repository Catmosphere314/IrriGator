"""ECMWF IFS ENS client for medium-range ensemble forecasts (0-15 days).

The IFS ensemble (ENS) provides 51 members at 0.25° resolution, updated
every 6 hours.  Free under CC-BY-4.0 since October 2025.

Why IFS ENS over ARPEGE for 48-96h:
- Higher skill: IFS consistently outperforms ARPEGE for European weather
  at all lead times.  ECMWF is the benchmark that national met services
  (including Météo-France) evaluate against.
- Ensemble spread: 51 members give probabilistic uncertainty, vs ARPEGE
  deterministic which gives one scenario. For irrigation decisions,
  knowing "rain is likely (40/51 members show >5mm)" is more useful
  than "the model says 8mm."
- Consistency: same model family as SEAS5 — one framework from day 1
  to month 6.
- Resolution: 0.25° ≈ 28 km, comparable to ARPEGE's ~10 km but with
  the ensemble advantage.  For irrigation at parcel scale, both get
  downscaled through extract_parcel_forcing anyway.

Dependencies
------------
pip install ecmwf-opendata cfgrib eccodes

Access
------
No API key needed.  Data available from ECMWF open data servers, AWS,
Azure, or via Open-Meteo.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from irrigator.config import BBoxWGS84

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# France metropolitan bounding box (matches cds_client.FRANCE_BBOX)
# ---------------------------------------------------------------------------

FRANCE_BBOX = BBoxWGS84(north=51.5, south=41.0, west=-6.0, east=10.0)

# ---------------------------------------------------------------------------
# Default paths — relative to repo root (assumes cwd = repo root)
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR = Path("data/raw")

# Variables we need for the irrigation pipeline
# param codes: https://apps.ecmwf.int/codes/grib/param-db
IFS_ENS_PARAMS = [
    "2t",  # 2m temperature [K]
    "2d",  # 2m dewpoint [K]
    "10u",  # 10m u-wind [m/s]
    "10v",  # 10m v-wind [m/s]
    "sp",  # surface pressure [Pa]
    "tp",  # total precipitation [m]
    "ssrd",  # surface solar rad downward [J/m²]
]

CFGRIB_VAR_NAMES = {
    "2t": "t2m",
    "2d": "d2m",
    "10u": "u10",
    "10v": "v10",
    "sp": "sp",
    "tp": "tp",
    "ssrd": "ssrd",
    "ssr": "ssr",
    "str": "str",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _output_dir(raw_dir: Path) -> Path:
    out = raw_dir / "ifs_ens"
    out.mkdir(parents=True, exist_ok=True)
    return out


def fetch_ifs_ens(
    raw_dir: Path = DEFAULT_RAW_DIR,
    run_date: date | None = None,
    run_hour: int = 0,
    steps: list[int] | None = None,
    overwrite: bool = False,
    download_mode: str = "step",
    max_workers: int = 3,
    source: str = "ecmwf",
) -> Path:
    """Download IFS ENS forecast from ECMWF open data.

    Parameters
    ----------
    raw_dir : output directory (default: data/raw)
    run_date : forecast initialization date (default: today)
    run_hour : initialization hour (0 or 12)
    steps : forecast hours to retrieve (default: 0 to 360 by 6)
    overwrite : re-download if exists
    download_mode : download strategy:
        - "step" (default): one request per step (~260 MB each).
          Fastest on ECMWF direct — each step is a single file on their
          servers, no byte-range extraction needed.
        - "param": one request per variable (~2.3 GB each).
          Better for AWS/Azure when ECMWF is unavailable.
        - "bulk": single request for everything (~16 GB).
          Fastest on fast connections but fragile.
    max_workers : parallel downloads (default: 3). Only used in
        "step" and "param" modes.
    source : ECMWF open data source ("ecmwf", "aws", "azure").
        Default "ecmwf" for step mode (fastest), "azure" may be
        better for param mode.

    Returns
    -------
    Path to downloaded GRIB2 file.
    """
    try:
        from ecmwf.opendata import Client
    except ImportError:
        raise ImportError(
            "Install ecmwf-opendata: pip install ecmwf-opendata\n"
            "No API key needed — data is open access."
        )

    if run_date is None:
        run_date = date.today()

    if steps is None:
        steps = list(range(0, 366, 6))

    out_dir = _output_dir(raw_dir)
    out_path = out_dir / f"ifs_ens_{run_date.isoformat()}_{run_hour:02d}z.grib2"

    if out_path.exists() and not overwrite:
        logger.info("IFS ENS already exists: %s", out_path)
        return out_path

    if download_mode == "step":
        _fetch_by_step(
            out_dir, out_path, run_date, run_hour, steps, max_workers=max_workers, source=source
        )
    elif download_mode == "param":
        _fetch_by_param(
            out_dir, out_path, run_date, run_hour, steps, max_workers=max_workers, source=source
        )
    else:
        _fetch_bulk(out_path, run_date, run_hour, steps, source=source)

    logger.info(
        "Downloaded IFS ENS: %s (%.1f MB)",
        out_path,
        out_path.stat().st_size / 1e6,
    )
    return out_path


def _combine_parts(part_files: list[Path], out_path: Path) -> None:
    """Concatenate GRIB part files into a single combined file."""
    logger.info("Combining %d parts into %s...", len(part_files), out_path.name)
    with open(out_path, "wb") as combined:
        for part_path in part_files:
            with open(part_path, "rb") as part:
                while True:
                    chunk = part.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    combined.write(chunk)

    for part_path in part_files:
        part_path.unlink(missing_ok=True)

    logger.info("Combined: %.1f MB", out_path.stat().st_size / 1e6)


def _download_with_fallback(
    param: list[str],
    run_date: date,
    run_hour: int,
    step: list[int],
    out_path: Path,
    source: str,
    label: str = "",
) -> Path:
    """Download one GRIB chunk with source fallback."""
    import time as _time
    from ecmwf.opendata import Client

    sources = [source] + [s for s in ("ecmwf", "aws", "azure") if s != source]

    for src in sources:
        try:
            client = Client(source=src)
            client.retrieve(
                type="ef",
                date=run_date.isoformat(),
                time=run_hour,
                param=param,
                step=step,
                target=str(out_path),
            )
            return out_path
        except Exception as exc:
            logger.warning("%s failed from %s: %s", label, src, exc)
            out_path.unlink(missing_ok=True)
            if src != sources[-1]:
                _time.sleep(5)

    raise RuntimeError(f"{label}: failed from all sources")


def _fetch_by_step(
    out_dir: Path,
    out_path: Path,
    run_date: date,
    run_hour: int,
    steps: list[int],
    max_workers: int = 3,
    source: str = "ecmwf",
) -> None:
    """Download one step at a time, then concatenate.

    Each step is a single ~260 MB file on ECMWF servers — no byte-range
    extraction, no server-side stitching. Fastest and most reliable,
    especially from ECMWF direct.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(
        "Downloading IFS ENS (by step, %d workers, source=%s): %s %02dZ, %d steps",
        max_workers,
        source,
        run_date,
        run_hour,
        len(steps),
    )

    def _download_one_step(step_h: int, index: int) -> Path:
        part_path = out_dir / f"_step_{run_date.isoformat()}_{run_hour:02d}z_{step_h:03d}h.grib2"

        if part_path.exists() and part_path.stat().st_size > 100_000:
            logger.debug("[%d/%d] step %dh — cached", index, len(steps), step_h)
            return part_path

        label = f"[{index}/{len(steps)}] step {step_h}h"
        _download_with_fallback(
            param=IFS_ENS_PARAMS,
            run_date=run_date,
            run_hour=run_hour,
            step=[step_h],
            out_path=part_path,
            source=source,
            label=label,
        )
        logger.info("%s done (%.0f MB)", label, part_path.stat().st_size / 1e6)
        return part_path

    part_files = [None] * len(steps)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one_step, step_h, i + 1): i for i, step_h in enumerate(steps)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                part_files[idx] = future.result()
            except Exception as exc:
                for pf in part_files:
                    if pf is not None:
                        pf.unlink(missing_ok=True)
                raise RuntimeError(f"Step {steps[idx]}h failed: {exc}") from exc

    _combine_parts(part_files, out_path)


def _fetch_by_param(
    out_dir: Path,
    out_path: Path,
    run_date: date,
    run_hour: int,
    steps: list[int],
    max_workers: int = 3,
    source: str = "azure",
) -> None:
    """Download one parameter at a time, then concatenate.

    Each parameter is ~2.3 GB (all steps, all members for one variable).
    Uses byte-range requests on S3/Azure.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(
        "Downloading IFS ENS (by param, %d workers, source=%s): %s %02dZ, %d params",
        max_workers,
        source,
        run_date,
        run_hour,
        len(IFS_ENS_PARAMS),
    )

    def _download_one_param(param: str, index: int) -> Path:
        part_path = out_dir / f"_part_{run_date.isoformat()}_{run_hour:02d}z_{param}.grib2"

        if part_path.exists() and part_path.stat().st_size > 1_000_000:
            logger.info(
                "[%d/%d] %s — cached (%.0f MB)",
                index,
                len(IFS_ENS_PARAMS),
                param,
                part_path.stat().st_size / 1e6,
            )
            return part_path

        label = f"[{index}/{len(IFS_ENS_PARAMS)}] {param}"
        _download_with_fallback(
            param=[param],
            run_date=run_date,
            run_hour=run_hour,
            step=steps,
            out_path=part_path,
            source=source,
            label=label,
        )
        logger.info("%s done (%.0f MB)", label, part_path.stat().st_size / 1e6)
        return part_path

    part_files = [None] * len(IFS_ENS_PARAMS)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one_param, param, i + 1): i
            for i, param in enumerate(IFS_ENS_PARAMS)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                part_files[idx] = future.result()
            except Exception as exc:
                for pf in part_files:
                    if pf is not None:
                        pf.unlink(missing_ok=True)
                raise RuntimeError(f"Param {IFS_ENS_PARAMS[idx]} failed: {exc}") from exc

    _combine_parts(part_files, out_path)


def _fetch_bulk(
    out_path: Path,
    run_date: date,
    run_hour: int,
    steps: list[int],
    source: str = "ecmwf",
) -> None:
    """Download all parameters in a single request (~16 GB)."""
    logger.info(
        "Downloading IFS ENS (bulk, source=%s): %s %02dZ, %d steps, %d params",
        source,
        run_date,
        run_hour,
        len(steps),
        len(IFS_ENS_PARAMS),
    )
    _download_with_fallback(
        param=IFS_ENS_PARAMS,
        run_date=run_date,
        run_hour=run_hour,
        step=steps,
        out_path=out_path,
        source=source,
        label="bulk",
    )


def fetch_latest_ifs_ens(raw_dir: Path = DEFAULT_RAW_DIR) -> Path | None:
    """Fetch the most recent available IFS ENS run.

    Tries today 00Z, then yesterday 12Z, then yesterday 00Z.
    """
    today = date.today()
    yesterday = today - timedelta(days=1)

    for run_date, run_hour in [(today, 0), (yesterday, 12), (yesterday, 0)]:
        try:
            return fetch_ifs_ens(raw_dir, run_date, run_hour)
        except Exception as exc:
            logger.debug("IFS ENS %s %02dZ not available: %s", run_date, run_hour, exc)
            continue

    logger.warning("Could not fetch any recent IFS ENS run")
    return None


# ---------------------------------------------------------------------------
# Open and process to daily xr.Dataset
# ---------------------------------------------------------------------------


def _open_cfgrib_group(path: Path, filter_by_keys: dict, bbox: BBoxWGS84) -> xr.Dataset | None:
    try:
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": filter_by_keys,
                # Important while debugging / overwriting GRIB files.
                # Prevents stale .idx files from being reused.
                "indexpath": "",
            },
            decode_timedelta=True,
        )

        # Add buffer for interpolation
        north = bbox.north + 1
        south = bbox.south - 1
        west = bbox.west - 1
        east = bbox.east + 1

        # ECMWF latitude is usually descending
        ds = ds.sel(latitude=slice(north, south))

        # If longitude is 0..360
        if float(ds.longitude.max()) > 180:
            west_360 = west % 360
            east_360 = east % 360

            if west_360 <= east_360:
                ds = ds.sel(longitude=slice(west_360, east_360))
            else:
                ds = xr.concat(
                    [
                        ds.sel(longitude=slice(west_360, 360)),
                        ds.sel(longitude=slice(0, east_360)),
                    ],
                    dim="longitude",
                )
        else:
            ds = ds.sel(longitude=slice(west, east))
        return ds
    except Exception:
        return None


def open_ifs_ens(path: Path, bbox: BBoxWGS84 = FRANCE_BBOX) -> xr.Dataset:
    """Open selected IFS ENS GRIB2 variables as one xarray Dataset.

    The GRIB is downloaded globally; this slices to the bounding box
    on read to keep memory manageable.

    Parameters
    ----------
    path : GRIB2 file from fetch_ifs_ens
    bbox : spatial extent for slicing (default: France metropolitan)

    Returns a merged dataset, typically with dims:
      number, time, step, latitude, longitude

    Notes
    -----
    cfgrib cannot always open 2 m, 10 m, and surface variables in one pass
    because their scalar GRIB coordinates differ, e.g. heightAboveGround=2
    versus heightAboveGround=10.
    """

    groups = [
        # 2 m fields: 2t, 2d
        {
            "typeOfLevel": "heightAboveGround",
            "level": 2,
        },
        # 10 m fields: 10u, 10v
        {
            "typeOfLevel": "heightAboveGround",
            "level": 10,
        },
        # Surface instantaneous fields, e.g. sp
        {
            "typeOfLevel": "surface",
            "stepType": "instant",
        },
        # Surface accumulated fields, e.g. tp, ssrd
        {
            "typeOfLevel": "surface",
            "stepType": "accum",
        },
    ]

    datasets: list[xr.Dataset] = []

    for filter_by_keys in groups:
        ds = _open_cfgrib_group(
            path=path,
            filter_by_keys=filter_by_keys,
            bbox=bbox,
        )

        if ds is None:
            continue

        # Keep only variables we care about, if present in this group.
        wanted = set(CFGRIB_VAR_NAMES.values())
        present = [v for v in ds.data_vars if v in wanted]

        if not present:
            continue

        ds = ds[present]

        # Drop scalar GRIB coords that prevent merging:
        # heightAboveGround=2 vs heightAboveGround=10, etc.
        ds = ds.drop_vars(
            [
                "heightAboveGround",
                "surface",
                "stepType",
            ],
            errors="ignore",
        )

        datasets.append(ds)

    if not datasets:
        raise RuntimeError(f"Could not open requested IFS ENS variables from: {path}")

    ds = xr.merge(
        datasets,
        compat="override",
        join="outer",
        combine_attrs="override",
    )

    # Optional: warn if something is missing.
    required = {
        "t2m",
        "d2m",
        "u10",
        "v10",
        "sp",
        "tp",
    }

    missing = required - set(ds.data_vars)

    if missing:
        raise RuntimeError(
            f"Opened file but missing variables {missing}. "
            f"Available variables: {sorted(ds.data_vars)}"
        )
    if "ssrd" not in ds and "ssr" not in ds:
        raise RuntimeError(
            "IFS dataset has neither ssrd nor ssr radiation."
        )

    return ds

def _precip_to_mm_factor(da: xr.DataArray) -> float:
    units = str(da.attrs.get("units", "")).lower()

    if units in {"m", "m of water equivalent"}:
        return 1000.0

    if "kg m**-2" in units or "kg m^-2" in units or "kg m-2" in units or units == "mm":
        return 1.0

    raise ValueError(f"Unknown precipitation units: {da.attrs.get('units')!r}")

def process_ifs_ens_to_daily(ds: xr.Dataset, shift_utc: int = 0) -> xr.Dataset:
    """Convert IFS ENS hourly/6-hourly data to daily format.

    Handles step-accumulation for tp and ssrd (same convention as ERA5-Land).
    Output matches ERA5-Land daily variable names for compatibility with
    extract_parcel_forcing().

    Returns dataset with dims (valid_time, number, latitude, longitude).
    """
    # Compute valid_time from forecast reference time + step
    if "valid_time" not in ds.dims and "step" in ds.dims:
        ref_time = ds.time if "time" in ds.coords else ds.coords.get("forecast_reference_time")
        if ref_time is not None:
            ds = ds.assign_coords(valid_time=ref_time + ds.step).swap_dims({"step": "valid_time"})

    ds = ds.assign_coords(valid_time=ds.valid_time + pd.Timedelta(f"{shift_utc}h"))

    tp_factor = None

    if "tp" in ds:
        tp_factor = _precip_to_mm_factor(ds["tp"])
    # Handle accumulated variables (tp, ssrd) — diff to get per-step values

    for acc_var in ["tp", "ssrd", "ssr", "str"]:
        if acc_var in ds:
            # Check if accumulated (monotonically increasing per forecast)
            da = ds[acc_var]
            diff = da.diff(dim="valid_time")
            # Clip negative resets (forecast boundary)
            ds[acc_var] = diff.clip(min=0)

    # Resample to daily
    time_dim = "valid_time" if "valid_time" in ds.dims else "step"
    has_members = "number" in ds.dims

    daily_vars = {}

    # Temperature: K → °C, daily min/max/mean
    for t_var in ["t2m", "2t"]:
        if t_var in ds:
            t_daily = ds[t_var].resample({time_dim: "1D"})
            daily_vars["t_min"] = t_daily.min() - 273.15
            daily_vars["t_max"] = t_daily.max() - 273.15
            daily_vars["t_mean"] = t_daily.mean() - 273.15
            break

    # Dewpoint: K → °C
    for d_var in ["d2m", "2d"]:
        if d_var in ds:
            daily_vars["dewpoint"] = ds[d_var].resample({time_dim: "1D"}).mean() - 273.15
            break

    # Wind speed at 10m
    u_var = "u10" if "u10" in ds else ("10u" if "10u" in ds else None)
    v_var = "v10" if "v10" in ds else ("10v" if "10v" in ds else None)
    if u_var and v_var:
        ws = np.sqrt(ds[u_var] ** 2 + ds[v_var] ** 2)
        daily_vars["wind_speed_10m"] = ws.resample({time_dim: "1D"}).mean()

    # Pressure: Pa → kPa
    for p_var in ["sp", "msl"]:
        if p_var in ds:
            daily_vars["pressure_kpa"] = ds[p_var].resample({time_dim: "1D"}).mean() / 1000.0
            break

    # Precipitation: m → mm (already de-accumulated above)
    if "tp" in ds:
        # daily_vars["precip_mm"] = ds["tp"].resample({time_dim: "1D"}).sum() * 1000.0
        # daily_vars["precip_mm"] = daily_vars["precip_mm"].clip(min=0)
        daily_vars["precip_mm"] = (ds["tp"].resample({time_dim: "1D"}).sum() * tp_factor).clip(
            min=0
        )

    # Solar radiation: J/m² → MJ/m²/day (already de-accumulated above)
    if "ssrd" in ds:
        daily_vars["rs_mj"] = ds["ssrd"].resample({time_dim: "1D"}).sum() / 1e6

    result = xr.Dataset(daily_vars)
    result.attrs["source"] = "ECMWF IFS ENS"
    result.attrs["n_members"] = ds.sizes.get("number", 1)

    logger.info(
        "IFS ENS daily: %d days, %d members, %d variables",
        result.sizes.get(time_dim, 0),
        result.sizes.get("number", 1),
        len(daily_vars),
    )
    return result


# ---------------------------------------------------------------------------
# Extract ensemble forcing at parcel level
# ---------------------------------------------------------------------------


def extract_ensemble_parcel_forcing(
    ifs_daily: xr.Dataset,
    parcel_lon: float,
    parcel_lat: float,
    terrain_elevation: float,
    era5_cell_elevation: float,
    lapse_rate: float = -0.0065,
) -> dict[int, dict[str, np.ndarray]]:
    """Extract and downscale IFS ENS daily data for each member at a parcel.

    Applies lapse rate and wind conversion. Returns a dict keyed by
    member number, each containing arrays ready for DailyForcing.

    For full downscaling (radiation correction etc.), pass individual
    member datasets through extract_parcel_forcing() from Block 2.
    """
    from irrigator.atmospheric.forcing import wind_10m_to_2m

    has_members = "number" in ifs_daily.dims
    members = ifs_daily.number.values if has_members else [0]
    time_dim = "valid_time" if "valid_time" in ifs_daily.dims else "time"

    dz = terrain_elevation - era5_cell_elevation
    t_correction = lapse_rate * dz

    result = {}
    for member in members:
        if has_members:
            cell = ifs_daily.sel(
                number=member,
                longitude=parcel_lon,
                latitude=parcel_lat,
                method="nearest",
            )
        else:
            cell = ifs_daily.sel(
                longitude=parcel_lon,
                latitude=parcel_lat,
                method="nearest",
            )

        member_data = {
            "dates": cell[time_dim].values,
            "t_min": cell["t_min"].values + t_correction,
            "t_max": cell["t_max"].values + t_correction,
            "t_mean": cell["t_mean"].values + t_correction,
            "dewpoint": cell["dewpoint"].values + t_correction if "dewpoint" in cell else None,
            "wind_speed_2m": wind_10m_to_2m(cell["wind_speed_10m"].values)
            if "wind_speed_10m" in cell
            else None,
            "pressure_kpa": cell["pressure_kpa"].values if "pressure_kpa" in cell else None,
            "rs_mj": cell["rs_mj"].values if "rs_mj" in cell else None,
            "precip_mm": cell["precip_mm"].values if "precip_mm" in cell else None,
        }
        result[int(member)] = member_data

    logger.info(
        "Extracted %d IFS ENS members at (%.2f, %.2f), lapse ΔT=%.2f°C",
        len(result),
        parcel_lat,
        parcel_lon,
        t_correction,
    )
    return result


# ---------------------------------------------------------------------------
# Daily archive — save/load/pipeline for France-level IFS ENS
# ---------------------------------------------------------------------------

DEFAULT_PROCESSED_DIR = Path("data/processed")


def save_ifs_daily(
    ds: xr.Dataset,
    run_date: date,
    run_hour: int = 0,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Save processed IFS ENS daily data as a France-level cache file.

    Parameters
    ----------
    ds : daily IFS ENS dataset (from process_ifs_ens_to_daily)
    run_date : forecast initialization date
    run_hour : initialization hour
    processed_dir : output directory

    Returns
    -------
    Path to saved file.
    """
    out_dir = Path(processed_dir) / "ifs_ens"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ifs_daily_{run_date.isoformat()}_{run_hour:02d}z.nc"

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)

    size_mb = out_path.stat().st_size / 1e6
    n_members = ds.sizes.get("number", 1)
    n_days = ds.sizes.get("valid_time", 0)
    logger.info(
        "Saved IFS ENS daily: %s (%.1f MB, %d members, %d days)",
        out_path,
        size_mb,
        n_members,
        n_days,
    )
    return out_path


def load_ifs_daily(
    run_date: date | None = None,
    run_hour: int = 0,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> xr.Dataset:
    """Load cached IFS ENS daily data.

    Parameters
    ----------
    run_date : specific run to load (default: most recent available)
    run_hour : initialization hour
    processed_dir : directory containing ifs_ens/ subdirectory

    Returns
    -------
    xr.Dataset with dims (valid_time, number, latitude, longitude).
    """
    ifs_dir = Path(processed_dir) / "ifs_ens"
    if not ifs_dir.exists():
        raise FileNotFoundError(f"No IFS ENS cache at {ifs_dir}. Run run_ifs_pipeline() first.")

    if run_date is not None:
        path = ifs_dir / f"ifs_daily_{run_date.isoformat()}_{run_hour:02d}z.nc"
        if not path.exists():
            raise FileNotFoundError(f"IFS ENS daily not found: {path}")
        return xr.open_dataset(path)

    # Find most recent
    files = sorted(ifs_dir.glob("ifs_daily_*.nc"))
    if not files:
        raise FileNotFoundError(f"No IFS ENS daily files in {ifs_dir}")
    return xr.open_dataset(files[-1])


# ---------------------------------------------------------------------------
# Pipeline: fetch → open France → daily → save (optionally delete raw)
# ---------------------------------------------------------------------------


def process_ifs_run(
    run_date: date,
    run_hour: int = 0,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    bbox: BBoxWGS84 = FRANCE_BBOX,
    shift_utc: int = 0,
    overwrite: bool = False,
    keep_raw: bool = False,
    mode: str = "bulk"
) -> Path | None:
    """Fetch, process, and cache one IFS ENS run at France level.

    Steps:
    1. Download global GRIB from ECMWF open data
    2. Open and slice to France bounding box
    3. Process 6-hourly → daily (de-accumulate tp/ssrd)
    4. Save compressed NetCDF
    5. Delete raw GRIB (unless keep_raw=True)

    Parameters
    ----------
    run_date : forecast initialization date
    run_hour : initialization hour (0 or 12)
    raw_dir : directory for raw GRIB download
    processed_dir : directory for daily NetCDF output
    bbox : spatial extent for slicing (default: France)
    shift_utc : timezone shift before daily aggregation
    overwrite : re-process even if daily file exists
    keep_raw : if False (default), delete the raw GRIB after processing

    Returns
    -------
    Path to the daily NetCDF, or None if download failed.
    """
    # Check if already processed
    out_dir = Path(processed_dir) / "ifs_ens"
    out_path = out_dir / f"ifs_daily_{run_date.isoformat()}_{run_hour:02d}z.nc"
    if out_path.exists() and not overwrite:
        logger.info("IFS ENS %s %02dZ already processed: %s", run_date, run_hour, out_path)
        return out_path

    tag = f"{run_date.isoformat()} {run_hour:02d}Z"

    # Step 1: Download
    logger.info("[%s] Downloading...", tag)
    try:
        grib_path = fetch_ifs_ens(raw_dir, run_date, run_hour, overwrite=overwrite, download_mode=mode)
    except Exception as exc:
        logger.warning("[%s] Download failed: %s", tag, exc)
        return None

    # Step 2: Open and slice to France
    logger.info("[%s] Opening and slicing to bbox...", tag)
    ds_raw = open_ifs_ens(grib_path, bbox)

    # Step 3: Process to daily
    logger.info("[%s] Processing to daily...", tag)
    ifs_daily = process_ifs_ens_to_daily(ds_raw, shift_utc=shift_utc)

    # Step 4: Save
    daily_path = save_ifs_daily(ifs_daily, run_date, run_hour, processed_dir)

    # Step 5: Clean up raw GRIB
    if not keep_raw and grib_path.exists():
        raw_mb = grib_path.stat().st_size / 1e6
        grib_path.unlink()
        # Also remove any .idx sidecar files
        for idx in grib_path.parent.glob(f"{grib_path.stem}*.idx"):
            idx.unlink(missing_ok=True)
        logger.info("[%s] Deleted raw GRIB (%.0f MB freed)", tag, raw_mb)

    return daily_path


def run_ifs_pipeline(
    start: date | None = None,
    end: date | None = None,
    run_hour: int = 0,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    shift_utc: int = 0,
    overwrite: bool = False,
    keep_raw: bool = False,
    mode: str = "bulk",
) -> list[Path]:
    """Fetch and process IFS ENS for a date range.

    Iterates day by day from start to end, downloading and processing
    each run. Skips runs that are already cached (unless overwrite=True).

    Parameters
    ----------
    start : first run date (default: today)
    end : last run date (default: same as start)
    run_hour : initialization hour for all runs (0 or 12)
    raw_dir : directory for raw GRIB downloads
    processed_dir : directory for daily NetCDF output
    shift_utc : timezone shift before daily aggregation
    overwrite : re-process even if daily file exists
    keep_raw : if False (default), delete raw GRIBs after processing

    Returns
    -------
    List of paths to daily NetCDF files.
    """
    if start is None:
        start = date.today()
    if end is None:
        end = date.today()

    daily_paths = []
    current = start

    while current <= end:
        result = process_ifs_run(
            current,
            run_hour,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            shift_utc=shift_utc,
            overwrite=overwrite,
            keep_raw=keep_raw,
            mode=mode,
        )
        if result is not None:
            daily_paths.append(result)
        current += timedelta(days=1)

        # Pause between downloads to avoid AWS/Azure S3 rate limiting
        if current <= end:
            import time

            time.sleep(30)

    logger.info(
        "IFS pipeline complete: %d/%d runs processed",
        len(daily_paths),
        (end - start).days + 1,
    )
    return daily_paths
