"""Historical ECMWF IFS ENS retrieval from the TIGGE archive.

This module complements :mod:`irrigator.ingestion.ifs_ens_client`, which
retrieves recent IFS ENS runs from ECMWF Open Data.  TIGGE contains the
historical operational ensemble and is therefore suitable for backtesting.

The processed output deliberately uses the same daily variable names as the
rest of IrriGator::

    t_min, t_max, t_mean, dewpoint, wind_speed_10m,
    pressure_kpa, precip_mm, rs_mj

but is stored separately under ``data/processed/ifs_tigge`` to preserve
provenance.

Notes
-----
* TIGGE total precipitation is accumulated from forecast start and encoded in
  kg m-2, numerically equivalent to mm of water.
* TIGGE archives surface *net* solar radiation (``ssr``), not downward solar
  radiation (``ssrd``).  IrriGator's FAO-56 forcing expects incoming shortwave
  radiation, so ``ssr`` is converted with the reference-crop albedo 0.23::

      Rs ~= Rns / (1 - 0.23)

  This is an approximation and should be validated against overlapping live
  IFS ``ssrd`` data (e.g. the 2026 archive).
* By default retrieval starts at +48 h.  The +48 h accumulated value is kept
  as the baseline required to recover the +48 -> +54 h increment; therefore
  the first complete daily precipitation total begins on issue_date + 2.
"""

from __future__ import annotations

import logging
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from irrigator.config import BBoxWGS84

logger = logging.getLogger(__name__)

ECDS_URL = "https://ecds.ecmwf.int/api"
TIGGE_DATASET = "tigge-forecasts"

FRANCE_BBOX = BBoxWGS84(north=51.5, south=41.0, west=-6.0, east=10.0)
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")

# TIGGE / ecCodes parameter identifiers.
# https://confluence.ecmwf.int/spaces/TIGGE/pages/40109884/Parameters
# TIGGE_PARAMS = {
#     "sp": "134",  # surface pressure [Pa]
#     "u10": "165",  # 10 m U wind [m/s]
#     "v10": "166",  # 10 m V wind [m/s]
#     "t2m": "167",  # 2 m temperature [K]
#     "d2m": "168",  # 2 m dewpoint [K]
#     "ssr": "176",  # surface net solar radiation [W m-2 s], accumulated
#     "tp": "228228",  # total precipitation [kg m-2], accumulated
#     "mx2t6": "121",
#     "mn2t6": "122",
# }

TIGGE_VARIABLES = [
    "10_m_u_component_of_wind",
    "10_m_v_component_of_wind",
    "2_m_dewpoint_temperature",
    "2_m_temperature",
    "maximum_2_m_temperature_in_the_last_6_hours",
    "minimum_2_m_temperature_in_the_last_6_hours",
    "surface_pressure",
    "surface_net_solar_radiation",
    "total_precipitation",
]

PARAM_ID_RENAMES = {
    121: "mx2t6",
    122: "mn2t6",
}

TIGGE_CFGRIB_NAMES = set(TIGGE_VARIABLES)
TIGGE_CFGRIB_NAMES = {"sp", "u10", "v10", "t2m", "d2m", "ssr", "tp", "mx2t6", "mn2t6"}

REQUIRED_TIGGE_VARS = {"sp", "u10", "v10", "t2m", "d2m", "ssr", "tp", "mx2t6", "mn2t6"}


def _tigge_raw_dir(raw_dir: Path, run_date: date, run_hour: int) -> Path:
    out = Path(raw_dir) / "ifs_tigge" / f"{run_date.isoformat()}_{run_hour:02d}z"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _tigge_daily_path(
    run_date: date,
    run_hour: int,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    out_dir = Path(processed_dir) / "ifs_tigge"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"ifs_daily_{run_date.isoformat()}_{run_hour:02d}z.nc"


def _mars_area(bbox: BBoxWGS84, grid: float) -> str:
    """Return a TIGGE/MARS area string in N/W/S/E order."""
    north, west, south, east = bbox.as_cds_area(grid_step=grid)
    return f"{north}/{west}/{south}/{east}"


def _step_spec(first_step: int, last_step: int, step_hours: int) -> str:
    if first_step < 0 or last_step < first_step:
        raise ValueError("Require 0 <= first_step <= last_step")
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    if (last_step - first_step) % step_hours != 0:
        raise ValueError("last_step - first_step must be divisible by step_hours")
    if first_step == last_step:
        return str(first_step)
    return f"{first_step}/to/{last_step}/by/{step_hours}"


def _get_ecds_client(key: str | None = None):
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover - dependency is in pyproject
        raise ImportError("Install cdsapi to retrieve TIGGE data: pip install cdsapi") from exc

    kwargs: dict[str, str] = {"url": ECDS_URL}
    if key is not None:
        kwargs["key"] = key
    return cdsapi.Client(**kwargs)

def _leadtime_hours(
    first_step: int,
    last_step: int,
    step_hours: int,
) -> list[str]:
    return [
        str(step)
        for step in range(
            first_step,
            last_step + 1,
            step_hours,
        )
    ]


def fetch_tigge_type(
    run_date: date,
    forecast_type: str,
    target: Path,
    *,
    run_hour: int = 0,
    bbox: BBoxWGS84 = FRANCE_BBOX,
    first_step: int = 48,
    last_step: int = 360,
    step_hours: int = 6,
    grid: float = 0.25,
    key: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Retrieve one TIGGE control or perturbed IFS forecast.

    Parameters
    ----------
    run_date
        Operational forecast initialisation date.
    forecast_type
        ``"cf"`` for the control forecast or ``"pf"`` for perturbed
        members 1..50.
    target
        Output GRIB2 path.
    run_hour
        Initialisation hour (normally 0 or 12 UTC).
    bbox
        Spatial subset.  Subsetting is performed server-side by ECDS.
    first_step, last_step, step_hours
        Lead-time range in hours.  The default +48..+360 range matches the
        AROME -> IFS handoff used by IrriGator.
    grid
        Output interpolation grid in degrees.
    key
        Optional ECDS API key.  If omitted, ``cdsapi`` reads ``~/.cdsapirc``.
    overwrite
        Replace an existing GRIB file.
    """
    if forecast_type not in {"cf", "pf"}:
        raise ValueError("forecast_type must be 'cf' or 'pf'")
    if run_hour not in {0, 12}:
        raise ValueError("run_hour must be 0 or 12 for ECMWF TIGGE forecasts")

    target = Path(target)
    if target.exists() and not overwrite:
        logger.info("TIGGE %s already exists: %s", forecast_type, target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

    forecast_type_ecds = {
            "cf": "control_forecast",
            "pf": "perturbed_forecast",
        }[forecast_type]

    request = {
        "origin": "ecmwf",
        "year": str(run_date.year),
        "month": f"{run_date.month:02d}",
        "day": f"{run_date.day:02d}",
        "time": f"{run_hour:02d}:00",
        "level_type": "single_level",
        "variable": TIGGE_VARIABLES,
        "forecast_type": forecast_type_ecds,
        "leadtime_hour": _leadtime_hours(
            first_step,
            last_step,
            step_hours,
        ),
        "area": [
            bbox.north,
            bbox.west,
            bbox.south,
            bbox.east,
        ],
        "grid": f"{grid}/{grid}",
        "data_format": "grib",
    }

    

    logger.info(
        "Retrieving TIGGE %s: %s %02dZ, steps %d-%d h, area=%s",
        forecast_type,
        run_date,
        run_hour,
        first_step,
        last_step,
        request["area"],
    )

    client = _get_ecds_client(key=key)
    try:
        client.retrieve(TIGGE_DATASET, request, str(target))
    except Exception:
        target.unlink(missing_ok=True)
        raise

    logger.info("Retrieved %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target


def fetch_tigge_ifs_run(
    run_date: date,
    *,
    run_hour: int = 0,
    raw_dir: Path = DEFAULT_RAW_DIR,
    bbox: BBoxWGS84 = FRANCE_BBOX,
    first_step: int = 48,
    last_step: int = 360,
    step_hours: int = 6,
    grid: float = 0.25,
    key: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Retrieve control + 50 perturbed historical IFS members from TIGGE."""
    out_dir = _tigge_raw_dir(raw_dir, run_date, run_hour)
    control = out_dir / "control.grib2"
    perturbed = out_dir / "perturbed.grib2"

    fetch_tigge_type(
        run_date,
        "cf",
        control,
        run_hour=run_hour,
        bbox=bbox,
        first_step=first_step,
        last_step=last_step,
        step_hours=step_hours,
        grid=grid,
        key=key,
        overwrite=overwrite,
    )
    fetch_tigge_type(
        run_date,
        "pf",
        perturbed,
        run_hour=run_hour,
        bbox=bbox,
        first_step=first_step,
        last_step=last_step,
        step_hours=step_hours,
        grid=grid,
        key=key,
        overwrite=overwrite,
    )
    return control, perturbed


def _slice_bbox(ds: xr.Dataset, bbox: BBoxWGS84) -> xr.Dataset:
    """Defensive bbox slice; ECDS should already have performed this subset."""
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        return ds

    north, south = bbox.north + 0.5, bbox.south - 0.5
    west, east = bbox.west - 0.5, bbox.east + 0.5

    lat = ds.latitude
    if lat.size > 1 and float(lat[0]) > float(lat[-1]):
        ds = ds.sel(latitude=slice(north, south))
    else:
        ds = ds.sel(latitude=slice(south, north))

    if float(ds.longitude.max()) > 180:
        west_360, east_360 = west % 360, east % 360
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
            ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
    else:
        lon = ds.longitude
        if lon.size > 1 and float(lon[0]) > float(lon[-1]):
            ds = ds.sel(longitude=slice(east, west))
        else:
            ds = ds.sel(longitude=slice(west, east))
    return ds


def _open_cfgrib_group(
    path: Path,
    filter_by_keys: dict[str, object],
    bbox: BBoxWGS84,
) -> xr.Dataset | None:
    try:
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": filter_by_keys,
                "indexpath": "",
            },
            decode_timedelta=True,
        )
    except Exception as exc:
        logger.debug("cfgrib group %s not present in %s: %s", filter_by_keys, path, exc)
        return None

    ds = _slice_bbox(ds, bbox)

    # cfgrib exposes the TIGGE 6-hour extrema as `t2m`.
    # Rename according to GRIB stepType so they remain distinct
    # from instantaneous 2 m temperature.
    step_type = filter_by_keys.get("stepType")

    for name in list(ds.data_vars):
        param_id = ds[name].attrs.get("GRIB_paramId")

        if param_id in PARAM_ID_RENAMES:
            ds = ds.rename(
                {name: PARAM_ID_RENAMES[param_id]}
            )

    present = [name for name in ds.data_vars if name in TIGGE_CFGRIB_NAMES]
    if not present:
        ds.close()
        return None

    ds = ds[present].drop_vars(
        ["heightAboveGround", "surface", "stepType"],
        errors="ignore",
    )
    return ds


def open_tigge_ifs(path: Path, bbox: BBoxWGS84 = FRANCE_BBOX) -> xr.Dataset:
    """Open TIGGE IFS GRIB fields into a single xarray Dataset."""
    path = Path(path)
    groups = [
        {
            "typeOfLevel": "heightAboveGround",
            "level": 2,
            "stepType": "instant",
        },
        {
            "typeOfLevel": "heightAboveGround",
            "level": 2,
            "stepType": "max",
        },
        {
            "typeOfLevel": "heightAboveGround",
            "level": 2,
            "stepType": "min",
        },
        {
            "typeOfLevel": "heightAboveGround",
            "level": 10,
            "stepType": "instant",
        },
        {
            "typeOfLevel": "surface",
            "stepType": "instant",
        },
        {
            "typeOfLevel": "surface",
            "stepType": "accum",
        },
    ]

    datasets = [ds for keys in groups if (ds := _open_cfgrib_group(path, keys, bbox)) is not None]
    if not datasets:
        raise RuntimeError(f"Could not open TIGGE variables from {path}")

    ds = xr.merge(
        datasets,
        compat="override",
        join="outer",
        combine_attrs="override",
    )

    

    missing = sorted(REQUIRED_TIGGE_VARS - set(ds.data_vars))
    if missing:
        raise RuntimeError(
            f"TIGGE file is missing required variables {missing}. Available: {sorted(ds.data_vars)}"
        )
    return ds


def _to_valid_time(ds: xr.Dataset, shift_utc: int = 0) -> xr.Dataset:
    if "valid_time" in ds.dims:
        result = ds
    elif "step" in ds.dims:
        ref_time = ds.coords.get("time")
        if ref_time is None:
            ref_time = ds.coords.get("forecast_reference_time")
        if ref_time is None:
            raise ValueError("TIGGE dataset has step but no forecast reference time")
        result = ds.assign_coords(valid_time=ref_time + ds.step).swap_dims({"step": "valid_time"})
    else:
        raise ValueError("TIGGE dataset has neither valid_time nor step dimension")

    if shift_utc:
        result = result.assign_coords(valid_time=result.valid_time + pd.Timedelta(hours=shift_utc))
    return result.sortby("valid_time")


def _deaccumulate(da: xr.DataArray) -> xr.DataArray:
    """Convert forecast-start accumulation into per-step increments.

    Accumulated fields are timestamped at the *end* of their accumulation
    interval.  Shift each increment infinitesimally backwards so an interval
    ending exactly at 00:00 (e.g. 18-24 UTC) is assigned to the preceding
    meteorological day during daily resampling.
    """
    increments = da.diff("valid_time", label="upper").clip(min=0)
    increments = increments.assign_coords(
        valid_time=increments.valid_time - np.timedelta64(1, "ns")
    )
    return increments


def load_albedo_for_tigge(
    albedo_path: str | Path,
    tigge_ds: xr.Dataset | xr.DataArray,
) -> xr.DataArray:
    """Load ERA5 albedo needed for a TIGGE forecast."""

    with xr.open_dataset(albedo_path) as ds:
        albedo = ds["forecast_albedo"]

        albedo = albedo.sel(valid_time=tigge_ds.valid_time).load()

    albedo = albedo.sortby("latitude").sortby("longitude")

    albedo = albedo.interp(
        latitude=tigge_ds.latitude,
        longitude=tigge_ds.longitude,
        method="linear",
    )

    return albedo


def _interval_end_to_previous_day(
    da: xr.DataArray,
) -> xr.DataArray:
    return da.assign_coords(valid_time=(da.valid_time - np.timedelta64(1, "ns")))


def process_tigge_to_daily(
    ds: xr.Dataset,
    *,
    shift_utc: int = 0,
    albedo_path: Path | None = None,
) -> xr.Dataset:
    """Convert one TIGGE control/PF Dataset to IrriGator daily fields.

    The first accumulated value is a baseline only.  For the default +48 h
    retrieval, precipitation/radiation increments start at +54 h and form a
    complete +48 -> +72 h total for issue_date + 2.

    Use the albedo from ERA5 at daily level.
    """
    if albedo_path is None:
        from irrigator.atmospheric.albedo import _albedo_output_path
        albedo_path = _albedo_output_path(DEFAULT_PROCESSED_DIR)

    ds = _to_valid_time(ds, shift_utc=shift_utc)

    tp_step = _deaccumulate(ds["tp"])
    ssr_step = _deaccumulate(ds["ssr"])

    daily: dict[str, xr.DataArray] = {}

    # temperature = ds["t2m"].resample(valid_time="1D")
    # daily["t_min"] = temperature.min() - 273.15
    # daily["t_max"] = temperature.max() - 273.15
    # daily["t_mean"] = temperature.mean() - 273.15

    mn2t6 = _interval_end_to_previous_day(
        ds["mn2t6"]
    )

    mx2t6 = _interval_end_to_previous_day(
        ds["mx2t6"]
    )


    daily["t_min"] = (
        mn2t6
        .resample(valid_time="1D")
        .min()
        - 273.15
    )

    daily["t_max"] = (
        mx2t6
        .resample(valid_time="1D")
        .max()
        - 273.15
    )

    daily["t_mean"] = ds["t2m"].resample(valid_time="1D").mean() - 273.15


    daily["dewpoint"] = ds["d2m"].resample(valid_time="1D").mean() - 273.15

    wind = np.hypot(ds["u10"], ds["v10"])
    daily["wind_speed_10m"] = wind.resample(valid_time="1D").mean()

    daily["pressure_kpa"] = ds["sp"].resample(valid_time="1D").mean() / 1000.0

    # TIGGE tp unit is kg m-2, i.e. 1 kg m-2 == 1 mm liquid water.
    daily["precip_mm"] = tp_step.resample(valid_time="1D").sum().clip(min=0)

    # TIGGE ssr is net shortwave energy accumulated from forecast start.
    # W m-2 s is dimensionally J m-2.  Convert to incoming shortwave so the
    # FAO-56 works fine.
    ssr_daily = (
        ssr_step
        .resample(valid_time="1D")
        .sum()
        / 1e6
    )

    albedo = load_albedo_for_tigge(
        albedo_path,
        ssr_daily,
    )


    daily["rs_mj"] = (
        ssr_daily
        / (1.0 - albedo)
    )

    result = xr.Dataset(daily)

    # The first resampled day can contain instantaneous fields but no complete
    # accumulated total if retrieval did not include the preceding baseline.
    # Keep only days for which precipitation/radiation are finite somewhere.
    completeness = result["precip_mm"].notnull() & result["rs_mj"].notnull()
    spatial_dims = [d for d in completeness.dims if d != "valid_time"]
    if spatial_dims:
        completeness = completeness.any(dim=spatial_dims)
    complete_times = result.valid_time.where(completeness, drop=True)
    result = result.sel(valid_time=complete_times)

    result.attrs.update(
        {
            "source": "ECMWF IFS ENS historical operational forecast (TIGGE)",
            "radiation_source": "TIGGE surface net solar radiation (ssr)",
            "radiation_conversion": ("rs_mj = ssr_mj / (1 - ERA5 forecast_albedo)"),
            "albedo_source": "ERA5 forecast_albedo",
            "albedo_file": str(albedo_path),
            "precipitation_source_units": "kg m-2 (numerically mm)",
            "n_members": int(ds.sizes.get("number", 1)),
        }
    )
    return result


def combine_tigge_members(control_daily: xr.Dataset, perturbed_daily: xr.Dataset) -> xr.Dataset:
    """Combine TIGGE control as member 0 with perturbed members 1..50."""
    control = control_daily
    if "number" not in control.dims:
        control = control.expand_dims(number=[0])
    else:
        control = control.assign_coords(number=[0])

    perturbed = perturbed_daily
    if "number" not in perturbed.dims:
        raise RuntimeError("Perturbed TIGGE forecast has no ensemble 'number' dimension")

    combined = xr.concat(
        [control, perturbed],
        dim="number",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="inner",
    ).sortby("number")

    numbers = [int(n) for n in combined.number.values]
    if 0 not in numbers:
        raise RuntimeError("Combined TIGGE ensemble is missing control member 0")
    if len(numbers) != len(set(numbers)):
        raise RuntimeError(f"Duplicate TIGGE ensemble member ids: {numbers}")

    combined.attrs.update(control_daily.attrs)
    combined.attrs["n_members"] = len(numbers)
    combined.attrs["member_definition"] = "0=control, 1..50=perturbed"
    return combined


def save_tigge_daily(
    ds: xr.Dataset,
    run_date: date,
    run_hour: int = 0,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Save one processed historical IFS run under ``ifs_tigge/``."""
    path = _tigge_daily_path(run_date, run_hour, processed_dir)
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    logger.info(
        "Saved TIGGE daily %s (%.1f MB, %d members, %d days)",
        path,
        path.stat().st_size / 1e6,
        ds.sizes.get("number", 1),
        ds.sizes.get("valid_time", 0),
    )
    return path


def load_tigge_daily(
    run_date: date,
    run_hour: int = 0,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> xr.Dataset:
    """Load a processed TIGGE daily cache."""
    path = _tigge_daily_path(run_date, run_hour, processed_dir)
    if not path.exists():
        raise FileNotFoundError(f"TIGGE daily cache not found: {path}")
    return xr.open_dataset(path)


def process_tigge_run(
    run_date: date,
    run_hour: int = 0,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    bbox: BBoxWGS84 = FRANCE_BBOX,
    first_step: int = 48,
    last_step: int = 360,
    step_hours: int = 6,
    grid: float = 0.25,
    shift_utc: int = 0,
    albedo_path: Path |None = None,
    key: str | None = None,
    overwrite: bool = False,
    keep_raw: bool = False,
) -> Path:
    """Retrieve, process, combine and cache one historical IFS ENS run."""
    out_path = _tigge_daily_path(run_date, run_hour, processed_dir)
    if out_path.exists() and not overwrite:
        logger.info("TIGGE %s %02dZ already processed: %s", run_date, run_hour, out_path)
        return out_path

    control_path, perturbed_path = fetch_tigge_ifs_run(
        run_date,
        run_hour=run_hour,
        raw_dir=raw_dir,
        bbox=bbox,
        first_step=first_step,
        last_step=last_step,
        step_hours=step_hours,
        grid=grid,
        key=key,
        overwrite=overwrite,
    )

    logger.info("Opening TIGGE control forecast...")
    control_raw = open_tigge_ifs(control_path, bbox=bbox)
    control_daily = process_tigge_to_daily(
        control_raw,
        shift_utc=shift_utc,
        albedo_path=albedo_path,
    )

    logger.info("Opening TIGGE perturbed forecast...")
    perturbed_raw = open_tigge_ifs(perturbed_path, bbox=bbox)
    perturbed_daily = process_tigge_to_daily(
        perturbed_raw,
        shift_utc=shift_utc,
        albedo_path=albedo_path,
    )

    combined = combine_tigge_members(control_daily, perturbed_daily)
    combined.attrs.update(
        {
            "forecast_reference_date": run_date.isoformat(),
            "forecast_reference_hour_utc": int(run_hour),
            "first_retrieved_step_h": int(first_step),
            "last_retrieved_step_h": int(last_step),
            "retrieval_grid_deg": float(grid),
        }
    )
    daily_path = save_tigge_daily(combined, run_date, run_hour, processed_dir)

    for ds in (control_raw, perturbed_raw):
        try:
            ds.close()
        except Exception:
            pass

    if not keep_raw:
        raw_run_dir = control_path.parent
        if raw_run_dir.exists():
            shutil.rmtree(raw_run_dir)
            logger.info("Deleted raw TIGGE directory: %s", raw_run_dir)

    return daily_path


def run_tigge_pipeline(
    start: date,
    end: date | None = None,
    run_hour: int = 0,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    bbox: BBoxWGS84 = FRANCE_BBOX,
    first_step: int = 48,
    last_step: int = 360,
    step_hours: int = 6,
    grid: float = 0.25,
    shift_utc: int = 0,
    albedo_path: Path | None = None,
    key: str | None = None,
    overwrite: bool = False,
    keep_raw: bool = False,
    pause_seconds: float = 0.0,
) -> list[Path]:
    """Process a date range of historical operational IFS runs."""
    if end is None:
        end = start
    if end < start:
        raise ValueError("end must be on or after start")

    paths: list[Path] = []
    current = start
    while current <= end:
        paths.append(
            process_tigge_run(
                current,
                run_hour,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                bbox=bbox,
                first_step=first_step,
                last_step=last_step,
                step_hours=step_hours,
                grid=grid,
                shift_utc=shift_utc,
                albedo_path=albedo_path,
                key=key,
                overwrite=overwrite,
                keep_raw=keep_raw,
            )
        )
        current += timedelta(days=1)
        if current <= end and pause_seconds > 0:
            time.sleep(pause_seconds)

    return paths
