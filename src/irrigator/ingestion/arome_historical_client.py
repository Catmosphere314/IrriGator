"""Historical AROME fixed-lead archive using Open-Meteo Previous Runs.

Météo-France does not expose a long public archive of arbitrary historical
AROME initialisation runs.  Open-Meteo's Previous Runs API provides a useful
alternative for backtesting: each ``*_previous_dayN`` series contains values
forecast N * 24 hours before their valid time.

This is *not* an exact archived 00Z AROME run.  The distinction is retained in
both file paths and dataset metadata.  Exact 2026 IrriGator AROME runs continue
to live under ``data/processed/arome/forecast``; these reconstructed historical
series are stored under::

    data/processed/arome/historical/<parcel_id>/

The API is point-based, so archives are intentionally parcel-specific.  Data
are requested with ``elevation=nan`` and ``cell_selection=nearest`` to avoid
Open-Meteo's terrain downscaling; downstream IrriGator processing can then use
the same correction path as the existing AROME/IFS forcing pipeline.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr

from irrigator.config import ParcelConfig

logger = logging.getLogger(__name__)

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODEL = "meteofrance_arome_france"

# Open-Meteo variable names.  Suffixes such as _previous_day1 are appended at
# request time.
HOURLY_VARIABLES = (
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation",
    "shortwave_radiation",
)


def _validate_lead_days(lead_days: tuple[int, ...]) -> tuple[int, ...]:
    leads = tuple(sorted(set(int(v) for v in lead_days)))
    if not leads:
        raise ValueError("lead_days must contain at least one lead")
    if leads[0] < 0 or leads[-1] > 7:
        raise ValueError("Open-Meteo Previous Runs supports lead days 0..7")
    return leads


def _archive_path(
    parcel_id: str,
    year: int,
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    model: str = DEFAULT_MODEL,
) -> Path:
    safe_model = model.replace("/", "_")
    out_dir = Path(processed_dir) / "arome" / "historical" / parcel_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"arome_previous_runs_{safe_model}_{year}.nc"


def _requested_hourly_names(lead_days: tuple[int, ...]) -> list[str]:
    names: list[str] = []
    for lead in lead_days:
        suffix = f"_previous_day{lead}"
        names.extend(f"{name}{suffix}" for name in HOURLY_VARIABLES)
    return names


def _request_chunk(
    start_date: date,
    end_date: date,
    *,
    lat: float,
    lon: float,
    lead_days: tuple[int, ...],
    model: str,
    timeout: float,
    session: requests.Session,
    api_key: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(_requested_hourly_names(lead_days)),
        "models": model,
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        # Disable Open-Meteo elevation downscaling.  We want the closest raw
        # model grid cell and then keep IrriGator's own correction pathway.
        "elevation": "nan",
        "cell_selection": "nearest",
    }
    if api_key is not None:
        params["apikey"] = api_key

    logger.info(
        "Open-Meteo AROME previous runs: %s -> %s, parcel=(%.4f, %.4f), leads=%s",
        start_date,
        end_date,
        lat,
        lon,
        lead_days,
    )

    response = session.get(PREVIOUS_RUNS_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason', payload)}")
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise RuntimeError(f"Unexpected Open-Meteo response: keys={list(payload)}")
    return payload


def _series(payload: dict[str, Any], name: str, time_index: pd.DatetimeIndex) -> pd.Series:
    values = payload["hourly"].get(name)
    if values is None:
        return pd.Series(np.nan, index=time_index, dtype=float)
    return pd.Series(pd.to_numeric(values, errors="coerce"), index=time_index, dtype=float)


def _payload_to_daily(
    payload: dict[str, Any],
    lead_days: tuple[int, ...],
) -> xr.Dataset:
    """Aggregate one Previous Runs JSON response to IrriGator daily fields."""
    times = pd.to_datetime(payload["hourly"]["time"], utc=True).tz_convert(None)
    if len(times) == 0:
        raise RuntimeError("Open-Meteo returned an empty hourly time axis")

    per_lead: list[xr.Dataset] = []

    for lead in lead_days:
        suffix = f"_previous_day{lead}"
        temperature = _series(payload, f"temperature_2m{suffix}", times)
        dewpoint = _series(payload, f"dew_point_2m{suffix}", times)
        wind = _series(payload, f"wind_speed_10m{suffix}", times)
        pressure_hpa = _series(payload, f"surface_pressure{suffix}", times)
        precip = _series(payload, f"precipitation{suffix}", times)
        shortwave = _series(payload, f"shortwave_radiation{suffix}", times)

        daily_index = temperature.resample("1D").mean().index
        frame = pd.DataFrame(
            {
                "t_min": temperature.resample("1D").min(),
                "t_max": temperature.resample("1D").max(),
                "t_mean": temperature.resample("1D").mean(),
                "dewpoint": dewpoint.resample("1D").mean(),
                "wind_speed_10m": wind.resample("1D").mean(),
                # Open-Meteo surface pressure is hPa.  1 hPa = 0.1 kPa.
                "pressure_kpa": pressure_hpa.resample("1D").mean() / 10.0,
                "precip_mm": precip.resample("1D").sum(min_count=1),
                # Hourly shortwave_radiation is the mean flux over the
                # preceding hour [W/m2]; multiply by 3600 s and convert J->MJ.
                "rs_mj": shortwave.resample("1D").sum(min_count=1) * 0.0036,
            },
            index=daily_index,
        )

        ds = xr.Dataset.from_dataframe(frame)
        ds = ds.rename({"index": "valid_time"}) if "index" in ds.dims else ds
        if "valid_time" not in ds.dims:
            # xarray uses the pandas index name if set, otherwise "index".
            only_dim = next(iter(ds.dims))
            ds = ds.rename({only_dim: "valid_time"})
        ds = ds.expand_dims(lead_day=[lead])
        per_lead.append(ds)

    result = xr.concat(per_lead, dim="lead_day", join="outer").sortby("valid_time")

    # Coordinates returned by Open-Meteo correspond to the selected model
    # cell.  Retaining length-one lat/lon dimensions makes the Dataset usable
    # by the existing extract_parcel_forcing() function.
    used_lat = float(payload.get("latitude", np.nan))
    used_lon = float(payload.get("longitude", np.nan))
    result = result.expand_dims(latitude=[used_lat], longitude=[used_lon])
    result = result.transpose("valid_time", "lead_day", "latitude", "longitude", ...)

    result.attrs.update(
        {
            "source": "Open-Meteo Previous Runs - Météo-France AROME",
            "archive_semantics": (
                "fixed lead-time series; previous_dayN was predicted N*24 hours before valid time"
            ),
            "model": str(payload.get("model", DEFAULT_MODEL)),
            "requested_latitude": float(payload.get("latitude", np.nan)),
            "requested_longitude": float(payload.get("longitude", np.nan)),
            "open_meteo_elevation_downscaling": "disabled (elevation=nan)",
            "cell_selection": "nearest",
        }
    )
    return result


def fetch_arome_previous_runs(
    start_date: date,
    end_date: date,
    *,
    lat: float,
    lon: float,
    lead_days: tuple[int, ...] = (0, 1, 2),
    model: str = DEFAULT_MODEL,
    chunk_days: int = 31,
    timeout: float = 120.0,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> xr.Dataset:
    """Fetch and aggregate historical fixed-lead AROME forecasts.

    Requests are split into modest date chunks to avoid very large API
    responses for an entire growing season/year.
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")

    leads = _validate_lead_days(lead_days)
    own_session = session is None
    session = session or requests.Session()

    datasets: list[xr.Dataset] = []
    current = start_date
    try:
        while current <= end_date:
            chunk_end = min(end_date, current + timedelta(days=chunk_days - 1))
            payload = _request_chunk(
                current,
                chunk_end,
                lat=lat,
                lon=lon,
                lead_days=leads,
                model=model,
                timeout=timeout,
                session=session,
                api_key=api_key,
            )
            chunk = _payload_to_daily(payload, leads)
            chunk.attrs["model"] = model
            datasets.append(chunk)
            current = chunk_end + timedelta(days=1)
    finally:
        if own_session:
            session.close()

    if not datasets:
        raise RuntimeError("No AROME Previous Runs data retrieved")

    result = xr.concat(
        datasets,
        dim="valid_time",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="outer",
    ).sortby("valid_time")

    # Remove overlap if an API response includes a boundary timestamp twice.
    _, unique_idx = np.unique(result.valid_time.values, return_index=True)
    result = result.isel(valid_time=np.sort(unique_idx))

    result.attrs.update(datasets[0].attrs)
    result.attrs["model"] = model
    result.attrs["coverage_start"] = start_date.isoformat()
    result.attrs["coverage_end"] = end_date.isoformat()
    result.attrs["lead_days"] = ",".join(str(v) for v in leads)
    return result


def archive_arome_previous_runs_year(
    parcel: ParcelConfig,
    year: int,
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    lead_days: tuple[int, ...] = (0, 1, 2),
    model: str = DEFAULT_MODEL,
    chunk_days: int = 31,
    timeout: float = 120.0,
    api_key: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create one parcel-level historical AROME cache for a calendar year."""
    path = _archive_path(parcel.id, year, processed_dir=processed_dir, model=model)
    if path.exists() and not overwrite:
        logger.info("Historical AROME cache already exists: %s", path)
        return path

    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    ds = fetch_arome_previous_runs(
        start_date,
        end_date,
        lat=parcel.lat,
        lon=parcel.lon,
        lead_days=lead_days,
        model=model,
        chunk_days=chunk_days,
        timeout=timeout,
        api_key=api_key,
    )
    ds.attrs["parcel_id"] = parcel.id
    ds.attrs["parcel_name"] = parcel.name

    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    logger.info("Saved historical AROME cache: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_arome_previous_runs(
    parcel_id: str,
    year: int,
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    model: str = DEFAULT_MODEL,
) -> xr.Dataset:
    """Load a parcel-level year cache created by this module."""
    path = _archive_path(parcel_id, year, processed_dir=processed_dir, model=model)
    if not path.exists():
        raise FileNotFoundError(
            f"Historical AROME cache not found: {path}. "
            "Run archive_arome_previous_runs_year(parcel, year) first."
        )
    return xr.open_dataset(path)


def select_fixed_lead_days(
    ds: xr.Dataset,
    selections: list[tuple[date, int]],
) -> xr.Dataset:
    """Build a deterministic daily forecast from (valid_date, lead_day) pairs.

    This helper is intentionally explicit: it prevents accidental mixing of
    ``previous_day0``/``previous_day1`` semantics during a backtest.
    """
    parts: list[xr.Dataset] = []
    for valid_date, lead_day in selections:
        timestamp = np.datetime64(valid_date)
        try:
            part = ds.sel(valid_time=timestamp, lead_day=lead_day)
        except KeyError as exc:
            raise KeyError(
                f"No AROME previous-run value for valid_date={valid_date}, lead_day={lead_day}"
            ) from exc

        # Restore a length-one valid_time dimension and remove scalar lead_day.
        part = part.drop_vars("lead_day", errors="ignore").expand_dims(valid_time=[timestamp])
        parts.append(part)

    result = xr.concat(
        parts,
        dim="valid_time",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="exact",
    ).sortby("valid_time")
    result.attrs.update(ds.attrs)
    result.attrs["selected_fixed_leads"] = ";".join(
        f"{valid.isoformat()}:D+{lead}" for valid, lead in selections
    )
    return result
