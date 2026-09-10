"""Historical ECMWF IFS HRES single-run archive using Open-Meteo.

Open-Meteo's Single Runs API preserves complete ECMWF IFS HRES forecast
initialisations.  Native 9 km HRES runs are available from March 2024 and are
therefore suitable for IrriGator retrospective/backtesting experiments.

This client is intentionally different from the former AROME Previous-Runs
client:

* each request corresponds to one exact forecast initialisation (``run``);
* the default model is ECMWF IFS HRES at native ~9 km resolution;
* the default horizon is 48 hours, matching the short-range deterministic
  segment before the TIGGE ensemble hand-off;
* data are cached per parcel and per forecast run;
* Open-Meteo response units are checked and converted into IrriGator's
  normalized daily schema::

      t_min, t_max, t_mean, dewpoint          [degC]
      wind_speed_10m                          [m/s]
      pressure_kpa                            [kPa]
      precip_mm                               [mm/day]
      rs_mj                                   [MJ/m2/day]

Open-Meteo's default wind unit is km/h, but IrriGator internally uses m/s
because the downstream 10 m -> 2 m conversion and FAO-56 ET0 calculation
expect m/s.  This client therefore explicitly requests ``wind_speed_unit=ms``.

Hourly ``shortwave_radiation`` is an average flux over the preceding hour
[W/m2].  For hourly data, daily incoming solar radiation is therefore

    Rs [MJ/m2/day] = sum(hourly W/m2) * 3600 / 1e6
                   = sum(hourly W/m2) * 0.0036.

The API is queried with ``timezone=GMT`` so run-relative timestamps remain
unambiguous, and with ``elevation=nan`` / ``cell_selection=nearest`` to disable
Open-Meteo's elevation downscaling and retain a model-grid-cell style forcing.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr

from irrigator.config import ParcelConfig


logger = logging.getLogger(__name__)

SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
DEFAULT_PROCESSED_DIR = Path("data/processed")

# Open-Meteo identifier for native ~9 km ECMWF IFS HRES.
DEFAULT_MODEL = "ecmwf_ifs"

# Open-Meteo documents native IFS HRES single runs from 2024-03-14.
HRES_ARCHIVE_START = date(2024, 3, 14)

# Required hourly fields.  These are converted to the same daily variable names
# used by the existing ERA5 / AROME / IFS processing pipelines.
HOURLY_VARIABLES = (
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation",
    "shortwave_radiation",
)

RUN_HOURS = {0, 6, 12, 18}


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def _run_timestamp(run_date: date, run_hour: int) -> str:
    """Return Open-Meteo run identifier, e.g. ``2025-06-01T00:00``."""
    if run_hour not in RUN_HOURS:
        raise ValueError(
            f"run_hour must be one of {sorted(RUN_HOURS)} for IFS HRES; got {run_hour}"
        )
    return f"{run_date.isoformat()}T{run_hour:02d}:00"


def _archive_path(
    parcel_id: str,
    run_date: date,
    run_hour: int = 0,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Return the cache path for one parcel and one exact HRES run."""
    out_dir = Path(processed_dir) / "hres" / "historical" / parcel_id
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / (f"hres_{_safe_model_name(model)}_{run_date.isoformat()}_{run_hour:02d}z.nc")


def _request_hres_run(
    run_date: date,
    *,
    run_hour: int,
    lat: float,
    lon: float,
    forecast_hours: int,
    model: str,
    timeout: float,
    session: requests.Session,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Request one exact ECMWF IFS HRES forecast run from Open-Meteo."""
    if run_date < HRES_ARCHIVE_START:
        raise ValueError(
            f"Open-Meteo IFS HRES single-run archive starts on "
            f"{HRES_ARCHIVE_START}; requested {run_date}"
        )
    if forecast_hours < 1:
        raise ValueError("forecast_hours must be >= 1")

    run = _run_timestamp(run_date, run_hour)

    params: dict[str, Any] = {
        "latitude": float(lat),
        "longitude": float(lon),
        "models": model,
        "run": run,
        "forecast_hours": int(forecast_hours),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",
        "temperature_unit": "celsius",
        # IrriGator normalizes wind to m/s.  Open-Meteo otherwise defaults km/h.
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        # Disable Open-Meteo statistical elevation downscaling.
        "elevation": "nan",
        "cell_selection": "nearest",
    }

    if api_key is not None:
        params["apikey"] = api_key

    logger.info(
        "Open-Meteo HRES single run: run=%s, parcel=(%.4f, %.4f), forecast_hours=%d, model=%s",
        run,
        lat,
        lon,
        forecast_hours,
        model,
    )

    response = session.get(
        SINGLE_RUN_URL,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason', payload)}")

    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise RuntimeError(f"Unexpected Open-Meteo response: keys={list(payload)}")

    return payload


def _series(
    payload: dict[str, Any],
    name: str,
    time_index: pd.DatetimeIndex,
) -> pd.Series:
    values = payload["hourly"].get(name)
    if values is None:
        raise RuntimeError(
            f"Open-Meteo HRES response is missing required variable {name!r}. "
            f"Available hourly variables: {sorted(payload['hourly'])}"
        )

    if len(values) != len(time_index):
        raise RuntimeError(
            f"Length mismatch for {name}: {len(values)} values for {len(time_index)} timestamps"
        )

    return pd.Series(
        pd.to_numeric(values, errors="coerce"),
        index=time_index,
        dtype=float,
        name=name,
    )


def _normalise_unit_text(unit: str | None) -> str:
    if unit is None:
        return ""
    return str(unit).strip().lower().replace("²", "2").replace("^", "").replace(" ", "")


def _temperature_to_celsius(
    values: pd.Series,
    unit: str | None,
    variable: str,
) -> pd.Series:
    u = _normalise_unit_text(unit)

    if u in {"°c", "c", "celsius"}:
        return values
    if u in {"°f", "f", "fahrenheit"}:
        return (values - 32.0) * (5.0 / 9.0)

    raise ValueError(f"Unexpected Open-Meteo unit for {variable}: {unit!r}")


def _wind_to_ms(
    values: pd.Series,
    unit: str | None,
) -> pd.Series:
    """Normalize Open-Meteo wind speed to m/s."""
    u = _normalise_unit_text(unit)

    if u in {"m/s", "ms-1", "m/s-1", "ms"}:
        return values
    if u in {"km/h", "kmh", "kmh-1"}:
        return values / 3.6
    if u in {"mph"}:
        return values * 0.44704
    if u in {"kn", "kt", "knot", "knots"}:
        return values * 0.514444

    raise ValueError(f"Unexpected Open-Meteo unit for wind_speed_10m: {unit!r}")


def _pressure_to_kpa(
    values: pd.Series,
    unit: str | None,
) -> pd.Series:
    """Normalize Open-Meteo surface pressure to kPa."""
    u = _normalise_unit_text(unit)

    if u in {"hpa", "mbar"}:
        return values / 10.0
    if u in {"pa"}:
        return values / 1000.0
    if u in {"kpa"}:
        return values

    raise ValueError(f"Unexpected Open-Meteo unit for surface_pressure: {unit!r}")


def _precip_to_mm(
    values: pd.Series,
    unit: str | None,
) -> pd.Series:
    u = _normalise_unit_text(unit)

    if u in {"mm"}:
        return values
    if u in {"inch", "in", "inches"}:
        return values * 25.4

    raise ValueError(f"Unexpected Open-Meteo unit for precipitation: {unit!r}")


def _shortwave_to_mj_per_hour(
    values: pd.Series,
    unit: str | None,
) -> pd.Series:
    """Convert hourly shortwave values to MJ/m2 for each timestamp interval.

    Open-Meteo's ``shortwave_radiation`` is normally W/m2 averaged over the
    preceding hour.  For a one-hour interval, multiply by 3600 s and convert
    J -> MJ.

    A J/m2 fallback is included defensively in case the API representation
    changes in the future.
    """
    u = _normalise_unit_text(unit)

    if u in {
        "w/m2",
        "wm-2",
        "watt/m2",
        "wattperm2",
    }:
        return values * 0.0036

    if u in {
        "j/m2",
        "jm-2",
    }:
        return values / 1e6

    if u in {
        "mj/m2",
        "mjm-2",
    }:
        return values

    raise ValueError(f"Unexpected Open-Meteo unit for shortwave_radiation: {unit!r}")


def _payload_to_daily(
    payload: dict[str, Any],
    *,
    run_date: date,
    run_hour: int,
    requested_lat: float,
    requested_lon: float,
    model: str,
) -> xr.Dataset:
    """Aggregate one exact HRES run to IrriGator daily forcing fields."""
    times = pd.to_datetime(payload["hourly"]["time"], utc=True).tz_convert(None)

    if len(times) == 0:
        raise RuntimeError("Open-Meteo returned an empty hourly time axis")

    units = payload.get("hourly_units", {})

    temperature = _temperature_to_celsius(
        _series(payload, "temperature_2m", times),
        units.get("temperature_2m"),
        "temperature_2m",
    )
    dewpoint = _temperature_to_celsius(
        _series(payload, "dew_point_2m", times),
        units.get("dew_point_2m"),
        "dew_point_2m",
    )
    wind_ms = _wind_to_ms(
        _series(payload, "wind_speed_10m", times),
        units.get("wind_speed_10m"),
    )
    pressure_kpa = _pressure_to_kpa(
        _series(payload, "surface_pressure", times),
        units.get("surface_pressure"),
    )
    precip_mm = _precip_to_mm(
        _series(payload, "precipitation", times),
        units.get("precipitation"),
    )
    shortwave_mj_hour = _shortwave_to_mj_per_hour(
        _series(payload, "shortwave_radiation", times),
        units.get("shortwave_radiation"),
    )

    # Require complete hourly coverage for each retained daily value.
    # With the default 00Z + 48 h request, this yields exactly issue_date and
    # issue_date + 1 as two complete calendar days.
    hourly = pd.DataFrame(
        {
            "temperature": temperature,
            "dewpoint": dewpoint,
            "wind_ms": wind_ms,
            "pressure_kpa": pressure_kpa,
            "precip_mm": precip_mm,
            "shortwave_mj": shortwave_mj_hour,
        },
        index=times,
    )

    grouped = hourly.resample("1D")

    frame = pd.DataFrame(
        {
            "t_min": grouped["temperature"].min(),
            "t_max": grouped["temperature"].max(),
            "t_mean": grouped["temperature"].mean(),
            "dewpoint": grouped["dewpoint"].mean(),
            "wind_speed_10m": grouped["wind_ms"].mean(),
            "pressure_kpa": grouped["pressure_kpa"].mean(),
            "precip_mm": grouped["precip_mm"].sum(min_count=1),
            "rs_mj": grouped["shortwave_mj"].sum(min_count=1),
            "_n_temperature": grouped["temperature"].count(),
            "_n_dewpoint": grouped["dewpoint"].count(),
            "_n_wind": grouped["wind_ms"].count(),
            "_n_pressure": grouped["pressure_kpa"].count(),
            "_n_precip": grouped["precip_mm"].count(),
            "_n_shortwave": grouped["shortwave_mj"].count(),
        }
    )

    # Keep only complete 24-hour calendar days.  This avoids silently using
    # partial days for non-00Z initialisations.  Users wanting 06/12/18Z runs
    # can request enough hours to span complete subsequent UTC days.
    count_columns = [c for c in frame.columns if c.startswith("_n_")]
    complete = (frame[count_columns] == 24).all(axis=1)
    frame = frame.loc[complete].drop(columns=count_columns)

    if frame.empty:
        raise RuntimeError(
            "No complete 24-hour UTC days were available in the requested "
            "HRES run. Increase forecast_hours or use a 00Z run."
        )

    frame.index.name = "valid_time"

    ds = xr.Dataset.from_dataframe(frame)

    used_lat = float(payload.get("latitude", np.nan))
    used_lon = float(payload.get("longitude", np.nan))

    # Preserve the same length-one spatial dimensions used by the existing
    # parcel forcing extraction pathway.
    ds = ds.expand_dims(
        latitude=[used_lat],
        longitude=[used_lon],
    )
    ds = ds.transpose(
        "valid_time",
        "latitude",
        "longitude",
        ...,
    )

    # Explicit variable units in the normalized IrriGator schema.
    attrs = {
        "t_min": {"units": "degC"},
        "t_max": {"units": "degC"},
        "t_mean": {"units": "degC"},
        "dewpoint": {"units": "degC"},
        "wind_speed_10m": {"units": "m s-1"},
        "pressure_kpa": {"units": "kPa"},
        "precip_mm": {"units": "mm day-1"},
        "rs_mj": {"units": "MJ m-2 day-1"},
    }
    for variable, variable_attrs in attrs.items():
        ds[variable].attrs.update(variable_attrs)

    ds.attrs.update(
        {
            "source": "Open-Meteo Single Runs - ECMWF IFS HRES",
            "archive_semantics": "exact forecast initialisation",
            "model": model,
            "forecast_reference_date": run_date.isoformat(),
            "forecast_reference_hour_utc": int(run_hour),
            "forecast_reference_time": _run_timestamp(run_date, run_hour),
            "requested_latitude": float(requested_lat),
            "requested_longitude": float(requested_lon),
            "selected_grid_latitude": used_lat,
            "selected_grid_longitude": used_lon,
            "selected_grid_elevation_m": float(payload.get("elevation", np.nan)),
            "open_meteo_elevation_downscaling": ("disabled (elevation=nan)"),
            "cell_selection": "nearest",
            "temperature_source_unit": str(units.get("temperature_2m", "")),
            "wind_source_unit": str(units.get("wind_speed_10m", "")),
            "pressure_source_unit": str(units.get("surface_pressure", "")),
            "precipitation_source_unit": str(units.get("precipitation", "")),
            "radiation_source_unit": str(units.get("shortwave_radiation", "")),
            "radiation_conversion": (
                "hourly Open-Meteo shortwave_radiation W m-2 "
                "x 3600 / 1e6 -> MJ m-2 per hour; daily sum"
            ),
        }
    )

    return ds


def fetch_hres_run(
    run_date: date,
    *,
    lat: float,
    lon: float,
    run_hour: int = 0,
    forecast_hours: int = 48,
    model: str = DEFAULT_MODEL,
    timeout: float = 120.0,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> xr.Dataset:
    """Fetch one exact historical ECMWF IFS HRES run and aggregate daily.

    Parameters
    ----------
    run_date
        Forecast initialisation date.
    lat, lon
        Parcel coordinates.
    run_hour
        HRES initialisation hour. One of 00, 06, 12 or 18 UTC.
    forecast_hours
        Number of hourly values requested. Default 48 gives two complete UTC
        days for a 00Z run and matches the deterministic short-range segment
        before the TIGGE hand-off.
    model
        Open-Meteo model identifier. ``ecmwf_ifs`` selects native ~9 km HRES.
    timeout
        HTTP timeout in seconds.
    api_key
        Optional Open-Meteo API key for commercial/customer endpoints.
    session
        Optional requests session.
    """
    own_session = session is None
    session = session or requests.Session()

    try:
        payload = _request_hres_run(
            run_date,
            run_hour=run_hour,
            lat=lat,
            lon=lon,
            forecast_hours=forecast_hours,
            model=model,
            timeout=timeout,
            session=session,
            api_key=api_key,
        )

        return _payload_to_daily(
            payload,
            run_date=run_date,
            run_hour=run_hour,
            requested_lat=lat,
            requested_lon=lon,
            model=model,
        )
    finally:
        if own_session:
            session.close()


def archive_hres_run(
    parcel: ParcelConfig,
    run_date: date,
    *,
    run_hour: int = 0,
    forecast_hours: int = 48,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    model: str = DEFAULT_MODEL,
    timeout: float = 120.0,
    api_key: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download/process/cache one exact HRES run for one parcel."""
    path = _archive_path(
        parcel.id,
        run_date,
        run_hour,
        processed_dir=processed_dir,
        model=model,
    )

    if path.exists() and not overwrite:
        logger.info("Historical HRES cache already exists: %s", path)
        return path

    ds = fetch_hres_run(
        run_date,
        lat=parcel.lat,
        lon=parcel.lon,
        run_hour=run_hour,
        forecast_hours=forecast_hours,
        model=model,
        timeout=timeout,
        api_key=api_key,
    )

    ds.attrs["parcel_id"] = parcel.id
    ds.attrs["parcel_name"] = parcel.name

    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
        for name in ds.data_vars
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=encoding)
    logger.info(
        "Saved historical HRES run: %s (%.1f MB)",
        path,
        path.stat().st_size / 1e6,
    )
    ds.close()

    return path


def load_hres_run(
    parcel_id: str,
    run_date: date,
    *,
    run_hour: int = 0,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    model: str = DEFAULT_MODEL,
) -> xr.Dataset:
    """Load a cached parcel-level HRES forecast run."""
    path = _archive_path(
        parcel_id,
        run_date,
        run_hour,
        processed_dir=processed_dir,
        model=model,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Historical HRES cache not found: {path}. "
            "Run archive_hres_run(parcel, run_date) first."
        )

    return xr.open_dataset(path)


def archive_hres_runs(
    parcel: ParcelConfig,
    start_date: date,
    end_date: date,
    *,
    run_hour: int = 0,
    forecast_hours: int = 48,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    model: str = DEFAULT_MODEL,
    timeout: float = 120.0,
    api_key: str | None = None,
    overwrite: bool = False,
    max_workers: int = 1,
) -> list[Path]:
    """Archive a daily sequence of exact HRES runs for one parcel.

    With ``run_hour=0`` and ``forecast_hours=48``, each cache contains the two
    complete UTC calendar days used as the deterministic D+0/D+1 segment.

    ``max_workers`` may be increased modestly for concurrent HTTP requests.
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    dates: list[date] = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    def _archive_one(run_date: date) -> Path:
        return archive_hres_run(
            parcel,
            run_date,
            run_hour=run_hour,
            forecast_hours=forecast_hours,
            processed_dir=processed_dir,
            model=model,
            timeout=timeout,
            api_key=api_key,
            overwrite=overwrite,
        )

    if max_workers == 1:
        return [_archive_one(run_date) for run_date in dates]

    completed: dict[date, Path] = {}
    failures: dict[date, Exception] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_archive_one, run_date): run_date for run_date in dates}

        for future in as_completed(futures):
            run_date = futures[future]
            try:
                completed[run_date] = future.result()
                logger.info(
                    "Completed HRES run %s (%d/%d)",
                    run_date,
                    len(completed),
                    len(dates),
                )
            except Exception as exc:
                failures[run_date] = exc
                logger.exception(
                    "Historical HRES retrieval failed for %s",
                    run_date,
                )

    if failures:
        failed_dates = ", ".join(d.isoformat() for d in sorted(failures))
        raise RuntimeError(
            f"{len(failures)} HRES run(s) failed: {failed_dates}. "
            "Existing successful cache files were kept; rerun with "
            "overwrite=False to retry only missing dates."
        )

    return [completed[run_date] for run_date in dates]


def select_hres_daily_window(
    ds: xr.Dataset,
    *,
    start_date: date | None = None,
    n_days: int = 2,
) -> xr.Dataset:
    """Select the deterministic daily HRES segment used before TIGGE.

    By default, select the first two complete daily values from the cached run.
    ``start_date`` can be supplied explicitly for defensive backtest logic.
    """
    if n_days < 1:
        raise ValueError("n_days must be >= 1")

    if "valid_time" not in ds.dims:
        raise ValueError("HRES dataset has no valid_time dimension")

    result = ds

    if start_date is not None:
        start = np.datetime64(start_date)
        result = result.sel(valid_time=slice(start, None))

    result = result.isel(valid_time=slice(0, n_days))

    if result.sizes.get("valid_time", 0) < n_days:
        raise ValueError(
            f"Requested {n_days} HRES day(s), but only "
            f"{result.sizes.get('valid_time', 0)} are available"
        )

    return result
