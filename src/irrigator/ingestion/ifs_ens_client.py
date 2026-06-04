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
import xarray as xr

from irrigator.config import RegionConfig

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _output_dir(cfg: RegionConfig) -> Path:
    out = cfg.raw_dir / "ifs_ens"
    out.mkdir(parents=True, exist_ok=True)
    return out


def fetch_ifs_ens(
    cfg: RegionConfig,
    run_date: date | None = None,
    run_hour: int = 0,
    steps: list[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Download IFS ENS forecast from ECMWF open data.

    Parameters
    ----------
    cfg : RegionConfig
    run_date : forecast initialization date (default: today)
    run_hour : initialization hour (0 or 12)
    steps : forecast hours to retrieve (default: 0 to 360 by 6)
    overwrite : re-download if exists

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
        # 0 to 360h by 6h = 15 days
        steps = list(range(0, 366, 6))

    out_dir = _output_dir(cfg)
    out_path = out_dir / f"ifs_ens_{run_date.isoformat()}_{run_hour:02d}z.grib2"

    if out_path.exists() and not overwrite:
        logger.info("IFS ENS already exists: %s", out_path)
        return out_path

    logger.info(
        "Downloading IFS ENS: %s %02dZ, %d steps, %d params",
        run_date,
        run_hour,
        len(steps),
        len(IFS_ENS_PARAMS),
    )

    client = Client(source="ecmwf")

    # Area clipping for the region
    bbox = cfg.bbox_wgs84
    # Add buffer for interpolation
    area = [bbox.north + 1, bbox.west - 1, bbox.south - 1, bbox.east + 1]

    client.retrieve(
        type="ef",  # ensemble forecast
        date=run_date.isoformat(),
        time=run_hour,
        param=IFS_ENS_PARAMS,
        step=steps,
        area=area,
        target=str(out_path),
    )

    logger.info(
        "Downloaded IFS ENS: %s (%.1f MB)",
        out_path,
        out_path.stat().st_size / 1e6,
    )
    return out_path


def fetch_latest_ifs_ens(cfg: RegionConfig) -> Path | None:
    """Fetch the most recent available IFS ENS run.

    Tries today 00Z, then yesterday 12Z, then yesterday 00Z.
    """
    today = date.today()
    yesterday = today - timedelta(days=1)

    for run_date, run_hour in [(today, 0), (yesterday, 12), (yesterday, 0)]:
        try:
            return fetch_ifs_ens(cfg, run_date, run_hour)
        except Exception as exc:
            logger.debug("IFS ENS %s %02dZ not available: %s", run_date, run_hour, exc)
            continue

    logger.warning("Could not fetch any recent IFS ENS run")
    return None


# ---------------------------------------------------------------------------
# Open and process to daily xr.Dataset
# ---------------------------------------------------------------------------


def open_ifs_ens(path: Path) -> xr.Dataset:
    """Open IFS ENS GRIB2 file as xarray Dataset.

    Requires cfgrib engine: pip install cfgrib eccodes

    Returns dataset with dims (number, step, latitude, longitude)
    where 'number' is the ensemble member (0-50, 0=control).
    """
    # cfgrib may split surface and pressure-level vars into separate datasets
    # Use backend_kwargs to handle this
    datasets = []
    try:
        # Try opening as single dataset first
        ds = xr.open_dataset(path, engine="cfgrib")
        datasets.append(ds)
    except Exception:
        # Multiple parameter types — open with filter_by_keys
        for param_type in ["sfc", "pl"]:
            try:
                ds = xr.open_dataset(
                    path,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
                )
                datasets.append(ds)
            except Exception:
                pass

    if not datasets:
        raise RuntimeError(f"Could not open IFS ENS file: {path}")

    return xr.merge(datasets) if len(datasets) > 1 else datasets[0]


def process_ifs_ens_to_daily(ds: xr.Dataset) -> xr.Dataset:
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

    # Handle accumulated variables (tp, ssrd) — diff to get per-step values
    for acc_var in ["tp", "ssrd"]:
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
        daily_vars["precip_mm"] = ds["tp"].resample({time_dim: "1D"}).sum() * 1000.0
        daily_vars["precip_mm"] = daily_vars["precip_mm"].clip(min=0)

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
