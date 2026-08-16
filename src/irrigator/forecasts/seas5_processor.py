"""SEAS5 monthly preprocessing and first-order bias correction.

The seasonal pipeline works in the same variable domain as processed
ERA5-Land monthly fields:

    t_min, t_mean, t_max, dewpoint, wind_speed_10m, pressure_kpa,
    rs_mj, precip_mm

For absolute SEAS5 monthly statistics, the recommended correction is based on
SEAS5 hindcasts for the same initialization month and lead time:

    additive variables:
        X_corrected = X_forecast - clim_SEAS5 + clim_ERA5

    positive-scale variables:
        X_corrected = X_forecast * clim_ERA5 / clim_SEAS5

This removes the first-order model climatology bias while preserving the
forecast anomaly.  The legacy path for already-postprocessed anomaly datasets
is retained for backwards compatibility.
"""

from __future__ import annotations

import calendar
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# SEAS5 CDS lead convention used by the existing IrriGator ingestion code:
# lead 1 is the initialization month.
LEAD_OFFSET = 0

# Canonical IrriGator monthly variables used by PCA / analog matching.
SEAS5_MATCHING_VARIABLES = [
    "t_min",
    "t_mean",
    "t_max",
    "dewpoint",
    "precip_mm",
    "rs_mj",
    "wind_speed_10m",
]

# ERA5 variables that can additionally be useful for ET0 reconstruction.
SEAS5_OPTIONAL_VARIABLES = ["pressure_kpa"]

# Raw NetCDF aliases seen in CDS/ECMWF encodings.
_RAW_ALIASES: dict[str, tuple[str, ...]] = {
    "t2m": ("t2m", "2m_temperature"),
    "mn2t24": (
        "mn2t24",
        "mn2t",
        "minimum_2m_temperature_in_the_last_24_hours",
        "minimum_2m_temperature_since_previous_post_processing",
    ),
    "mx2t24": (
        "mx2t24",
        "mx2t",
        "maximum_2m_temperature_in_the_last_24_hours",
        "maximum_2m_temperature_since_previous_post_processing",
    ),
    "d2m": ("d2m", "2m_dewpoint_temperature"),
    "u10": ("u10", "10m_u_component_of_wind"),
    "v10": ("v10", "10m_v_component_of_wind"),
    "si10": ("si10", "10si", "10m_wind_speed", "wind_speed_10m"),
    "sp": ("sp", "surface_pressure"),
    "ssrd": (
        "ssrd",
        "msdwswrf",
        "avg_sdswrf",
        "surface_solar_radiation_downwards",
        "mean_surface_downward_short_wave_radiation_flux",
    ),
    "tp": (
        "tp",
        "tprate",
        "avg_tprate",
        "total_precipitation",
        "mean_total_precipitation_rate",
    ),
}

_ADDITIVE_VARS = {"t_min", "t_mean", "t_max", "dewpoint", "pressure_kpa"}
_MULTIPLICATIVE_VARS = {"precip_mm", "rs_mj", "wind_speed_10m"}


def compute_valid_month(init_month: int, leadtime_month: int) -> int:
    """Compute target calendar month for one SEAS5 lead."""
    return ((init_month - 1) + (leadtime_month - 1) + LEAD_OFFSET) % 12 + 1


def compute_valid_year(init_year: int, init_month: int, leadtime_month: int) -> int:
    """Compute target year for one SEAS5 lead."""
    zero_based = (init_month - 1) + (leadtime_month - 1) + LEAD_OFFSET
    return init_year + zero_based // 12


def _find_coord_or_dim(ds: xr.Dataset, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in ds.dims or name in ds.coords:
            return name
    return None


def _find_var(ds: xr.Dataset, aliases: Iterable[str]) -> str | None:
    for name in aliases:
        if name in ds.data_vars:
            return name
    return None


def _standardize_spatial_names(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    if "lat" in ds.dims or "lat" in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.dims or "lon" in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def _units(da: xr.DataArray) -> str:
    return str(da.attrs.get("units", "")).strip().lower().replace("**", "^")


def _temperature_to_c(da: xr.DataArray) -> xr.DataArray:
    units = _units(da)
    # CDS temperature is K.  The value-based fallback keeps synthetic tests and
    # manually preprocessed files convenient.
    if "k" == units or "kelvin" in units or float(da.mean(skipna=True)) > 150:
        return da - 273.15
    return da


def _pressure_to_kpa(da: xr.DataArray) -> xr.DataArray:
    units = _units(da)
    if units in {"pa", "pascal", "pascals"} or float(da.mean(skipna=True)) > 2_000:
        return da / 1000.0
    return da


def _radiation_to_mj_day(da: xr.DataArray) -> xr.DataArray:
    """Convert monthly-mean solar-radiation statistic to MJ m-2 day-1."""
    units = _units(da)
    # Seasonal monthly statistics are normally W m-2.  Some manually prepared
    # files may already use MJ m-2 day-1.
    if "w" in units and "m" in units:
        return da * 0.0864  # W m-2 * 86400 s/day / 1e6
    if "j" in units and "m" in units:
        return da / 1e6
    return da


def _precip_to_monthly_mm(
    da: xr.DataArray,
    *,
    init_year: int,
    init_month: int,
    lead_dim: str,
) -> xr.DataArray:
    """Convert SEAS5 monthly precipitation statistic to a monthly total [mm]."""
    units = _units(da)
    pieces = []
    for lead in da[lead_dim].values:
        lead_int = int(lead)
        year = compute_valid_year(init_year, init_month, lead_int)
        month = compute_valid_month(init_month, lead_int)
        n_days = calendar.monthrange(year, month)[1]
        part = da.sel({lead_dim: lead})

        if ("m s" in units or "m/s" in units) and "kg" not in units:
            # metres of water per second -> monthly mm
            part = part * 1000.0 * 86400.0 * n_days
        elif "kg" in units and "s" in units:
            # kg m-2 s-1 = mm s-1
            part = part * 86400.0 * n_days
        elif ("m/day" in units or "m day" in units or "m d-1" in units) and "mm" not in units:
            part = part * 1000.0 * n_days
        elif units in {"m", "metre", "meter", "metres", "meters"}:
            part = part * 1000.0
        elif "mm/day" in units or "mm d" in units:
            part = part * n_days
        # Otherwise assume the caller already supplied a monthly total in mm.
        pieces.append(part)

    return xr.concat(pieces, dim=da[lead_dim])


def prepare_seas5_monthly(
    ds: xr.Dataset,
    *,
    init_year: int,
    init_month: int,
) -> xr.Dataset:
    """Convert raw CDS monthly SEAS5 data to IrriGator variable names/units.

    The returned dataset keeps the original ensemble and lead dimensions, adds
    a ``valid_time`` coordinate along the lead dimension, and represents
    precipitation as a monthly total while the other matching variables are
    monthly means.
    """
    ds = _standardize_spatial_names(ds)
    lead_dim = _find_coord_or_dim(
        ds, ("forecastMonth", "forecast_month", "leadtime_month", "lead", "lead_time")
    )
    if lead_dim is None:
        raise ValueError("Could not find a SEAS5 lead-month dimension.")

    t2m = _find_var(ds, _RAW_ALIASES["t2m"])
    if t2m is None:
        raise ValueError("SEAS5 dataset is missing 2m temperature.")

    out: dict[str, xr.DataArray] = {"t_mean": _temperature_to_c(ds[t2m])}

    mn2t = _find_var(ds, _RAW_ALIASES["mn2t24"])
    if mn2t is not None:
        out["t_min"] = _temperature_to_c(ds[mn2t])
    else:
        logger.warning("SEAS5 has no monthly minimum 2m temperature; t_min omitted.")

    mx2t = _find_var(ds, _RAW_ALIASES["mx2t24"])
    if mx2t is not None:
        out["t_max"] = _temperature_to_c(ds[mx2t])
    else:
        # Keep the pipeline usable with old two-variable downloads.  Matching
        # will automatically use only variables shared with the forecast.
        logger.warning("SEAS5 has no monthly maximum 2m temperature; t_max omitted.")

    d2m = _find_var(ds, _RAW_ALIASES["d2m"])
    if d2m is not None:
        out["dewpoint"] = _temperature_to_c(ds[d2m])

    si10 = _find_var(ds, _RAW_ALIASES["si10"])
    if si10 is not None:
        out["wind_speed_10m"] = ds[si10].clip(min=0)
    else:
        u10 = _find_var(ds, _RAW_ALIASES["u10"])
        v10 = _find_var(ds, _RAW_ALIASES["v10"])
        if u10 is not None and v10 is not None:
            out["wind_speed_10m"] = np.hypot(ds[u10], ds[v10])

    sp = _find_var(ds, _RAW_ALIASES["sp"])
    if sp is not None:
        out["pressure_kpa"] = _pressure_to_kpa(ds[sp])

    ssrd = _find_var(ds, _RAW_ALIASES["ssrd"])
    if ssrd is not None:
        out["rs_mj"] = _radiation_to_mj_day(ds[ssrd]).clip(min=0)

    tp = _find_var(ds, _RAW_ALIASES["tp"])
    if tp is not None:
        out["precip_mm"] = _precip_to_monthly_mm(
            ds[tp], init_year=init_year, init_month=init_month, lead_dim=lead_dim
        ).clip(min=0)

    result = xr.Dataset(out)
    valid_times = []
    for lead in result[lead_dim].values:
        year = compute_valid_year(init_year, init_month, int(lead))
        month = compute_valid_month(init_month, int(lead))
        valid_times.append(np.datetime64(f"{year:04d}-{month:02d}-01"))
    result = result.assign_coords(valid_time=(lead_dim, valid_times))

    result.attrs.update(
        {
            "source": "SEAS5 monthly statistics",
            "init_year": int(init_year),
            "init_month": int(init_month),
            "precip_mm_semantics": "monthly_total",
            "other_variable_semantics": "monthly_mean",
        }
    )
    return result


def open_era5_daily_archive(
    processed_dir: str | Path,
    start_year: int,
    end_year: int,
) -> xr.Dataset:
    """Open an annual processed ERA5-Land daily archive lazily.

    All requested annual files must exist.  The returned multi-file dataset is
    suitable both for seasonal PCA analog construction and for the matching
    ERA5 climatology used in SEAS5 bias correction.
    """
    processed_dir = Path(processed_dir)
    files = [
        processed_dir / "atmospheric" / f"era5_daily_{year}.nc"
        for year in range(start_year, end_year + 1)
    ]
    missing = [p for p in files if not p.exists()]
    if missing:
        preview = ", ".join(str(p) for p in missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} ERA5 daily climatology file(s), e.g. {preview}. "
            "The requested climatology requires every year in the reference period."
        )
    return xr.open_mfdataset(files, combine="by_coords")


def build_era5_monthly_climatology(
    start_year: int = 1993,
    end_year: int = 2016,
    processed_dir: str | Path = "data/processed",
    *,
    era5_daily: xr.Dataset | None = None,
) -> xr.Dataset:
    """Build a 12-month ERA5-Land climatology in IrriGator units.

    Precipitation is aggregated as a monthly total before climatological
    averaging; all other variables are monthly means.
    """
    daily = era5_daily
    if daily is None:
        daily = open_era5_daily_archive(processed_dir, start_year, end_year)

    time_dim = "valid_time" if "valid_time" in daily.dims else "time"
    daily = daily.sel({time_dim: slice(f"{start_year}-01-01", f"{end_year}-12-31")})

    monthly_vars: dict[str, xr.DataArray] = {}
    for var in daily.data_vars:
        if var == "precip_mm":
            monthly_vars[var] = daily[var].resample({time_dim: "1MS"}).sum()
        else:
            monthly_vars[var] = daily[var].resample({time_dim: "1MS"}).mean()
    monthly = xr.Dataset(monthly_vars)
    clim = monthly.groupby(f"{time_dim}.month").mean(time_dim)
    clim.attrs.update(
        {
            "climatology_period": f"{start_year}-{end_year}",
            "source": "ERA5-Land monthly climatology",
            "precip_mm_semantics": "monthly_total",
        }
    )
    return clim


def build_seas5_hindcast_climatology(
    *,
    raw_dir: str | Path,
    init_month: int,
    start_year: int = 1993,
    end_year: int = 2016,
) -> xr.Dataset:
    """Build SEAS5 model climatology from downloaded hindcast initializations.

    This function intentionally does not download 24 years automatically.  Use
    ``fetch_seas5_hindcasts`` from ``ingestion.cds_client`` first, then build
    the climatology locally.  The result retains lead and spatial dimensions
    and averages over hindcast years and ensemble members.
    """
    from irrigator.ingestion.cds_client import open_seas5

    prepared = []
    for year in range(start_year, end_year + 1):
        raw = open_seas5(raw_dir=raw_dir, year=year, month=init_month)
        p = prepare_seas5_monthly(raw, init_year=year, init_month=init_month)
        p = p.expand_dims(hindcast_year=[year])
        prepared.append(p)

    if not prepared:
        raise ValueError("No SEAS5 hindcasts supplied.")

    all_hindcasts = xr.concat(prepared, dim="hindcast_year", join="inner")
    member_dim = _find_coord_or_dim(all_hindcasts, ("number", "member", "realization"))
    reduce_dims = ["hindcast_year"] + ([member_dim] if member_dim else [])
    clim = all_hindcasts.mean(dim=reduce_dims, skipna=True)
    clim.attrs.update(
        {
            "source": "SEAS5 hindcast climatology",
            "hindcast_period": f"{start_year}-{end_year}",
            "init_month": int(init_month),
        }
    )
    return clim


def _align_era5_clim_to_forecast(
    era5_climatology: xr.Dataset,
    forecast_da: xr.DataArray,
    *,
    valid_month: int,
) -> xr.DataArray:
    if "month" not in era5_climatology.dims:
        raise ValueError("ERA5 climatology must have a 'month' dimension.")
    clim = era5_climatology[forecast_da.name].sel(month=valid_month)
    # Harmonize latitude orientation and interpolate from ERA5-Land/coarsened
    # grid to the SEAS5 1° grid.
    if "latitude" in clim.coords and "latitude" in forecast_da.coords:
        if clim.latitude.values[0] < clim.latitude.values[-1]:
            clim = clim.sortby("latitude", ascending=False)
    spatial_template = forecast_da
    extra_dims = [d for d in spatial_template.dims if d not in {"latitude", "longitude"}]
    if extra_dims:
        spatial_template = spatial_template.isel({d: 0 for d in extra_dims}, drop=True)
    if {"latitude", "longitude"}.issubset(clim.dims) and {
        "latitude",
        "longitude",
    }.issubset(spatial_template.dims):
        clim = clim.interp(
            latitude=spatial_template.latitude,
            longitude=spatial_template.longitude,
            method="linear",
        )
    return clim


def correct_seas5_monthly(
    seas5_monthly: xr.Dataset,
    era5_climatology: xr.Dataset,
    init_month: int,
    *,
    init_year: int | None = None,
    seas5_climatology: xr.Dataset | None = None,
    ratio_clip: tuple[float, float] | None = (0.2, 5.0),
) -> xr.Dataset:
    """Bias-correct SEAS5 monthly data into the ERA5-Land climate domain.

    Two input modes are supported:

    1. **Absolute monthly statistics** (preferred): pass a SEAS5 hindcast
       climatology.  Additive delta correction is used for temperatures and
       pressure; ratio correction is used for precipitation, radiation and
       wind.
    2. **Already-postprocessed anomalies**: variables ending in ``_anomaly``
       are added to the matching ERA5 climatology (legacy behavior).

    Raw CDS monthly statistics can be passed directly when ``init_year`` is
    supplied; they are normalized first with :func:`prepare_seas5_monthly`.
    """
    # Detect raw CDS input.  Canonical prepared data already has t_mean.
    if "t_mean" not in seas5_monthly.data_vars and not any(
        str(v).endswith("_anomaly") for v in seas5_monthly.data_vars
    ):
        if init_year is None:
            raise ValueError("init_year is required when correcting raw SEAS5 monthly data.")
        seas5_monthly = prepare_seas5_monthly(
            seas5_monthly, init_year=init_year, init_month=init_month
        )

    lead_dim = _find_coord_or_dim(
        seas5_monthly,
        ("forecastMonth", "forecast_month", "leadtime_month", "lead", "lead_time"),
    )
    if lead_dim is None:
        raise ValueError("No SEAS5 lead-month dimension found.")

    # Legacy anomaly product path.
    anomaly_vars = [v for v in seas5_monthly.data_vars if str(v).endswith("_anomaly")]
    if anomaly_vars:
        corrected: dict[str, xr.DataArray] = {}
        for var in anomaly_vars:
            base = var.removesuffix("_anomaly")
            if base not in era5_climatology.data_vars:
                logger.warning("No ERA5 climatology variable for %s", var)
                continue
            pieces = []
            for lead in seas5_monthly[lead_dim].values:
                month = compute_valid_month(init_month, int(lead))
                anom = seas5_monthly[var].sel({lead_dim: lead})
                clim = era5_climatology[base].sel(month=month, drop=True)
                if {"latitude", "longitude"}.issubset(clim.dims) and {
                    "latitude",
                    "longitude",
                }.issubset(anom.dims):
                    clim = clim.interp(
                        latitude=anom.latitude, longitude=anom.longitude, method="linear"
                    )
                pieces.append(clim + anom)
            corrected[base] = xr.concat(
                pieces, dim=seas5_monthly[lead_dim], coords="minimal", compat="override"
            )
        result = xr.Dataset(corrected)
        result.attrs["bias_correction"] = "ERA5 climatology + SEAS5 postprocessed anomaly"
        return result

    if seas5_climatology is None:
        raise ValueError(
            "Absolute SEAS5 monthly statistics require seas5_climatology. "
            "Build it from 1993-2016 hindcasts with build_seas5_hindcast_climatology()."
        )

    corrected_vars: dict[str, xr.DataArray] = {}
    available = [
        v
        for v in seas5_monthly.data_vars
        if v in era5_climatology.data_vars and v in seas5_climatology.data_vars
    ]
    if not available:
        raise ValueError(
            "No common variables across forecast, SEAS5 climatology and ERA5 climatology."
        )

    for var in available:
        pieces = []
        forecast = seas5_monthly[var]
        model_clim = seas5_climatology[var]
        for lead in seas5_monthly[lead_dim].values:
            lead_int = int(lead)
            valid_month = compute_valid_month(init_month, lead_int)
            fc = forecast.sel({lead_dim: lead})
            mc = model_clim.sel({lead_dim: lead}) if lead_dim in model_clim.dims else model_clim

            era = era5_climatology[var].sel(month=valid_month, drop=True)
            if {"latitude", "longitude"}.issubset(era.dims) and {
                "latitude",
                "longitude",
            }.issubset(fc.dims):
                era = era.interp(latitude=fc.latitude, longitude=fc.longitude, method="linear")
            if {"latitude", "longitude"}.issubset(mc.dims) and {
                "latitude",
                "longitude",
            }.issubset(fc.dims):
                mc = mc.interp(latitude=fc.latitude, longitude=fc.longitude, method="linear")

            if var in _MULTIPLICATIVE_VARS:
                ratio = xr.where(np.abs(mc) > 1e-12, era / mc, 1.0)
                if ratio_clip is not None:
                    ratio = ratio.clip(min=ratio_clip[0], max=ratio_clip[1])
                corr = fc * ratio
                corr = corr.clip(min=0)
            else:
                corr = fc - mc + era
            pieces.append(corr)

        corrected_vars[var] = xr.concat(
            pieces, dim=seas5_monthly[lead_dim], coords="minimal", compat="override"
        )

    result = xr.Dataset(corrected_vars)
    if "valid_time" in seas5_monthly.coords:
        result = result.assign_coords(valid_time=seas5_monthly.valid_time)
    result.attrs.update(seas5_monthly.attrs)
    result.attrs["bias_correction"] = "lead-dependent SEAS5-hindcast -> ERA5 delta/ratio"
    return result
