"""SEAS5 monthly -> daily weather scenarios using France-level analogs.

The seasonal forecast carries useful information at monthly scale, while
AquaCrop needs daily forcing.  IrriGator therefore uses the SEAS5 forecast to
condition a set of historical ERA5-Land daily sequences rather than treating a
months-ahead daily SEAS5 trajectory as deterministic weather.

Pipeline
--------
1. Aggregate historical ERA5-Land to monthly fields and regrid to the SEAS5
   1° France grid.
2. Fit one multivariate PCA per calendar month to the *complete France field*.
3. Project each bias-corrected SEAS5 member/lead into the same PCA space.
4. Rank the closest historical months by Mahalanobis distance and sample from
   the top-k analogs.
5. Extract the selected ERA5-Land month at native resolution and rescale it so
   its monthly statistics match the corrected SEAS5 member.
6. Redate the historical daily sequence to the forecast month.

The output remains an ERA5-Land-like gridded Dataset so the existing
``extract_parcel_forcing`` path performs parcel-level downscaling exactly as it
does for historical data.
"""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from irrigator.config import RegionConfig
from irrigator.forecasts.pca_framework import (
    RegionalPCA,
    fit_monthly_regional_pca,
    rank_regional_analogs,
    transform_to_regional_pca,
)
from irrigator.forecasts.seas5_processor import (
    SEAS5_MATCHING_VARIABLES,
    compute_valid_month,
    compute_valid_year,
)

logger = logging.getLogger(__name__)

# Deliberately excludes soil moisture: AquaCrop already carries the parcel's
# actual soil-water state forward.  These are atmospheric drivers shared by
# processed ERA5-Land and the SEAS5 monthly catalogue.
MATCHING_VARIABLES = list(SEAS5_MATCHING_VARIABLES)
SEAS5_RESOLUTION_DEG = 1.0


@dataclass
class AnalogMatch:
    """Analog selected for one SEAS5 member and lead month."""

    target_month: int
    target_year: int
    analog_year: int
    analog_month: int
    distance: float
    weight: float
    rank: int
    member_id: int

    @property
    def similarity(self) -> float:
        """Backward-compatible inverse-distance diagnostic."""
        return 1.0 / max(self.distance, 1e-12)


@dataclass
class SeasonalScenario:
    """One SEAS5 member represented as a daily gridded weather scenario."""

    member_id: int
    daily_ds: xr.Dataset
    analogs: list[AnalogMatch]


# ---------------------------------------------------------------------------
# Grid / monthly helpers
# ---------------------------------------------------------------------------


def _time_dim(ds: xr.Dataset) -> str:
    for name in ("valid_time", "time"):
        if name in ds.dims:
            return name
    raise ValueError("Dataset has no valid_time/time dimension.")


def _standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "lat" in ds.dims or "lat" in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.dims or "lon" in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    if "latitude" in ds.coords and ds.latitude.values[0] < ds.latitude.values[-1]:
        ds = ds.sortby("latitude", ascending=False)
    return ds


def _clip_bbox(ds: xr.Dataset, bbox_wgs84: tuple[float, float, float, float]) -> xr.Dataset:
    ds = _standardize_latlon(ds)
    w, s, e, n = bbox_wgs84
    lat = ds.latitude
    lat_slice = slice(n, s) if lat.values[0] > lat.values[-1] else slice(s, n)
    return ds.sel(latitude=lat_slice, longitude=slice(w, e))


def _coarsen_to_seas5(
    ds: xr.Dataset,
    target_resolution: float = SEAS5_RESOLUTION_DEG,
) -> xr.Dataset:
    """Block-average a regular fine grid to approximately SEAS5 resolution."""
    ds = _standardize_latlon(ds)
    if ds.sizes.get("latitude", 0) < 2 or ds.sizes.get("longitude", 0) < 2:
        return ds
    current_res = abs(float(ds.latitude.values[1] - ds.latitude.values[0]))
    factor = max(1, int(round(target_resolution / current_res)))
    if factor <= 1:
        return ds
    return ds.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()


def build_historical_monthly_fields(
    era5_daily: xr.Dataset,
    *,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    variables: list[str] = MATCHING_VARIABLES,
    target_grid: xr.Dataset | xr.DataArray | None = None,
) -> xr.Dataset:
    """Aggregate ERA5-Land daily data to monthly France fields for PCA.

    Precipitation is a monthly sum; other matching variables are monthly means.
    When a SEAS5 target grid is supplied the historical fields are interpolated
    onto that exact 1° grid so PCA feature positions are identical at fit and
    transform time.
    """
    ds = _standardize_latlon(era5_daily)
    if bbox_wgs84 is not None:
        ds = _clip_bbox(ds, bbox_wgs84)

    available = [v for v in variables if v in ds.data_vars]
    if not available:
        raise ValueError("No requested analog variables are present in ERA5-Land.")

    td = _time_dim(ds)
    monthly: dict[str, xr.DataArray] = {}
    for var in available:
        if var == "precip_mm":
            monthly[var] = ds[var].resample({td: "1MS"}).sum()
        else:
            monthly[var] = ds[var].resample({td: "1MS"}).mean()
    result = xr.Dataset(monthly)

    if target_grid is None:
        result = _coarsen_to_seas5(result)
    else:
        target = _standardize_latlon(
            target_grid.to_dataset(name="_x")
            if isinstance(target_grid, xr.DataArray)
            else target_grid
        )
        result = result.interp(
            latitude=target.latitude,
            longitude=target.longitude,
            method="linear",
        )

    if td != "time":
        result = result.rename({td: "time"})
    return result


def build_historical_pca(
    era5_daily: xr.Dataset,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    variables: list[str] = MATCHING_VARIABLES,
    *,
    target_grid: xr.Dataset | None = None,
    n_components: int = 4,
) -> RegionalPCA:
    """Fit the preferred France-level monthly PCA analog model."""
    monthly = build_historical_monthly_fields(
        era5_daily,
        bbox_wgs84=bbox_wgs84,
        variables=variables,
        target_grid=target_grid,
    )
    return fit_monthly_regional_pca(
        monthly,
        variables=variables,
        n_components=n_components,
        standardize=True,
    )


# ---------------------------------------------------------------------------
# PCA analog ranking
# ---------------------------------------------------------------------------


def _target_as_time_dataset(
    cell: xr.Dataset,
    *,
    variables: list[str],
    target_year: int,
    target_month: int,
    pca_model: RegionalPCA,
) -> xr.Dataset:
    target = _standardize_latlon(cell[variables])
    # Remove scalar coordinates/dimensions left by member/lead selection.
    for dim in list(target.dims):
        if dim not in {"latitude", "longitude"} and target.sizes[dim] == 1:
            target = target.isel({dim: 0}, drop=True)
    # Recover the PCA grid from feature metadata.  The feature vectors repeat
    # the same lat/lon grid for every variable.
    lats = np.unique(pca_model.feature_latitude.values.astype(float))[::-1]
    lons = np.unique(pca_model.feature_longitude.values.astype(float))
    target = target.interp(latitude=lats, longitude=lons, method="linear")
    target = target.expand_dims(time=[np.datetime64(f"{target_year:04d}-{target_month:02d}-01")])
    return target


def rank_analogs_pca(
    pca_model: RegionalPCA,
    new_ds: xr.Dataset,
    variables: list[str],
    *,
    target_year: int,
    target_month: int,
    top_k: int = 5,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Return top-k historical analogs for one corrected SEAS5 month."""
    target = _target_as_time_dataset(
        new_ds,
        variables=variables,
        target_year=target_year,
        target_month=target_month,
        pca_model=pca_model,
    )
    scores = transform_to_regional_pca(target, pca_model, variables)
    return rank_regional_analogs(
        pca_model,
        scores,
        top_k=top_k,
        temperature=temperature,
    )


# Backward-compatible name.  The return type is now a ranked DataFrame rather
# than the old (timestamp, diagnostics) winner tuple.
def find_analogs_pca(
    pca_model: RegionalPCA,
    new_ds: xr.Dataset,
    variables: list[str],
    method: str = "soft",
    *,
    target_year: int | None = None,
    target_month: int | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    del method  # legacy argument; regional PCA no longer spatial-votes cells.
    if target_month is None:
        if "valid_time" in new_ds.coords:
            target_month = pd.Timestamp(np.asarray(new_ds.valid_time).item()).month
        else:
            raise ValueError("target_month is required when new_ds has no valid_time coordinate.")
    if target_year is None:
        target_year = 2000
    return rank_analogs_pca(
        pca_model,
        new_ds,
        variables,
        target_year=target_year,
        target_month=target_month,
        top_k=top_k,
    )


def project_seas5_members_to_pca(
    seas5_corrected: xr.Dataset,
    pca_model: RegionalPCA,
    variables: list[str],
    *,
    init_year: int,
    init_month: int,
    lead: int,
) -> pd.DataFrame:
    """Project all SEAS5 members for one lead into the France PCA space.

    Returns one row per ensemble member with ``PC1``, ``PC2``, ... columns.
    This is intended for diagnostics/website figures comparing the SEAS5
    ensemble cloud with historical ERA5 months of the same calendar month.
    """
    ds = _standardize_latlon(seas5_corrected)
    member_dim = _find_dim(ds, ("number", "member", "realization"))
    lead_dim = _find_dim(
        ds, ("forecastMonth", "forecast_month", "leadtime_month", "lead", "lead_time")
    )
    if lead_dim is None:
        raise ValueError("SEAS5 corrected dataset has no lead-month dimension.")

    target_month = compute_valid_month(init_month, int(lead))
    target_year = compute_valid_year(init_year, init_month, int(lead))
    member_ids = [int(v) for v in ds[member_dim].values] if member_dim else [0]
    rows: list[dict[str, float | int]] = []

    for member_id in member_ids:
        target = _select_member_lead(
            ds,
            member_dim=member_dim,
            member_id=member_id,
            lead_dim=lead_dim,
            lead=int(lead),
        )
        target_time = _target_as_time_dataset(
            target,
            variables=variables,
            target_year=target_year,
            target_month=target_month,
            pca_model=pca_model,
        )
        score = transform_to_regional_pca(target_time, pca_model, variables).isel(time=0)
        row: dict[str, float | int] = {
            "member": int(member_id),
            "target_year": int(target_year),
            "target_month": int(target_month),
        }
        for component in score.component.values:
            row[str(component)] = float(score.sel(component=component))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analog extraction, redating and monthly rescaling
# ---------------------------------------------------------------------------


def _redate_month(
    month_data: xr.Dataset,
    *,
    target_year: int,
    target_month: int,
) -> xr.Dataset:
    """Move an analog daily sequence to the target month/year."""
    td = _time_dim(month_data)
    n_target = calendar.monthrange(target_year, target_month)[1]
    n_source = month_data.sizes[td]

    # February is the only practical month-length mismatch because analogs are
    # matched within the same calendar month.  Interpolate only when needed.
    if n_source != n_target:
        src = month_data.assign_coords(_day=(td, np.linspace(0.0, 1.0, n_source))).swap_dims(
            {td: "_day"}
        )
        target_pos = np.linspace(0.0, 1.0, n_target)
        month_data = src.interp(_day=target_pos).rename({"_day": td})

    dates = pd.date_range(f"{target_year:04d}-{target_month:02d}-01", periods=n_target, freq="D")
    month_data = month_data.assign_coords({td: dates})
    return month_data


def _target_to_native_grid(target: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    target_ds = _standardize_latlon(target.to_dataset(name="target"))["target"]
    template_ds = _standardize_latlon(template.to_dataset(name="template"))["template"]
    return target_ds.interp(
        latitude=template_ds.latitude,
        longitude=template_ds.longitude,
        method="linear",
    )


def _rescale_month_to_target(
    month_data: xr.Dataset,
    target: xr.Dataset,
    *,
    variables: list[str],
    precip_scale_bounds: tuple[float, float] | None = None,
) -> xr.Dataset:
    td = _time_dim(month_data)
    result = month_data.copy(deep=True)

    for var in variables:
        if var not in result.data_vars or var not in target.data_vars:
            continue
        da = result[var]
        target_native = _target_to_native_grid(target[var], da)

        if var == "precip_mm":
            hist_total = da.sum(td, skipna=True)
            safe = hist_total > 1e-6
            factor = xr.where(safe, target_native / hist_total, 1.0)
            if precip_scale_bounds is not None:
                lo, hi = precip_scale_bounds
                if bool(((factor.where(safe) < lo) | (factor.where(safe) > hi)).any()):
                    raise ValueError(
                        "Selected precipitation analog requires a scaling factor outside "
                        f"{precip_scale_bounds}; try another top-k analog."
                    )
            scaled = da * factor

            # Rare dry-cell fallback: borrow the France-wide event timing rather
            # than spreading rain uniformly through the month.
            if bool((~safe & (target_native > 1e-6)).any()):
                spatial_dims = [d for d in da.dims if d != td]
                event_pattern = da.mean(spatial_dims, skipna=True).clip(min=0)
                pattern_sum = event_pattern.sum(td)
                if float(pattern_sum) <= 1e-12:
                    event_pattern = xr.ones_like(event_pattern)
                    pattern_sum = event_pattern.sum(td)
                event_fraction = event_pattern / pattern_sum
                fallback = event_fraction * target_native
                scaled = xr.where(safe, scaled, fallback)
            result[var] = scaled.clip(min=0)

        elif var in {"t_min", "t_mean", "t_max", "dewpoint", "pressure_kpa"}:
            hist_mean = da.mean(td, skipna=True)
            result[var] = da + (target_native - hist_mean)

        elif var in {"rs_mj", "wind_speed_10m"}:
            hist_mean = da.mean(td, skipna=True)
            factor = xr.where(hist_mean > 1e-9, target_native / hist_mean, 1.0)
            result[var] = (da * factor).clip(min=0)

    # Preserve basic thermodynamic consistency after independent monthly deltas.
    if "t_min" in result and "t_mean" in result:
        result["t_min"] = xr.where(result.t_min <= result.t_mean, result.t_min, result.t_mean - 0.1)
    if "t_max" in result and "t_mean" in result:
        result["t_max"] = xr.where(result.t_max >= result.t_mean, result.t_max, result.t_mean + 0.1)
    if "dewpoint" in result and "t_mean" in result:
        result["dewpoint"] = xr.where(
            result.dewpoint <= result.t_mean, result.dewpoint, result.t_mean
        )

    return result


def extract_analog_daily_gridded(
    era5_daily: xr.Dataset,
    analog_year: int,
    analog_month: int,
    seas5_monthly_target: xr.Dataset | dict[str, float] | None = None,
    *,
    target_year: int | None = None,
    target_month: int | None = None,
    variables: list[str] = MATCHING_VARIABLES,
    precip_scale_bounds: tuple[float, float] | None = None,
) -> xr.Dataset:
    """Extract, redate and optionally rescale one ERA5-Land analog month."""
    td = _time_dim(era5_daily)
    selector = (era5_daily[td].dt.year == analog_year) & (era5_daily[td].dt.month == analog_month)
    month_data = era5_daily.sel({td: selector})
    if month_data.sizes[td] == 0:
        raise ValueError(f"No ERA5-Land data for {analog_year}-{analog_month:02d}")

    if target_year is not None or target_month is not None:
        target_year = analog_year if target_year is None else target_year
        target_month = analog_month if target_month is None else target_month
        month_data = _redate_month(
            month_data,
            target_year=int(target_year),
            target_month=int(target_month),
        )

    if seas5_monthly_target is not None:
        if isinstance(seas5_monthly_target, dict):
            # Compatibility with old scalar-spatial-mean calls.
            target_vars = {}
            template = month_data.isel({td: 0}, drop=True)
            for var, value in seas5_monthly_target.items():
                if var in template:
                    target_vars[var] = xr.full_like(template[var], float(value))
            target_ds = xr.Dataset(target_vars)
        else:
            target_ds = seas5_monthly_target
        month_data = _rescale_month_to_target(
            month_data,
            target_ds,
            variables=variables,
            precip_scale_bounds=precip_scale_bounds,
        )

    month_data.attrs.update(
        {
            "analog_source_year": int(analog_year),
            "analog_source_month": int(analog_month),
            "seasonal_disaggregation": "ERA5 analog rescaled to corrected SEAS5 monthly target",
        }
    )
    return month_data


# ---------------------------------------------------------------------------
# Full ensemble scenario generation
# ---------------------------------------------------------------------------


def _find_dim(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in ds.dims), None)


def _select_member_lead(
    ds: xr.Dataset,
    *,
    member_dim: str | None,
    member_id: int,
    lead_dim: str,
    lead: int,
) -> xr.Dataset:
    sel: dict[str, int] = {lead_dim: lead}
    if member_dim is not None:
        sel[member_dim] = member_id
    return ds.sel(sel, drop=True)


def generate_seasonal_scenarios(
    cfg: RegionConfig,
    era5_daily: xr.Dataset,
    seas5_corrected: xr.Dataset,
    init_month: int,
    n_leads: int = 6,
    variables: list[str] = MATCHING_VARIABLES,
    method: str = "sample",
    *,
    init_year: int | None = None,
    analog_top_k: int = 5,
    analog_temperature: float = 1.0,
    random_seed: int = 42,
    start_date: date | None = None,
    n_components: int = 4,
    precip_scale_bounds: tuple[float, float] | None = None,
    pca_bbox_wgs84: tuple[float, float, float, float] | None = None,
    pca_model: RegionalPCA | None = None,
) -> list[SeasonalScenario]:
    """Generate one daily ERA5-like scenario per SEAS5 ensemble member.

    Parameters
    ----------
    method
        ``"sample"`` samples from the top-k analogs using PCA-distance weights;
        ``"nearest"`` always takes rank 1.  The old ``"soft"``/``"hard"``
        values are accepted as aliases for ``"nearest"``.
    start_date
        Optional transition date from the medium-range forecast. Months ending
        before this date are skipped, but the first overlapping month is kept
        complete. ``extend_ifs_members_with_seasonal`` performs the final slice
        and can use the complete month to preserve the corrected SEAS5 monthly
        budget after subtracting observed/medium-range days.
    pca_bbox_wgs84
        Optional (west, south, east, north) PCA domain.  ``None`` (default)
        uses the complete ERA5/SEAS5 input domain, which is the intended
        France-level configuration.  Pass ``cfg.bbox_wgs84.as_tuple()`` only
        when deliberately building a regional PCA.
    pca_model
        Optional pre-fitted France-level PCA.  Supplying it avoids rebuilding
        monthly historical fields on every daily run; this is useful when the
        ERA5 analog archive is large.
    """
    if init_year is None:
        init_year = int(seas5_corrected.attrs.get("init_year", pd.Timestamp.today().year))

    seas5_corrected = _standardize_latlon(seas5_corrected)
    member_dim = _find_dim(seas5_corrected, ("number", "member", "realization"))
    lead_dim = _find_dim(
        seas5_corrected,
        ("forecastMonth", "forecast_month", "leadtime_month", "lead", "lead_time"),
    )
    if lead_dim is None:
        raise ValueError("SEAS5 corrected dataset has no lead-month dimension.")

    available = [
        v for v in variables if v in era5_daily.data_vars and v in seas5_corrected.data_vars
    ]
    if len(available) < 2:
        raise ValueError(
            f"Need at least two common analog variables; available={available}. "
            "Re-fetch SEAS5 with the expanded IrriGator variable list if using an old cache."
        )

    # ``cfg`` remains in the public signature for backward compatibility, but
    # PCA scope is intentionally independent from the parcel/region config.
    # With France-wide inputs the default therefore fits one France-wide PCA.
    _ = cfg
    if pca_model is None:
        pca_model = build_historical_pca(
            era5_daily,
            pca_bbox_wgs84,
            available,
            target_grid=seas5_corrected,
            n_components=n_components,
        )

    member_ids = (
        [int(v) for v in seas5_corrected[member_dim].values] if member_dim is not None else [0]
    )
    lead_values = [int(v) for v in seas5_corrected[lead_dim].values][:n_leads]
    td = _time_dim(era5_daily)
    scenarios: list[SeasonalScenario] = []

    nearest_only = method in {"nearest", "soft", "hard"}
    if method not in {"sample", "nearest", "soft", "hard"}:
        raise ValueError("method must be 'sample' or 'nearest'.")

    for member_id in member_ids:
        monthly_datasets: list[xr.Dataset] = []
        member_analogs: list[AnalogMatch] = []

        for lead in lead_values:
            target_month = compute_valid_month(init_month, lead)
            target_year = compute_valid_year(init_year, init_month, lead)
            target = _select_member_lead(
                seas5_corrected,
                member_dim=member_dim,
                member_id=member_id,
                lead_dim=lead_dim,
                lead=lead,
            )

            ranked = rank_analogs_pca(
                pca_model,
                target,
                available,
                target_year=target_year,
                target_month=target_month,
                top_k=analog_top_k,
                temperature=analog_temperature,
            )
            if ranked.empty:
                continue

            if nearest_only:
                chosen = ranked.iloc[0]
            else:
                # Stable independent RNG per member/lead: adding/removing another
                # member does not change existing selections.
                rng = np.random.default_rng(random_seed + 1009 * int(member_id) + 9176 * int(lead))
                idx = rng.choice(len(ranked), p=ranked["weight"].to_numpy(dtype=float))
                chosen = ranked.iloc[int(idx)]

            analog = AnalogMatch(
                target_month=target_month,
                target_year=target_year,
                analog_year=int(chosen.year),
                analog_month=int(chosen.month),
                distance=float(chosen.distance),
                weight=float(chosen.weight),
                rank=int(chosen["rank"]),
                member_id=int(member_id),
            )
            member_analogs.append(analog)

            try:
                month_ds = extract_analog_daily_gridded(
                    era5_daily,
                    analog_year=analog.analog_year,
                    analog_month=analog.analog_month,
                    seas5_monthly_target=target,
                    target_year=target_year,
                    target_month=target_month,
                    variables=available,
                    precip_scale_bounds=precip_scale_bounds,
                )
            except ValueError as exc:
                # If strict precipitation scaling bounds are requested, try the
                # next-ranked analog before giving up on this lead.
                month_ds = None
                if precip_scale_bounds is not None:
                    for _, fallback in ranked.iloc[1:].iterrows():
                        try:
                            month_ds = extract_analog_daily_gridded(
                                era5_daily,
                                analog_year=int(fallback.year),
                                analog_month=int(fallback.month),
                                seas5_monthly_target=target,
                                target_year=target_year,
                                target_month=target_month,
                                variables=available,
                                precip_scale_bounds=precip_scale_bounds,
                            )
                            analog.analog_year = int(fallback.year)
                            analog.analog_month = int(fallback.month)
                            analog.distance = float(fallback.distance)
                            analog.weight = float(fallback.weight)
                            analog.rank = int(fallback["rank"])
                            break
                        except ValueError:
                            continue
                if month_ds is None:
                    raise ValueError(
                        f"No acceptable analog for member {member_id}, lead {lead}: {exc}"
                    ) from exc

            if start_date is not None:
                month_start = date(target_year, target_month, 1)
                month_end = date(
                    target_year,
                    target_month,
                    calendar.monthrange(target_year, target_month)[1],
                )
                if month_end < start_date:
                    continue
                # Keep the full overlapping month.  The joining layer needs
                # days before ``start_date`` to compute the residual SEAS5
                # monthly budget after observations/AROME/IFS are accounted for.

            monthly_datasets.append(month_ds)
            logger.debug(
                "SEAS5 member %s lead %s -> %04d-%02d analog %04d-%02d rank=%d d=%.3f",
                member_id,
                lead,
                target_year,
                target_month,
                analog.analog_year,
                analog.analog_month,
                analog.rank,
                analog.distance,
            )

        if monthly_datasets:
            season_ds = xr.concat(monthly_datasets, dim=td).sortby(td)
            season_ds.attrs.update(
                {
                    "source": "SEAS5-conditioned ERA5-Land analog scenario",
                    "member_id": int(member_id),
                    "analog_method": "nearest" if nearest_only else "top-k weighted sample",
                    "analog_top_k": int(analog_top_k),
                }
            )
            scenarios.append(
                SeasonalScenario(
                    member_id=int(member_id),
                    daily_ds=season_ds,
                    analogs=member_analogs,
                )
            )

    logger.info(
        "Generated %d SEAS5-conditioned daily scenarios with %d shared variables",
        len(scenarios),
        len(available),
    )
    return scenarios
