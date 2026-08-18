"""SEAS5 monthly -> daily weather scenarios using France-level analogs.

The seasonal forecast carries useful information at monthly scale, while
AquaCrop needs daily forcing.  IrriGator therefore uses the SEAS5 forecast to
condition a set of historical ERA5-Land daily sequences rather than treating a
months-ahead daily SEAS5 trajectory as deterministic weather.

Pipeline
--------
1. Aggregate historical ERA5-Land monthly fields into the exact SEAS5 grid
   cells and retain cells intersecting metropolitan France.
2. Remove the calendar-month climatology, scale every feature by one residual
   standard deviation estimated from the complete archive, and fit **one**
   pooled France-wide PCA to all historical months.
3. Project each bias-corrected SEAS5 member/lead into the same PCA space.
4. Restrict candidate analogs to the same calendar month, rank them by PC-space
   distance (Euclidean by default), and optionally sample from the top-k.
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
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from irrigator.config import RegionConfig
from irrigator.forecasts.pca_framework import (
    RegionalPCA,
    fit_regional_anomaly_pca,
    rank_regional_analogs,
    transform_to_regional_pca,
)
from irrigator.forecasts.seas5_processor import (
    SEAS5_MATCHING_VARIABLES,
    compute_valid_month,
    compute_valid_year,
)

from irrigator.atmospheric.era5_processor import load_daily

logger = logging.getLogger(__name__)

# Deliberately excludes soil moisture: AquaCrop already carries the parcel's
# actual soil-water state forward.  These are atmospheric drivers shared by
# processed ERA5-Land and the SEAS5 monthly catalogue.
MATCHING_VARIABLES = list(SEAS5_MATCHING_VARIABLES)
SEAS5_RESOLUTION_DEG = 1.0

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")

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


def centers_to_edges(centers: np.ndarray | list[float]) -> np.ndarray:
    """Convert monotonically increasing grid-cell centers to bin edges."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must contain at least two 1-D coordinates.")
    if not np.all(np.diff(centers) > 0):
        raise ValueError("centers must be strictly increasing.")
    mid = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def aggregate_to_target_grid(
    ds: xr.Dataset,
    target_latitude: np.ndarray | list[float],
    target_longitude: np.ndarray | list[float],
) -> xr.Dataset:
    """Average fine-grid cells into the exact target-grid cell footprints.

    This is the production version of the notebook helper.  Unlike linear
    interpolation, each SEAS5 feature is the mean of the ERA5-Land cells whose
    centres fall inside that SEAS5 cell.  The returned coordinates are reordered
    to the original target-grid orientation.
    """
    ds = ds.sortby("latitude").sortby("longitude")
    target_lat_original = np.asarray(target_latitude, dtype=float)
    target_lon_original = np.asarray(target_longitude, dtype=float)
    target_lat = np.sort(target_lat_original)
    target_lon = np.sort(target_lon_original)

    lat_edges = centers_to_edges(target_lat)
    lon_edges = centers_to_edges(target_lon)

    out = (
        ds.groupby_bins(
            "latitude",
            lat_edges,
            labels=target_lat,
            include_lowest=True,
        )
        .mean("latitude")
        .rename({"latitude_bins": "latitude"})
    )
    out = (
        out.groupby_bins(
            "longitude",
            lon_edges,
            labels=target_lon,
            include_lowest=True,
        )
        .mean("longitude")
        .rename({"longitude_bins": "longitude"})
    )
    out = out.sel(latitude=target_lat_original, longitude=target_lon_original)
    non_spatial = [d for d in ds.dims if d not in {"latitude", "longitude"}]
    return out.transpose(*non_spatial, "latitude", "longitude", missing_dims="ignore")


def build_intersection_mask(ds: xr.Dataset, geometry) -> xr.DataArray:
    """Return cells whose rectangular footprint intersects ``geometry``."""
    from shapely.geometry import box

    lat = np.asarray(ds.latitude, dtype=float)
    lon = np.asarray(ds.longitude, dtype=float)
    if lat.size < 2 or lon.size < 2:
        raise ValueError("At least two latitude/longitude cells are required for a cell mask.")

    dlat = float(np.median(np.abs(np.diff(lat))))
    dlon = float(np.median(np.abs(np.diff(lon))))
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, y in enumerate(lat):
        for j, x in enumerate(lon):
            cell = box(x - dlon / 2, y - dlat / 2, x + dlon / 2, y + dlat / 2)
            mask[i, j] = cell.intersection(geometry).area > 0

    return xr.DataArray(
        mask,
        coords={"latitude": ds.latitude, "longitude": ds.longitude},
        dims=("latitude", "longitude"),
        name="france_mask",
    )


def get_metropolitan_france_geometry():
    """Load metropolitan France (mainland + Corsica) from Natural Earth.

    Cartopy is imported lazily because only PCA fitting needs this helper.  The
    user's PCA notebook already uses the same Natural Earth source, so the
    production mask matches the exploratory analysis.
    """
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "cartopy and shapely are required to build the metropolitan-France PCA mask. "
            "Pass a precomputed spatial_mask to build_historical_pca(), or install cartopy."
        ) from exc

    shp = shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )
    france = None
    for record in shpreader.Reader(shp).records():
        attrs = record.attributes
        if (
            attrs.get("ADM0_A3") == "FRA"
            or attrs.get("NAME") == "France"
            or attrs.get("NAME_LONG") == "France"
        ):
            france = record.geometry
            break
    if france is None:
        raise RuntimeError("France not found in Natural Earth.")
    return france.intersection(box(-6.0, 41.0, 10.0, 52.0))


def build_metropolitan_france_mask(ds: xr.Dataset) -> xr.DataArray:
    """Keep every target grid cell intersecting metropolitan France."""
    return build_intersection_mask(ds, get_metropolitan_france_geometry())


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


def get_historical_monthly_fields(
    overwrite: bool = False,
    # year_min : int = 1970,
    year_min: int = 2013,
    year_max: int = 2026,
):
    """Open the era5 monthly averages."""
    out_dir = (DEFAULT_RAW_DIR / "era5_candidates")

    out_dir.mkdir(parents=True, exist_ok=True)
    file = list(out_dir.glob("*.nc"))[0]
    if file.exists() and not overwrite:
        ds = xr.open_dataset(file)
        return ds.sel(time=slice(f"{year_min}-01-01", f"{year_max+1}-01-01"))
    else:
        from irrigator.ingestion.cds_client import fetch_era5_month
        fetch_era5_month(year_min=year_min, year_max=year_max, raw_dir=out_dir)
        return out_dir


def month_bounds(year, month):
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])
    return first_day, last_day

# def build_historical_pca(
#     overwrite_era5 : bool = False,
#     variables: list[str] = MATCHING_VARIABLES,
#     *,
#     target_grid: xr.Dataset | None = None,
#     n_components: int = 10,
#     spatial_mask: xr.DataArray | None = None,
#     mask_metropolitan_france: bool = True,
#     latitude_weighting: bool = False,
# ) -> RegionalPCA:
#     """Fit the pooled France-level monthly-anomaly PCA analog model.

#     If ``spatial_mask`` is omitted, metropolitan France is built from the same
#     Natural Earth geometry used in the exploratory PCA notebook.  Cells are
#     retained when their 1° footprint intersects the French polygon, including
#     border/crossing cells.
#     """
#     monthly = get_historical_monthly_fields(overwrite = overwrite_era5)
#     if spatial_mask is None and mask_metropolitan_france:
#         spatial_mask = build_metropolitan_france_mask(monthly)
#     return fit_regional_anomaly_pca(
#         monthly,
#         variables=variables,
#         n_components=n_components,
#         spatial_mask=spatial_mask,
#         latitude_weighting=latitude_weighting,
#     )


def build_historical_pca(
    overwrite_era5: bool = False,
    variables: list[str] = MATCHING_VARIABLES,
    *,
    target_grid: xr.Dataset | None = None,
    n_components: int = 10,
    spatial_mask: xr.DataArray | None = None,
    mask_metropolitan_france: bool = True,
    latitude_weighting: bool = False,
) -> RegionalPCA:

    monthly = get_historical_monthly_fields(overwrite=overwrite_era5)

    monthly = _standardize_latlon(monthly)

    if target_grid is not None:
        target_grid = _standardize_latlon(target_grid)

        monthly = aggregate_to_target_grid(
            monthly,
            target_latitude=target_grid.latitude.values,
            target_longitude=target_grid.longitude.values,
        )

    if spatial_mask is None and mask_metropolitan_france:
        spatial_mask = build_metropolitan_france_mask(monthly)

    return fit_regional_anomaly_pca(
        monthly,
        variables=variables,
        n_components=n_components,
        spatial_mask=spatial_mask,
        latitude_weighting=latitude_weighting,
    )

# ---------------------------------------------------------------------------
# PCA analog ranking
# ---------------------------------------------------------------------------


# def _target_as_time_dataset(
#     cell: xr.Dataset,
#     *,
#     variables: list[str],
#     target_year: int,
#     target_month: int,
#     pca_model: RegionalPCA,
# ) -> xr.Dataset:
#     target = _standardize_latlon(cell[variables])
#     # Remove scalar coordinates/dimensions left by member/lead selection.
#     for dim in list(target.dims):
#         if dim not in {"latitude", "longitude"} and target.sizes[dim] == 1:
#             target = target.isel({dim: 0}, drop=True)
#     # Recover the PCA grid from feature metadata.  The feature vectors repeat
#     # the same lat/lon grid for every variable.
#     lats = np.unique(pca_model.feature_latitude.values.astype(float))[::-1]
#     lons = np.unique(pca_model.feature_longitude.values.astype(float))
#     target = target.interp(latitude=lats, longitude=lons, method="linear")
#     target = target.expand_dims(time=[np.datetime64(f"{target_year:04d}-{target_month:02d}-01")])
#     return target


def _target_as_time_dataset(
    cell: xr.Dataset,
    *,
    variables: list[str],
    target_year: int,
    target_month: int,
    pca_model: RegionalPCA,
) -> xr.Dataset:

    target = _standardize_latlon(cell[variables])

    for dim in list(target.dims):
        if dim not in {"latitude", "longitude"} and target.sizes[dim] == 1:
            target = target.isel({dim: 0}, drop=True)

    if pca_model.spatial_mask is not None:
        lats = pca_model.spatial_mask.latitude.values
        lons = pca_model.spatial_mask.longitude.values
    else:
        lats = np.unique(pca_model.feature_latitude.values.astype(float))[::-1]
        lons = np.unique(pca_model.feature_longitude.values.astype(float))

    target = target.interp(
        latitude=lats,
        longitude=lons,
        method="linear",
    )

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
    metric: Literal["euclidean", "mahalanobis"] = "euclidean",
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
        metric=metric,
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
    metric: Literal["euclidean", "mahalanobis"] = "euclidean",
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
        metric=metric,
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

    print(n_target)
    print(month_data)

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

    era5_daily = load_daily(year=analog_year)
    print(era5_daily)
    print("Faily loaded")

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
    seas5_corrected: xr.Dataset,
    init_month: int,
    n_leads: int = 6,
    variables: list[str] = MATCHING_VARIABLES,
    method: str = "sample",
    *,
    init_year: int | None = None,
    analog_top_k: int = 5,
    analog_temperature: float = 1.0,
    analog_metric: Literal["euclidean", "mahalanobis"] = "euclidean",
    random_seed: int = 42,
    start_date: date | None = None,
    n_components: int = 10,
    precip_scale_bounds: tuple[float, float] | None = None,
    pca_bbox_wgs84: tuple[float, float, float, float] | None = None,
    pca_spatial_mask: xr.DataArray | None = None,
    mask_metropolitan_france: bool = True,
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
    pca_spatial_mask
        Optional precomputed mask on the SEAS5/PCA grid.  When omitted and
        ``mask_metropolitan_france=True``, cells intersecting metropolitan France
        are retained using Natural Earth.
    analog_metric
        PC-space distance used for same-month analog ranking.  ``"euclidean"``
        matches the exploratory notebook and is the default; ``"mahalanobis"``
        remains available for comparison.
    pca_model
        Optional pre-fitted France-level pooled anomaly PCA.  Supplying it avoids
        rebuilding monthly historical fields on every daily run; this is useful
        when the ERA5 analog archive is large.
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
        v for v in variables if v in seas5_corrected.data_vars
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
    # if pca_model is None:
    #     pca_model = build_historical_pca(
    #         overwrite_era5=False,
    #         variables=available,
    #         target_grid=seas5_corrected,
    #         n_components=n_components,
    #         spatial_mask=pca_spatial_mask,
    #         mask_metropolitan_france=mask_metropolitan_france,
    #     )

    member_ids = (
        [int(v) for v in seas5_corrected[member_dim].values] if member_dim is not None else [0]
    )
    lead_values = [int(v) for v in seas5_corrected[lead_dim].values][:n_leads]
    td = "valid_time"
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
                metric=analog_metric,
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
                print(analog.analog_year)
                print(analog.analog_month)
                month_ds = extract_analog_daily_gridded(
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
                        print(fallback.year)
                        print(fallback.month)
                        try:
                            
                            month_ds = extract_analog_daily_gridded(
                                analog_year=int(fallback.year),
                                analog_month=int(fallback.month),
                                seas5_monthly_target=target,
                                target_year=target_year,
                                target_month=target_month,
                                variables=available,
                                precip_scale_bounds=precip_scale_bounds,
                            )
                            print(month_ds)
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
