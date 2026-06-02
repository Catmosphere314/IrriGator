"""Analog-based disaggregation: monthly SEAS5 → daily xr.Dataset.

Finds historical months whose climate pattern best matches each SEAS5
ensemble member prediction, then returns that month's full ERA5-Land
daily data at native resolution.  The caller then passes the result
through ``atmospheric.forcing.extract_parcel_forcing()`` for
parcel-level downscaling — no duplication.

Resolution workflow
-------------------
1. MATCHING happens at SEAS5 resolution (~1°):
   ERA5-Land monthly stats are coarsened to 1° before computing
   similarity, because SEAS5 can't resolve finer detail.

2. EXTRACTION happens at ERA5-Land resolution (~9 km / 0.1°):
   once the best analog month is selected, its full daily data
   at native resolution is returned as an xr.Dataset.

3. DOWNSCALING to parcel level is NOT done here — the returned
   Dataset is passed through ``extract_parcel_forcing()`` from
   Block 2 (atmospheric/forcing.py), which handles lapse rate,
   radiation correction, wind conversion, and bias correction.

This keeps one downscaling path for all data sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from irrigator.config import RegionConfig

logger = logging.getLogger(__name__)

# Variables used for analog matching
MATCHING_VARIABLES = ["t_mean", "precip_mm", "rs_mj"]

SEAS5_RESOLUTION_DEG = 1.0  # ~1° grid for matching


@dataclass
class AnalogMatch:
    """Result of analog matching for one month."""

    target_month: int
    analog_year: int
    analog_month: int
    similarity: float
    member_id: int


@dataclass
class SeasonalScenario:
    """One ensemble member's full daily scenario as a gridded Dataset.

    The ``daily_ds`` field is an xr.Dataset with the same structure as
    ERA5-Land daily output, ready to be passed through
    ``extract_parcel_forcing()`` for parcel-level downscaling.
    """

    member_id: int
    daily_ds: xr.Dataset
    analogs: list[AnalogMatch]


# ---------------------------------------------------------------------------
# Coarsening ERA5-Land to SEAS5 resolution for matching
# ---------------------------------------------------------------------------


def _coarsen_to_seas5(
    ds: xr.Dataset,
    target_resolution: float = SEAS5_RESOLUTION_DEG,
) -> xr.Dataset:
    """Coarsen an ERA5-Land dataset from ~0.1° to ~1° for analog matching.

    Uses block averaging.  The ratio 1.0 / 0.1 = 10, so we coarsen by
    a factor of 10 in each spatial dimension.
    """
    # Detect spatial dimension names
    lat_dim = "latitude" if "latitude" in ds.dims else "y"
    lon_dim = "longitude" if "longitude" in ds.dims else "x"

    # Compute coarsening factor
    if lat_dim in ds.coords:
        lat_vals = ds[lat_dim].values
        current_res = abs(float(lat_vals[1] - lat_vals[0])) if len(lat_vals) > 1 else 0.1
    else:
        current_res = 0.1

    factor = max(1, int(round(target_resolution / current_res)))

    if factor <= 1:
        return ds  # already at or coarser than target

    coarsened = ds.coarsen(
        {lat_dim: factor, lon_dim: factor},
        boundary="trim",
    ).mean()

    logger.debug(
        "Coarsened from %.2f° to %.2f° (factor %d)",
        current_res,
        current_res * factor,
        factor,
    )
    return coarsened


# ---------------------------------------------------------------------------
# Historical monthly stats at SEAS5 resolution
# ---------------------------------------------------------------------------


def build_historical_monthly_stats(
    era5_daily: xr.Dataset,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    variables: list[str] = MATCHING_VARIABLES,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """Build monthly statistics at SEAS5 resolution for analog matching.

    Also returns the coarsened monthly dataset for spatial matching.

    Parameters
    ----------
    era5_daily : ERA5-Land daily at native resolution
    bbox_wgs84 : optional (west, south, east, north) to clip
    variables : which variables to use

    Returns
    -------
    (stats_df, coarse_monthly)
        stats_df: DataFrame indexed by (year, month) with spatial-mean values
        coarse_monthly: xr.Dataset at ~1° monthly resolution (for spatial matching)
    """
    ds = era5_daily
    if bbox_wgs84:
        w, s, e, n = bbox_wgs84
        lat_name = "latitude" if "latitude" in ds.dims else "y"
        lon_name = "longitude" if "longitude" in ds.dims else "x"
        ds = ds.sel({lat_name: slice(n, s), lon_name: slice(w, e)})

    # Coarsen to SEAS5 resolution
    coarse = _coarsen_to_seas5(ds)

    # Monthly aggregation
    monthly_parts = {}
    for var in variables:
        if var not in coarse.data_vars:
            continue
        if "precip" in var:
            monthly_parts[var] = coarse[var].resample(valid_time="1ME").sum()
        else:
            monthly_parts[var] = coarse[var].resample(valid_time="1ME").mean()

    coarse_monthly = xr.Dataset(monthly_parts)

    # Spatial mean for the d-sphere matching
    lat_dim = "latitude" if "latitude" in coarse_monthly.dims else "y"
    lon_dim = "longitude" if "longitude" in coarse_monthly.dims else "x"
    spatial_mean = coarse_monthly.mean(dim=[lat_dim, lon_dim])

    records = []
    for t in spatial_mean.valid_time.values:
        ts = pd.Timestamp(t)
        row = {"year": ts.year, "month": ts.month}
        for var in variables:
            if var in spatial_mean.data_vars:
                row[var] = float(spatial_mean[var].sel(valid_time=t))
        records.append(row)

    stats_df = pd.DataFrame(records).set_index(["year", "month"])

    logger.info(
        "Historical stats at 1°: %d months, %d variables",
        len(stats_df),
        len([v for v in variables if v in stats_df.columns]),
    )
    return stats_df, coarse_monthly


# ---------------------------------------------------------------------------
# D-sphere matching
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_analogs(
    historical_stats: pd.DataFrame,
    target_values: dict[str, float],
    target_month: int,
    variables: list[str] = MATCHING_VARIABLES,
    k: int = 5,
) -> list[tuple[int, int, float]]:
    """Find k most similar historical months on the d-sphere.

    Matching is at SEAS5 resolution (spatial means at 1°).

    Parameters
    ----------
    historical_stats : from build_historical_monthly_stats
    target_values : SEAS5 corrected monthly values (spatial mean)
    target_month : calendar month to match
    variables : matching variables
    k : number of analogs

    Returns
    -------
    List of (year, month, cosine_similarity), best first.
    """
    candidates = historical_stats.loc[
        historical_stats.index.get_level_values("month") == target_month
    ]

    available_vars = [v for v in variables if v in candidates.columns and v in target_values]
    if not available_vars:
        return []

    # Standardize using historical distribution
    hist_matrix = candidates[available_vars].values
    target_vec = np.array([target_values[v] for v in available_vars])

    mu = hist_matrix.mean(axis=0)
    sigma = hist_matrix.std(axis=0)
    sigma[sigma < 1e-12] = 1.0

    hist_std = (hist_matrix - mu) / sigma
    target_std = (target_vec - mu) / sigma

    # Normalize to unit sphere
    target_unit = target_std / (np.linalg.norm(target_std) + 1e-12)

    similarities = []
    for i, (idx, _) in enumerate(candidates.iterrows()):
        hist_unit = hist_std[i] / (np.linalg.norm(hist_std[i]) + 1e-12)
        sim = _cosine_similarity(target_unit, hist_unit)
        year, month = idx
        similarities.append((year, month, sim))

    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:k]


# ---------------------------------------------------------------------------
# Extract analog daily at ERA5-Land native resolution
# ---------------------------------------------------------------------------


def extract_analog_daily_gridded(
    era5_daily: xr.Dataset,
    analog_year: int,
    analog_month: int,
    seas5_monthly_target: dict[str, float] | None = None,
) -> xr.Dataset:
    """Extract the analog month's daily data at ERA5-Land native resolution.

    Optionally rescales so the monthly mean/total matches SEAS5 prediction.
    Returns an xr.Dataset with the same structure as ERA5-Land daily —
    ready to be passed through ``extract_parcel_forcing()``.

    Parameters
    ----------
    era5_daily : full ERA5-Land daily dataset at native resolution (~9 km)
    analog_year, analog_month : selected analog
    seas5_monthly_target : optional dict of var -> target value for rescaling

    Returns
    -------
    xr.Dataset for the analog month, at ERA5-Land native resolution.
    """
    time_dim = "valid_time" if "valid_time" in era5_daily.dims else "time"

    month_data = era5_daily.sel(
        {
            time_dim: (era5_daily[time_dim].dt.year == analog_year)
            & (era5_daily[time_dim].dt.month == analog_month)
        }
    )

    if len(month_data[time_dim]) == 0:
        raise ValueError(f"No data for {analog_year}-{analog_month:02d}")

    # Rescale to match SEAS5 target if provided
    if seas5_monthly_target:
        result_vars = {}
        for var in month_data.data_vars:
            da = month_data[var].copy()
            if var in seas5_monthly_target:
                target = seas5_monthly_target[var]
                if "precip" in var:
                    hist_total = float(da.mean(dim=[d for d in da.dims if d != time_dim]).sum())
                    if hist_total > 0.1:
                        da = da * (target / hist_total)
                else:
                    hist_mean = float(da.mean())
                    da = da + (target - hist_mean)
            result_vars[var] = da
        month_data = xr.Dataset(result_vars)

    return month_data


# ---------------------------------------------------------------------------
# Full scenario generation
# ---------------------------------------------------------------------------


def generate_seasonal_scenarios(
    cfg: RegionConfig,
    era5_daily: xr.Dataset,
    seas5_corrected: xr.Dataset,
    init_month: int,
    n_leads: int = 6,
    variables: list[str] = MATCHING_VARIABLES,
) -> list[SeasonalScenario]:
    """Generate gridded daily scenarios from SEAS5 ensemble.

    For each member + lead: match at 1°, extract at 0.1°, concatenate.
    The output is xr.Datasets at ERA5-Land resolution — the caller
    passes them through ``extract_parcel_forcing()`` for parcel downscaling.

    Parameters
    ----------
    cfg : RegionConfig
    era5_daily : ERA5-Land daily (historical, native resolution)
    seas5_corrected : bias-corrected SEAS5 monthly
    init_month : SEAS5 initialization month
    n_leads : number of lead months
    variables : variables for analog matching

    Returns
    -------
    List of SeasonalScenario, one per ensemble member.
    """
    # Step 1: build historical stats at SEAS5 resolution (1°)
    bbox = cfg.bbox_wgs84.as_tuple()
    hist_stats, _ = build_historical_monthly_stats(era5_daily, bbox, variables)

    # Identify dimensions
    member_dim = None
    for cand in ("number", "member", "realization"):
        if cand in seas5_corrected.dims:
            member_dim = cand
            break

    lead_dim = None
    for cand in ("forecastMonth", "leadtime_month", "lead"):
        if cand in seas5_corrected.dims:
            lead_dim = cand
            break

    n_members = seas5_corrected.sizes.get(member_dim, 1) if member_dim else 1
    member_ids = seas5_corrected[member_dim].values if member_dim else [0]

    lat_dim = "latitude" if "latitude" in seas5_corrected.dims else "lat"
    lon_dim = "longitude" if "longitude" in seas5_corrected.dims else "lon"
    time_dim = "valid_time" if "valid_time" in era5_daily.dims else "time"

    scenarios = []

    for member_id in member_ids:
        monthly_datasets = []
        member_analogs = []

        for lead in range(1, n_leads + 1):
            valid_month = ((init_month - 1) + (lead - 1)) % 12 + 1

            # Extract SEAS5 spatial mean for this member + lead
            sel = {}
            if member_dim:
                sel[member_dim] = member_id
            if lead_dim:
                sel[lead_dim] = lead

            cell = seas5_corrected.sel(**sel).mean(dim=[lat_dim, lon_dim])
            target_values = {}
            for var in variables:
                if var in cell.data_vars:
                    target_values[var] = float(cell[var])

            if not target_values:
                continue

            # Match at 1° resolution
            analogs = find_analogs(hist_stats, target_values, valid_month, variables, k=3)
            if not analogs:
                continue

            best_year, best_month, similarity = analogs[0]

            member_analogs.append(
                AnalogMatch(
                    target_month=valid_month,
                    analog_year=best_year,
                    analog_month=best_month,
                    similarity=similarity,
                    member_id=int(member_id),
                )
            )

            # Extract at ERA5-Land native resolution
            month_ds = extract_analog_daily_gridded(
                era5_daily,
                best_year,
                best_month,
                target_values,
            )
            monthly_datasets.append(month_ds)

            logger.debug(
                "Member %d, lead %d (month %d): analog %d-%02d (sim=%.3f)",
                member_id,
                lead,
                valid_month,
                best_year,
                best_month,
                similarity,
            )

        # Concatenate months into full season gridded dataset
        if monthly_datasets:
            season_ds = xr.concat(monthly_datasets, dim=time_dim)
            scenarios.append(
                SeasonalScenario(
                    member_id=int(member_id),
                    daily_ds=season_ds,
                    analogs=member_analogs,
                )
            )

    logger.info("Generated %d gridded scenarios (%d leads)", len(scenarios), n_leads)
    return scenarios
