"""Helpers for joining IrriGator forecast horizons.

AROME (0-48 h) -> IFS ENS (~48 h-day 15) -> SEAS5-conditioned analogs.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np

from irrigator.atmospheric.forcing import DailyForcing, extract_parcel_forcing
from irrigator.config import ParcelConfig
from irrigator.forecasts.disaggregation import SeasonalScenario
from irrigator.static_layers.terrain import TerrainParams


def _last_date(forcing: DailyForcing) -> date:
    if forcing.n_days == 0:
        raise ValueError("Cannot extend an empty IFS member forcing.")
    return np.datetime64(forcing.dates[-1], "D").astype(object)


def _with_arrays(forcing: DailyForcing, **updates: np.ndarray) -> DailyForcing:
    values = {
        "dates": forcing.dates,
        "t_min": forcing.t_min,
        "t_max": forcing.t_max,
        "t_mean": forcing.t_mean,
        "dewpoint": forcing.dewpoint,
        "wind_speed_2m": forcing.wind_speed_2m,
        "pressure_kpa": forcing.pressure_kpa,
        "rs_mj": forcing.rs_mj,
        "precip_mm": forcing.precip_mm,
    }
    values.update(updates)
    return DailyForcing(**values)


def _preserve_transition_month_budget(
    seasonal_full: DailyForcing,
    known: DailyForcing,
    *,
    start: date,
) -> DailyForcing:
    """Adjust the seasonal tail so the complete transition month keeps its target.

    ``seasonal_full`` is the full analog month already rescaled to corrected
    SEAS5.  Observed/AROME/IFS days before ``start`` are treated as known.
    Precipitation uses the remaining monthly water budget; mean variables use
    the remaining monthly mean budget.
    """
    month_start = date(start.year, start.month, 1)
    month_end = date(start.year, start.month, calendar.monthrange(start.year, start.month)[1])
    full = seasonal_full.slice(start=month_start, end=month_end)
    tail = seasonal_full.slice(start=start)
    if full.n_days == 0 or tail.n_days == 0 or start == month_start:
        return tail

    known_month = known.slice(start=month_start, end=start - timedelta(days=1))
    n_month = calendar.monthrange(start.year, start.month)[1]
    n_remaining = (month_end - start).days + 1
    if known_month.n_days == 0:
        return tail

    mask = tail.dates.astype("datetime64[M]") == np.datetime64(month_start, "M")
    n_tail_month = int(mask.sum())
    if n_tail_month == 0:
        return tail

    updates: dict[str, np.ndarray] = {}

    # Exact residual precipitation budget. If known rainfall already exceeds
    # the corrected monthly SEAS5 total, the residual target is zero.
    target_total = float(np.nansum(full.precip_mm))
    known_total = float(np.nansum(known_month.precip_mm))
    remaining_total = max(0.0, target_total - known_total)
    precip = tail.precip_mm.copy()
    source_total = float(np.nansum(precip[mask]))
    if source_total > 1e-9:
        precip[mask] *= remaining_total / source_total
    elif remaining_total > 0:
        precip[mask] = remaining_total / n_tail_month
    else:
        precip[mask] = 0.0
    updates["precip_mm"] = np.clip(precip, 0.0, None)

    # Preserve the full-month mean for continuous variables.  This is a
    # first-order constraint; daily sequencing still comes from the analog.
    additive = ("t_min", "t_mean", "t_max", "dewpoint", "pressure_kpa")
    positive = ("wind_speed_2m", "rs_mj")
    for name in additive:
        arr = getattr(tail, name).copy()
        target_mean = float(np.nanmean(getattr(full, name)))
        known_sum = float(np.nansum(getattr(known_month, name)))
        remaining_mean = (target_mean * n_month - known_sum) / max(1, n_remaining)
        source_mean = float(np.nanmean(arr[mask]))
        arr[mask] += remaining_mean - source_mean
        updates[name] = arr
    for name in positive:
        arr = getattr(tail, name).copy()
        target_mean = float(np.nanmean(getattr(full, name)))
        known_sum = float(np.nansum(getattr(known_month, name)))
        remaining_mean = max(0.0, (target_mean * n_month - known_sum) / max(1, n_remaining))
        source_mean = float(np.nanmean(arr[mask]))
        if source_mean > 1e-9:
            arr[mask] *= remaining_mean / source_mean
        else:
            arr[mask] = remaining_mean
        updates[name] = np.clip(arr, 0.0, None)

    # Restore basic temperature/dewpoint consistency after independent shifts.
    tmin = updates["t_min"]
    tmean = updates["t_mean"]
    tmax = updates["t_max"]
    dew = updates["dewpoint"]
    updates["t_min"] = np.minimum(tmin, tmean - 0.05)
    updates["t_max"] = np.maximum(tmax, tmean + 0.05)
    updates["dewpoint"] = np.minimum(dew, tmean)
    return _with_arrays(tail, **updates)


def extend_ifs_members_with_seasonal(
    ifs_member_forcings: dict[int, DailyForcing],
    seasonal_scenarios: list[SeasonalScenario],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    *,
    seasonal_start: date | None = None,
    end_date: date | None = None,
    known_prefix_forcing: DailyForcing | None = None,
    preserve_transition_month: bool = True,
) -> dict[int, DailyForcing]:
    """Append SEAS5-conditioned daily scenarios to IFS ensemble members.

    Pass ``known_prefix_forcing=historical.concat(arome)`` when the IFS/SEAS5
    boundary falls mid-month.  The first seasonal tail is then adjusted using
    the *remaining* corrected SEAS5 monthly budget after all known and IFS days
    are subtracted. This is especially important for precipitation.
    """
    if not ifs_member_forcings:
        raise ValueError("ifs_member_forcings is empty.")
    if not seasonal_scenarios:
        raise ValueError("seasonal_scenarios is empty.")

    exact = {int(s.member_id): s for s in seasonal_scenarios}
    ordered = sorted(seasonal_scenarios, key=lambda s: int(s.member_id))
    member_ids = sorted(int(m) for m in ifs_member_forcings)

    result: dict[int, DailyForcing] = {}
    for position, member_id in enumerate(member_ids):
        ifs = ifs_member_forcings[member_id]
        scenario = exact.get(member_id, ordered[position % len(ordered)])
        seasonal_full = extract_parcel_forcing(
            scenario.daily_ds, parcel=parcel, terrain=terrain, precip_correction=None
        )
        start = seasonal_start or (_last_date(ifs) + timedelta(days=1))

        if preserve_transition_month and start.day != 1:
            known = ifs if known_prefix_forcing is None else known_prefix_forcing.concat(ifs)
            seasonal = _preserve_transition_month_budget(seasonal_full, known, start=start)
        else:
            seasonal = seasonal_full.slice(start=start)

        if end_date is not None:
            seasonal = seasonal.slice(end=end_date)
        result[member_id] = ifs.concat(seasonal)

    return result
