"""Late-season grain-moisture stage estimation for grain maize.

This module deliberately sits *beside* AquaCrop rather than inside it.
AquaCrop models crop-water physiology, but it does not expose the French
H50/H45 grain-moisture stages commonly used to decide when maize irrigation
can stop.  ARVALIS/IRRINOV expresses those stages as thermal-time requirements
(base 6-30 degC) from female flowering.

The table lives in ``configs/grain_moisture_stages.yaml`` so the agronomic
values remain transparent and can be replaced/refined without touching model
code.

The intended operational use is intentionally limited:

* before H50: keep the normal rolling optimization horizon;
* H50 <= stage < H45: shorten the optimization horizon to the forecast H45 date;
* at/after H45: do not propose new irrigation events.

AquaCrop still decides whether water is needed inside the active horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from irrigator.atmospheric.forcing import DailyForcing

DEFAULT_GRAIN_MOISTURE_TABLE = Path("configs/grain_moisture_stages.yaml")


class GrainMoistureStage(str, Enum):
    """Coarse late-season irrigation stage."""

    PRE_H50 = "pre_h50"
    H50_TO_H45 = "h50_to_h45"
    POST_H45 = "post_h45"


@dataclass(frozen=True)
class GrainMoistureThresholds:
    """Thermal thresholds for one maize maturity group.

    All GDD values use the thermal convention stored in the source table,
    currently ARVALIS base 6-30 degC.
    """

    maturity_group: str
    label: str
    flowering_gdd: float
    h50_after_flowering_gdd: float
    h45_after_flowering_gdd: float
    t_base_c: float
    t_upper_c: float
    h50_after_flowering_range: tuple[float, float] | None = None
    h45_after_flowering_range: tuple[float, float] | None = None

    @property
    def h50_gdd(self) -> float:
        return self.flowering_gdd + self.h50_after_flowering_gdd

    @property
    def h45_gdd(self) -> float:
        return self.flowering_gdd + self.h45_after_flowering_gdd


@dataclass(frozen=True)
class ForecastThresholdDate:
    """Ensemble estimate of the date a GDD threshold is crossed."""

    date: date | None
    member_dates: dict[int, date | None]
    coverage: float
    quantile: float


def _midpoint(value: list[float] | tuple[float, float]) -> float:
    if len(value) != 2:
        raise ValueError(f"Expected a two-value range, got {value!r}")
    lo, hi = map(float, value)
    if hi < lo:
        raise ValueError(f"Invalid range {value!r}: upper bound < lower bound")
    return 0.5 * (lo + hi)


def _as_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected a two-value range, got {value!r}")
    lo, hi = map(float, value)
    if hi < lo:
        raise ValueError(f"Invalid range {value!r}: upper bound < lower bound")
    return lo, hi


def load_grain_moisture_table(
    path: str | Path = DEFAULT_GRAIN_MOISTURE_TABLE,
) -> dict[str, Any]:
    """Load the user-editable ARVALIS/IRRINOV H50/H45 table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grain-moisture stage table not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "groups" not in raw:
        raise ValueError(f"Missing 'groups' section in {path}")
    return raw


def normalize_maturity_group(
    maturity_group: str,
    table: dict[str, Any],
) -> str:
    """Return a canonical G0..G6 group, accepting table aliases."""
    value = str(maturity_group).strip()
    groups = table["groups"]

    if value in groups:
        return value

    upper = value.upper().replace(" ", "_").replace("-", "_")
    for group in groups:
        if group.upper() == upper:
            return group

    aliases = table.get("aliases", {})
    for alias, canonical in aliases.items():
        alias_norm = str(alias).upper().replace(" ", "_").replace("-", "_")
        if upper == alias_norm:
            if canonical not in groups:
                raise ValueError(f"Alias {alias!r} maps to unknown maturity group {canonical!r}")
            return str(canonical)

    raise ValueError(
        f"Unknown maize maturity group {maturity_group!r}. "
        f"Available groups: {sorted(groups)}; aliases: {sorted(aliases)}"
    )


def thresholds_for_crop(
    crop_cfg: dict[str, Any],
    *,
    table_path: str | Path = DEFAULT_GRAIN_MOISTURE_TABLE,
) -> GrainMoistureThresholds:
    """Build H50/H45 thresholds for one parcel crop configuration.

    A variety-specific flowering GDD from ``crop.gdd_thresholds.flowering``
    takes priority.  If it is unavailable, the midpoint of the maturity-group
    sowing-to-flowering range is used.

    Optional parcel overrides can be supplied as::

        grain_moisture_stages:
          flowering_to_h50_gdd: 615
          flowering_to_h45_gdd: 690

    This lets a future hybrid-specific calibration override the group table.
    """
    table = load_grain_moisture_table(table_path)

    maturity_group = crop_cfg.get("maturity_group")
    if maturity_group is None:
        raise ValueError("crop.maturity_group is required for H50/H45 stage estimation")

    group = normalize_maturity_group(str(maturity_group), table)
    row = table["groups"][group]

    thermal = table.get("thermal_time", {})
    t_base = float(thermal.get("base_c", 6.0))
    t_upper = float(thermal.get("upper_c", 30.0))

    gdd_cfg = crop_cfg.get("gdd_thresholds") or {}
    flowering = gdd_cfg.get("flowering")
    if flowering is None:
        flowering_range = row.get("sowing_to_flowering_gdd_range")
        if flowering_range is None:
            raise ValueError(
                f"No crop.gdd_thresholds.flowering and no group flowering range for {group}"
            )
        flowering = _midpoint(flowering_range)

    h50_range = _as_range(row.get("flowering_to_h50_gdd_range"))
    h45_range = _as_range(row.get("flowering_to_h45_gdd_range"))

    overrides = crop_cfg.get("grain_moisture_stages") or {}
    h50 = overrides.get("flowering_to_h50_gdd")
    h45 = overrides.get("flowering_to_h45_gdd")

    if h50 is None:
        if h50_range is None:
            raise ValueError(f"No H50 GDD requirement configured for group {group}")
        h50 = _midpoint(h50_range)

    if h45 is None:
        if h45_range is None:
            raise ValueError(f"No H45 GDD requirement configured for group {group}")
        h45 = _midpoint(h45_range)

    h50 = float(h50)
    h45 = float(h45)
    if h45 <= h50:
        raise ValueError(
            f"Invalid grain-moisture thresholds for {group}: "
            f"H45 offset {h45} must exceed H50 offset {h50}"
        )

    return GrainMoistureThresholds(
        maturity_group=group,
        label=str(row.get("label", group)),
        flowering_gdd=float(flowering),
        h50_after_flowering_gdd=h50,
        h45_after_flowering_gdd=h45,
        t_base_c=t_base,
        t_upper_c=t_upper,
        h50_after_flowering_range=h50_range,
        h45_after_flowering_range=h45_range,
    )


def arvalis_daily_gdd(
    t_min: float | np.ndarray,
    t_max: float | np.ndarray,
    *,
    t_base: float = 6.0,
    t_upper: float = 30.0,
) -> float | np.ndarray:
    """Compute maize thermal time using the ARVALIS base 6-30 convention.

    The daily maximum is capped at ``t_upper``.  Degree-days are then the
    positive part of daily mean temperature minus ``t_base``.  The resulting
    mean is defensively capped at ``t_upper`` as well for exceptionally hot
    nights.
    """
    tmin = np.asarray(t_min, dtype=float)
    tmax = np.asarray(t_max, dtype=float)

    tmax_eff = np.minimum(tmax, float(t_upper))
    tmean_eff = np.minimum((tmin + tmax_eff) / 2.0, float(t_upper))
    result = np.maximum(0.0, tmean_eff - float(t_base))

    if result.ndim == 0:
        return float(result)
    return result


def cumulative_gdd_series(
    forcing: DailyForcing,
    *,
    planting_date: date,
    t_base: float = 6.0,
    t_upper: float = 30.0,
) -> pd.Series:
    """Return cumulative base-6-30 GDD indexed by forcing date."""
    dates = pd.to_datetime(forcing.dates).normalize()
    frame = pd.DataFrame(
        {
            "t_min": np.asarray(forcing.t_min, dtype=float),
            "t_max": np.asarray(forcing.t_max, dtype=float),
        },
        index=dates,
    )
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    frame = frame.loc[frame.index.date >= planting_date]

    if frame.empty:
        return pd.Series(dtype=float, name="cumulative_gdd")

    daily = arvalis_daily_gdd(
        frame["t_min"].to_numpy(),
        frame["t_max"].to_numpy(),
        t_base=t_base,
        t_upper=t_upper,
    )
    return pd.Series(
        np.cumsum(daily),
        index=frame.index,
        name="cumulative_gdd",
    )


def cumulative_gdd_on_date(
    forcing: DailyForcing,
    *,
    planting_date: date,
    on_date: date,
    t_base: float = 6.0,
    t_upper: float = 30.0,
) -> float:
    """Return accumulated thermal time through ``on_date``."""
    series = cumulative_gdd_series(
        forcing,
        planting_date=planting_date,
        t_base=t_base,
        t_upper=t_upper,
    )
    if series.empty:
        return 0.0

    eligible = series.loc[series.index.date <= on_date]
    return float(eligible.iloc[-1]) if len(eligible) else 0.0


def grain_moisture_stage(
    cumulative_gdd: float,
    thresholds: GrainMoistureThresholds,
) -> GrainMoistureStage:
    """Classify thermal progress relative to H50/H45."""
    if cumulative_gdd < thresholds.h50_gdd:
        return GrainMoistureStage.PRE_H50
    if cumulative_gdd < thresholds.h45_gdd:
        return GrainMoistureStage.H50_TO_H45
    return GrainMoistureStage.POST_H45


def threshold_crossing_date(
    forcing: DailyForcing,
    *,
    planting_date: date,
    threshold_gdd: float,
    t_base: float = 6.0,
    t_upper: float = 30.0,
) -> date | None:
    """Return the first forcing date whose cumulative GDD reaches a threshold."""
    series = cumulative_gdd_series(
        forcing,
        planting_date=planting_date,
        t_base=t_base,
        t_upper=t_upper,
    )
    crossed = series[series >= float(threshold_gdd)]
    if crossed.empty:
        return None
    return crossed.index[0].date()


def forecast_threshold_date(
    base_forcing: DailyForcing,
    member_forcings: dict[int, DailyForcing],
    *,
    planting_date: date,
    threshold_gdd: float,
    t_base: float = 6.0,
    t_upper: float = 30.0,
    quantile: float = 0.50,
    min_member_fraction: float = 0.80,
) -> ForecastThresholdDate:
    """Estimate a common threshold date from deterministic + ensemble forcing.

    Each ensemble member is appended to the deterministic/history prefix and
    gets its own threshold-crossing date.  The common date is the requested
    date quantile (median by default).  It is returned only if at least
    ``min_member_fraction`` of members reach the threshold inside the supplied
    forecast horizon.
    """
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    if not 0 < min_member_fraction <= 1:
        raise ValueError("min_member_fraction must be in (0, 1]")

    # Deterministic-only path, useful if H45 falls inside the HRES segment.
    if not member_forcings:
        crossing = threshold_crossing_date(
            base_forcing,
            planting_date=planting_date,
            threshold_gdd=threshold_gdd,
            t_base=t_base,
            t_upper=t_upper,
        )
        return ForecastThresholdDate(
            date=crossing,
            member_dates={0: crossing},
            coverage=1.0 if crossing is not None else 0.0,
            quantile=quantile,
        )

    member_dates: dict[int, date | None] = {}
    reached: list[date] = []

    for member, forcing in sorted(member_forcings.items()):
        full = base_forcing.concat(forcing)
        crossing = threshold_crossing_date(
            full,
            planting_date=planting_date,
            threshold_gdd=threshold_gdd,
            t_base=t_base,
            t_upper=t_upper,
        )
        member_dates[int(member)] = crossing
        if crossing is not None:
            reached.append(crossing)

    coverage = len(reached) / len(member_forcings)
    if not reached or coverage < min_member_fraction:
        return ForecastThresholdDate(
            date=None,
            member_dates=member_dates,
            coverage=float(coverage),
            quantile=quantile,
        )

    ordinals = np.asarray([d.toordinal() for d in reached], dtype=float)
    common = date.fromordinal(int(round(float(np.quantile(ordinals, quantile)))))

    return ForecastThresholdDate(
        date=common,
        member_dates=member_dates,
        coverage=float(coverage),
        quantile=quantile,
    )
