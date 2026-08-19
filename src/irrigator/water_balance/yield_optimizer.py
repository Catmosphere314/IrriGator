"""Yield-constrained irrigation optimization for AquaCrop.

The optimizer deliberately does *not* minimize water-stress diagnostics.
Stress is used only to locate agronomically meaningful intervention windows.
The objective is the smallest irrigation schedule that preserves dry yield
within a configurable tolerance of a reference schedule under the same weather.

Two workflows are provided:

``optimize_historical_irrigation_amounts``
    Keep the farmer's recorded dates fixed and reduce/remove doses while
    preserving the recorded schedule's dry yield (or its projected ensemble
    yield for an ongoing season).

``optimize_operational_irrigation``
    Receding-horizon forecast optimization.  A no-future-irrigation run first
    determines whether intervention is needed.  If it is, AquaCrop stress is
    used only to propose event dates; doses are then minimized against an
    ensemble dry-yield constraint.  This replaces the old combinatorial
    enumeration of arbitrary day/dose schedules.

The implementation reuses ``run_aquacrop`` and the existing candidate schedule
builder rather than maintaining a second crop-model integration.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from irrigator.atmospheric.forcing import DailyForcing
from irrigator.config import ParcelConfig
from irrigator.static_layers.soil import SoilProfile
from irrigator.static_layers.terrain import TerrainParams
from irrigator.water_balance.aquacrop_adapter import (
    AquaCropResult,
    AquaCropYieldBranchingEvaluator,
    IrrigationCandidate,
    _build_candidate_irrigation,
    run_aquacrop,
)

logger = logging.getLogger(__name__)

_WEEKDAY = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass(frozen=True)
class YieldConstraint:
    """Acceptable loss relative to the member-wise reference dry yield."""

    max_relative_loss: float = 0.01
    min_member_probability: float = 0.80
    absolute_tolerance_t_ha: float = 0.01

    def __post_init__(self) -> None:
        if not 0 <= self.max_relative_loss < 1:
            raise ValueError("max_relative_loss must be in [0, 1).")
        if not 0 < self.min_member_probability <= 1:
            raise ValueError("min_member_probability must be in (0, 1].")
        if self.absolute_tolerance_t_ha < 0:
            raise ValueError("absolute_tolerance_t_ha must be >= 0.")


@dataclass
class YieldScheduleEvaluation:
    """Member-wise AquaCrop outcome for one candidate future schedule."""

    candidate: IrrigationCandidate
    member_yield_t_ha: dict[int, float]
    first_stress_date: dict[int, date | None]

    @property
    def total_irrigation_mm(self) -> float:
        return self.candidate.total_mm

    @property
    def mean_yield_t_ha(self) -> float:
        return float(np.mean(list(self.member_yield_t_ha.values())))

    def yield_ratios(self, reference_yields: dict[int, float]) -> dict[int, float]:
        ratios: dict[int, float] = {}
        for member, value in self.member_yield_t_ha.items():
            ref = float(reference_yields[member])
            ratios[member] = 1.0 if ref <= 1e-12 else float(value / ref)
        return ratios

    def success_probability(
        self,
        reference_yields: dict[int, float],
        constraint: YieldConstraint,
    ) -> float:
        outcomes = []
        for member, value in self.member_yield_t_ha.items():
            ref = float(reference_yields[member])
            threshold = (1.0 - constraint.max_relative_loss) * ref
            outcomes.append(value + constraint.absolute_tolerance_t_ha >= threshold)
        return float(np.mean(outcomes)) if outcomes else 0.0

    def satisfies(
        self,
        reference_yields: dict[int, float],
        constraint: YieldConstraint,
    ) -> bool:
        return (
            self.success_probability(reference_yields, constraint)
            >= constraint.min_member_probability
        )


@dataclass
class HistoricalOptimizationResult:
    """Dose-only counterfactual on the farmer's recorded irrigation dates."""

    original_events: list[tuple[date, float]]
    optimized_events: list[tuple[date, float]]
    reference_yields_t_ha: dict[int, float]
    optimized_evaluation: YieldScheduleEvaluation
    constraint: YieldConstraint

    @property
    def original_mm(self) -> float:
        return float(sum(v for _, v in self.original_events))

    @property
    def optimized_mm(self) -> float:
        return float(sum(v for _, v in self.optimized_events))

    @property
    def saved_mm(self) -> float:
        return self.original_mm - self.optimized_mm

    @property
    def saved_fraction(self) -> float:
        return self.saved_mm / self.original_mm if self.original_mm > 0 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        optimized = {d: q for d, q in self.optimized_events}
        rows = []
        for d, original in self.original_events:
            new = float(optimized.get(d, 0.0))
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "recorded_mm": float(original),
                    "optimized_mm": new,
                    "saved_mm": float(original - new),
                    "removed": bool(new <= 1e-9),
                }
            )
        return pd.DataFrame(rows)


@dataclass
class OperationalOptimizationResult:
    """Daily recommendation plus diagnostic reference/no-irrigation outcomes."""

    today: date
    recommended_events: list[tuple[date, float]]
    committed_dates: set[date]
    reference_evaluation: YieldScheduleEvaluation
    no_irrigation_evaluation: YieldScheduleEvaluation
    optimized_evaluation: YieldScheduleEvaluation
    reference_yields_t_ha: dict[int, float]
    constraint: YieldConstraint
    feasible: bool
    reason: str

    @property
    def recommended_total_mm(self) -> float:
        return float(sum(q for _, q in self.recommended_events))

    @property
    def success_probability(self) -> float:
        return self.optimized_evaluation.success_probability(
            self.reference_yields_t_ha, self.constraint
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(d),
                    "day_offset": (d - self.today).days,
                    "dose_mm": float(q),
                    "committed": d in self.committed_dates,
                }
                for d, q in self.recommended_events
            ]
        )


# ---------------------------------------------------------------------------
# AquaCrop evaluation engine
# ---------------------------------------------------------------------------


def _extract_dry_yield(result: AquaCropResult) -> float:
    final = result.final_results
    if len(final) and "Dry yield (tonne/ha)" in final.columns:
        value = float(final.iloc[-1]["Dry yield (tonne/ha)"])
        if np.isfinite(value):
            return value

    cg = result.crop_growth
    if "DryYield" in cg.columns:
        values = cg["DryYield"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite):
            return float(finite[-1])

    # Same partial-season fallback already used by aquacrop_adapter.
    if len(cg):
        row = cg.iloc[-2] if len(cg) > 1 else cg.iloc[-1]
        biomass = float(row.get("biomass", 0.0))
        hi = float(row.get("harvest_index", 0.0))
        return biomass * hi / 1000.0
    return 0.0


def _first_stress_date(
    result: AquaCropResult,
    *,
    threshold_ks: float,
    not_before: date | None,
) -> date | None:
    if result.weather_daily is None:
        return None
    stress = result.daily_stress
    dates = pd.to_datetime(result.weather_daily["Date"].iloc[: len(stress)]).dt.date
    mask = stress["ks"].to_numpy(dtype=float) < threshold_ks
    if not_before is not None:
        mask &= np.asarray([d >= not_before for d in dates], dtype=bool)
    idx = np.flatnonzero(mask)
    return dates.iloc[int(idx[0])] if len(idx) else None


_YIELD_WORKER_CTX: dict[str, object] = {}


def _init_yield_worker(
    base_forcing: DailyForcing,
    member_forcings: dict[int, DailyForcing],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    sim_end: date,
    anchor: date,
) -> None:
    _YIELD_WORKER_CTX.clear()
    _YIELD_WORKER_CTX.update(
        {
            "base_forcing": base_forcing,
            "member_forcings": member_forcings,
            "parcel": parcel,
            "terrain": terrain,
            "soil_profile": soil_profile,
            "sim_start": sim_start,
            "sim_end": sim_end,
            "anchor": anchor,
        }
    )


def _yield_worker(
    job: tuple[int, IrrigationCandidate, float, date | None],
) -> tuple[int, float, date | None]:
    member, candidate, stress_threshold_ks, stress_not_before = job
    ctx = _YIELD_WORKER_CTX
    forcing = ctx["base_forcing"].concat(ctx["member_forcings"][member])  # type: ignore[union-attr,index]
    irrigation = _build_candidate_irrigation(
        ctx["parcel"],
        candidate,
        ctx["anchor"],  # type: ignore[arg-type]
    )
    result = run_aquacrop(
        forcing=forcing,
        parcel=ctx["parcel"],  # type: ignore[arg-type]
        terrain=ctx["terrain"],  # type: ignore[arg-type]
        soil_profile=ctx["soil_profile"],  # type: ignore[arg-type]
        sim_start=ctx["sim_start"],  # type: ignore[arg-type]
        sim_end=ctx["sim_end"],  # type: ignore[arg-type]
        irrigation_management=irrigation,
    )
    return (
        int(member),
        _extract_dry_yield(result),
        _first_stress_date(
            result,
            threshold_ks=stress_threshold_ks,
            not_before=stress_not_before,
        ),
    )


class _YieldEvaluator:
    """Cache schedule evaluations and optionally reuse a process pool."""

    def __init__(
        self,
        *,
        base_forcing: DailyForcing,
        member_forcings: dict[int, DailyForcing] | None,
        parcel: ParcelConfig,
        terrain: TerrainParams,
        soil_profile: SoilProfile,
        sim_start: date,
        sim_end: date,
        anchor: date,
        n_workers: int = 1,
        branching_historical_forcing: DailyForcing | None = None,
        branching_arome_forcing: DailyForcing | None = None,
    ) -> None:
        self.base_forcing = base_forcing
        self.member_forcings = member_forcings or {}
        self.parcel = parcel
        self.terrain = terrain
        self.soil_profile = soil_profile
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.anchor = anchor
        self.n_workers = max(1, int(n_workers))
        self.cache: dict[tuple, YieldScheduleEvaluation] = {}
        self.pool: mp.pool.Pool | None = None
        self.branching: AquaCropYieldBranchingEvaluator | None = None

        # Operational optimization has a natural deterministic/ensemble branch
        # boundary.  Reuse the private AquaCrop state stepper there instead of
        # restarting the crop from sim_start for every adaptive schedule trial.
        if (
            self.member_forcings
            and branching_historical_forcing is not None
            and branching_arome_forcing is not None
        ):
            member_end = min(
                pd.Timestamp(f.dates[-1]).date() for f in self.member_forcings.values()
            )
            if member_end != self.sim_end:
                logger.warning(
                    "Fast yield branching disabled because sim_end=%s differs from "
                    "member horizon=%s.",
                    self.sim_end,
                    member_end,
                )
            else:
                self.branching = AquaCropYieldBranchingEvaluator(
                    historical_forcing=branching_historical_forcing,
                    arome_forcing=branching_arome_forcing,
                    member_forcings=self.member_forcings,
                    parcel=parcel,
                    terrain=terrain,
                    soil_profile=soil_profile,
                    sim_start=sim_start,
                    today=anchor,
                    workers=self.n_workers,
                )

        if self.branching is None and self.member_forcings and self.n_workers > 1:
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
            self.pool = ctx.Pool(
                processes=min(self.n_workers, len(self.member_forcings)),
                initializer=_init_yield_worker,
                initargs=(
                    base_forcing,
                    self.member_forcings,
                    parcel,
                    terrain,
                    soil_profile,
                    sim_start,
                    sim_end,
                    anchor,
                ),
            )

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self) -> "_YieldEvaluator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.pool is not None:
            if exc_type is None:
                self.pool.close()
            else:
                self.pool.terminate()
            self.pool.join()
            self.pool = None

    @staticmethod
    def _key(
        candidate: IrrigationCandidate,
        stress_threshold_ks: float,
        stress_not_before: date | None,
    ) -> tuple:
        events = tuple((int(d), round(float(q), 4)) for d, q in candidate.events if q > 1e-9)
        return events, round(float(stress_threshold_ks), 4), stress_not_before

    def evaluate(
        self,
        candidate: IrrigationCandidate,
        *,
        stress_threshold_ks: float = 0.98,
        stress_not_before: date | None = None,
    ) -> YieldScheduleEvaluation:
        key = self._key(candidate, stress_threshold_ks, stress_not_before)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.branching is not None:
            branch_rows = self.branching.evaluate(
                candidate,
                stress_threshold_ks=stress_threshold_ks,
                stress_not_before=stress_not_before,
            )
            outputs = [
                (m, row.dry_yield_t_ha, row.first_stress_date)
                for m, row in sorted(branch_rows.items())
            ]
        elif self.member_forcings:
            jobs = [
                (int(member), candidate, float(stress_threshold_ks), stress_not_before)
                for member in sorted(self.member_forcings)
            ]
            if self.pool is not None:
                outputs = self.pool.map(_yield_worker, jobs)
            else:
                # Keep the serial path debuggable without touching global state.
                outputs = []
                for member, _, _, _ in jobs:
                    forcing = self.base_forcing.concat(self.member_forcings[member])
                    irrigation = _build_candidate_irrigation(self.parcel, candidate, self.anchor)
                    result = run_aquacrop(
                        forcing=forcing,
                        parcel=self.parcel,
                        terrain=self.terrain,
                        soil_profile=self.soil_profile,
                        sim_start=self.sim_start,
                        sim_end=self.sim_end,
                        irrigation_management=irrigation,
                    )
                    outputs.append(
                        (
                            member,
                            _extract_dry_yield(result),
                            _first_stress_date(
                                result,
                                threshold_ks=stress_threshold_ks,
                                not_before=stress_not_before,
                            ),
                        )
                    )
        else:
            irrigation = _build_candidate_irrigation(self.parcel, candidate, self.anchor)
            result = run_aquacrop(
                forcing=self.base_forcing,
                parcel=self.parcel,
                terrain=self.terrain,
                soil_profile=self.soil_profile,
                sim_start=self.sim_start,
                sim_end=self.sim_end,
                irrigation_management=irrigation,
            )
            outputs = [
                (
                    0,
                    _extract_dry_yield(result),
                    _first_stress_date(
                        result,
                        threshold_ks=stress_threshold_ks,
                        not_before=stress_not_before,
                    ),
                )
            ]

        evaluation = YieldScheduleEvaluation(
            candidate=candidate,
            member_yield_t_ha={m: float(y) for m, y, _ in outputs},
            first_stress_date={m: d for m, _, d in outputs},
        )
        self.cache[key] = evaluation
        return evaluation


# ---------------------------------------------------------------------------
# Shared schedule helpers
# ---------------------------------------------------------------------------


def _expected_harvest(parcel: ParcelConfig) -> date | None:
    value = parcel.crop.get("expected_harvest")
    return pd.Timestamp(value).date() if value else None


def _end_date_from_forcings(
    base: DailyForcing,
    members: dict[int, DailyForcing] | None,
) -> date:
    if members:
        # Every member must support the common objective horizon.
        return min(np.datetime64(v.dates[-1], "D").astype(object) for v in members.values())
    return np.datetime64(base.dates[-1], "D").astype(object)


def _candidate_from_dates(
    events: Iterable[tuple[date, float]],
    *,
    anchor: date,
    label: str,
) -> IrrigationCandidate:
    return IrrigationCandidate(
        events=[
            ((d - anchor).days, float(q)) for d, q in sorted(events) if d >= anchor and q > 1e-9
        ],
        label=label,
    )


def _candidate_events_as_dates(
    candidate: IrrigationCandidate, anchor: date
) -> list[tuple[date, float]]:
    return [
        (anchor + timedelta(days=int(offset)), float(dose))
        for offset, dose in candidate.events
        if dose > 1e-9
    ]


def _available_weekdays(parcel: ParcelConfig) -> set[int]:
    values = parcel.irrigation.get("available_days")
    if not values:
        return set(range(7))
    result = set()
    for value in values:
        key = str(value).strip().lower()
        if key not in _WEEKDAY:
            raise ValueError(f"Unknown irrigation weekday: {value!r}")
        result.add(_WEEKDAY[key])
    return result


def _next_available(d: date, weekdays: set[int]) -> date:
    for shift in range(8):
        candidate = d + timedelta(days=shift)
        if candidate.weekday() in weekdays:
            return candidate
    raise RuntimeError("No available irrigation weekday.")


def _latest_available_at_or_before(d: date, lower: date, weekdays: set[int]) -> date | None:
    cur = d
    while cur >= lower:
        if cur.weekday() in weekdays:
            return cur
        cur -= timedelta(days=1)
    return None


def _last_recorded_irrigation(parcel: ParcelConfig, today: date) -> date | None:
    dates = [
        pd.Timestamp(row["date"]).date()
        for row in parcel.irrigation_log
        if row.get("date") and pd.Timestamp(row["date"]).date() <= today
    ]
    return max(dates) if dates else None


def _max_feasible_events(
    parcel: ParcelConfig,
    *,
    today: date,
    sim_end: date,
    earliest_date: date | None = None,
) -> list[tuple[date, float]]:
    min_interval = int(parcel.irrigation.get("min_interval_days", 3))
    max_dose = float(parcel.irrigation.get("max_dose_mm", 40.0))
    weekdays = _available_weekdays(parcel)
    last = _last_recorded_irrigation(parcel, today)

    start = earliest_date or (today + timedelta(days=1))
    if last is not None:
        start = max(start, last + timedelta(days=min_interval))
    current = _next_available(start, weekdays)

    events: list[tuple[date, float]] = []
    while current <= sim_end:
        events.append((current, max_dose))
        current = _next_available(current + timedelta(days=min_interval), weekdays)
    return events


def _reference_yields(
    evaluations: Iterable[YieldScheduleEvaluation],
) -> dict[int, float]:
    """Member-wise attainable-yield benchmark from several wet schedules.

    A single "maximum irrigation" schedule can itself be suboptimal because of
    drainage/aeration effects.  Taking the member-wise maximum across a small
    family of wet schedules plus the no-irrigation case gives a more robust
    operational reference without introducing a second optimization problem.
    """
    rows = list(evaluations)
    if not rows:
        raise ValueError("At least one evaluation is required for the yield reference.")
    members = rows[0].member_yield_t_ha
    return {member: max(float(row.member_yield_t_ha[member]) for row in rows) for member in members}


def _quantized_levels(
    *,
    max_dose: float,
    min_dose: float,
    resolution: float,
    allow_zero: bool,
) -> list[float]:
    if resolution <= 0:
        raise ValueError("dose_resolution_mm must be > 0.")
    positive = list(np.arange(min_dose, max_dose + resolution * 0.5, resolution, dtype=float))
    if not positive or positive[-1] < max_dose - 1e-9:
        positive.append(max_dose)
    positive = sorted(set(round(min(max_dose, q), 6) for q in positive if q <= max_dose + 1e-9))
    return ([0.0] if allow_zero else []) + positive


def _minimize_event_doses(
    evaluator: _YieldEvaluator,
    events: list[tuple[date, float]],
    *,
    anchor: date,
    reference_yields: dict[int, float],
    constraint: YieldConstraint,
    min_dose: float,
    max_dose: float,
    dose_resolution_mm: float,
    committed_dates: set[date],
    max_passes: int = 2,
) -> tuple[list[tuple[date, float]], YieldScheduleEvaluation]:
    """Coordinate-wise monotone dose reduction; no date combinations are enumerated."""
    current = [(d, float(q)) for d, q in events]
    current_eval = evaluator.evaluate(
        _candidate_from_dates(current, anchor=anchor, label="dose-search")
    )

    for _ in range(max_passes):
        changed = False
        # Latest first: remove unnecessary late-season water before modifying
        # water that may have enabled earlier canopy development.
        for idx in range(len(current) - 1, -1, -1):
            event_date, old_dose = current[idx]
            is_committed = event_date in committed_dates
            effective_min_dose = max(min_dose, dose_resolution_mm) if is_committed else min_dose
            levels = _quantized_levels(
                max_dose=min(max_dose, old_dose),
                min_dose=effective_min_dose,
                resolution=dose_resolution_mm,
                allow_zero=not is_committed,
            )
            if not levels:
                continue

            lo, hi = 0, len(levels) - 1
            best_dose = old_dose
            best_eval = current_eval
            while lo <= hi:
                mid = (lo + hi) // 2
                trial_dose = levels[mid]
                trial = current.copy()
                trial[idx] = (event_date, trial_dose)
                candidate = _candidate_from_dates(trial, anchor=anchor, label="dose-search")
                evaluation = evaluator.evaluate(candidate)
                if evaluation.satisfies(reference_yields, constraint):
                    best_dose = trial_dose
                    best_eval = evaluation
                    hi = mid - 1
                else:
                    lo = mid + 1

            if best_dose < old_dose - 1e-9:
                current[idx] = (event_date, best_dose)
                current_eval = best_eval
                changed = True

        if not changed:
            break

    current = [(d, q) for d, q in current if q > 1e-9 or d in committed_dates]
    final_candidate = _candidate_from_dates(current, anchor=anchor, label="optimized")
    final_eval = evaluator.evaluate(final_candidate)
    return current, final_eval


# ---------------------------------------------------------------------------
# Retrospective / ongoing historical dose optimization
# ---------------------------------------------------------------------------


def optimize_historical_irrigation_amounts(
    observed_forcing: DailyForcing,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    *,
    sim_end: date | None = None,
    future_member_forcings: dict[int, DailyForcing] | None = None,
    shared_future_events: list[tuple[date, float]] | None = None,
    constraint: YieldConstraint = YieldConstraint(
        max_relative_loss=0.005, min_member_probability=1.0
    ),
    dose_resolution_mm: float = 1.0,
    n_workers: int = 1,
    max_passes: int = 3,
) -> HistoricalOptimizationResult:
    """Reduce doses on recorded dates while preserving reference dry yield.

    Completed season
        Pass the complete observed forcing and no ``future_member_forcings``.
        The recorded schedule is the member-wise yield reference.

    Ongoing season
        Pass observations through today plus member-wise weather through
        harvest.  ``shared_future_events`` are appended *identically* to the
        recorded and counterfactual schedules so that only past irrigation is
        changed.  If omitted, the common future policy is no further irrigation.
    """
    original = [
        (pd.Timestamp(row["date"]).date(), float(row.get("amount_mm", row.get("Depth", 0.0))))
        for row in parcel.irrigation_log
        if row.get("date")
    ]
    if not original:
        raise ValueError("parcel.irrigation_log contains no irrigation events.")

    if sim_end is None:
        sim_end = _end_date_from_forcings(observed_forcing, future_member_forcings)

    # Prevent parcel_to_irrigation from adding the original schedule a second time.
    blank_parcel = replace(parcel, irrigation_log=[])
    fixed_future = shared_future_events or []
    reference_events = original + fixed_future
    anchor = sim_start

    with _YieldEvaluator(
        base_forcing=observed_forcing,
        member_forcings=future_member_forcings,
        parcel=blank_parcel,
        terrain=terrain,
        soil_profile=soil_profile,
        sim_start=sim_start,
        sim_end=sim_end,
        anchor=anchor,
        n_workers=n_workers,
    ) as evaluator:
        reference_eval = evaluator.evaluate(
            _candidate_from_dates(reference_events, anchor=anchor, label="recorded schedule")
        )
        reference_yields = dict(reference_eval.member_yield_t_ha)

        min_dose = float(parcel.irrigation.get("min_dose_mm", 0.0))
        max_dose = float(parcel.irrigation.get("max_dose_mm", max(q for _, q in original)))
        current = list(reference_events)
        fixed_dates = {d for d, _ in fixed_future}

        # Only recorded events are mutable; future policy remains common.
        for _ in range(max_passes):
            changed = False
            for event_date, _ in reversed(original):
                idx = next(i for i, (d, _) in enumerate(current) if d == event_date)
                old_dose = current[idx][1]
                levels = _quantized_levels(
                    max_dose=min(max_dose, old_dose),
                    min_dose=min(min_dose, old_dose) if min_dose > 0 else dose_resolution_mm,
                    resolution=dose_resolution_mm,
                    allow_zero=True,
                )
                lo, hi = 0, len(levels) - 1
                best = old_dose
                while lo <= hi:
                    mid = (lo + hi) // 2
                    trial = current.copy()
                    trial[idx] = (event_date, levels[mid])
                    evaluation = evaluator.evaluate(
                        _candidate_from_dates(trial, anchor=anchor, label="historical dose search")
                    )
                    if evaluation.satisfies(reference_yields, constraint):
                        best = levels[mid]
                        hi = mid - 1
                    else:
                        lo = mid + 1
                if best < old_dose - 1e-9:
                    current[idx] = (event_date, best)
                    changed = True
            if not changed:
                break

        optimized_historical = [(d, q) for d, q in current if d not in fixed_dates and q > 1e-9]
        full_optimized = [(d, q) for d, q in current if q > 1e-9]
        optimized_eval = evaluator.evaluate(
            _candidate_from_dates(full_optimized, anchor=anchor, label="optimized historical")
        )

    return HistoricalOptimizationResult(
        original_events=original,
        optimized_events=optimized_historical,
        reference_yields_t_ha=reference_yields,
        optimized_evaluation=optimized_eval,
        constraint=constraint,
    )


# ---------------------------------------------------------------------------
# Operational event-triggered optimizer
# ---------------------------------------------------------------------------


def optimize_operational_irrigation(
    historical_forcing: DailyForcing,
    arome_forcing: DailyForcing,
    future_member_forcings: dict[int, DailyForcing],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    *,
    today: date,
    sim_start: date,
    sim_end: date | None = None,
    previous_plan: list[tuple[date, float]] | None = None,
    commitment_days: int = 2,
    constraint: YieldConstraint = YieldConstraint(),
    stress_trigger_ks: float = 0.98,
    trigger_quantile: float = 0.25,
    irrigation_lead_days: int = 1,
    dose_resolution_mm: float = 2.0,
    max_events: int = 8,
    n_workers: int = 1,
    require_harvest_horizon: bool = True,
) -> OperationalOptimizationResult:
    """Find a sparse, yield-preserving forecast irrigation plan.

    ``future_member_forcings`` should start after the deterministic AROME
    horizon and extend to harvest.  In normal use this is IFS ENS extended with
    ``extend_ifs_members_with_seasonal``.  The function evaluates only a small
    number of event-driven schedules instead of constructing every possible
    day/dose combination.
    """
    if not future_member_forcings:
        raise ValueError("future_member_forcings is empty.")
    if not 0 <= trigger_quantile <= 1:
        raise ValueError("trigger_quantile must be in [0, 1].")
    if max_events < 1:
        raise ValueError("max_events must be >= 1.")

    base_forcing = historical_forcing.concat(arome_forcing)

    # Maximum date supported by every forecast member.
    common_end = _end_date_from_forcings(
        base_forcing,
        future_member_forcings,
    )

    # User-requested horizon, constrained by available weather.
    sim_end = common_end if sim_end is None else min(sim_end, common_end)

    expected_harvest = _expected_harvest(parcel)

    # First check that enough forecast weather exists to cover the crop season.
    if require_harvest_horizon and expected_harvest is not None and sim_end < expected_harvest:
        raise ValueError(
            f"Yield optimization needs weather through expected harvest {expected_harvest}, "
            f"but the common forecast horizon ends {sim_end}. Extend IFS members with the "
            "SEAS5-conditioned scenarios first (or set require_harvest_horizon=False for a "
            "diagnostic partial-horizon run)."
        )

    # There is no reason to optimize beyond the configured harvest date.
    if expected_harvest is not None:
        sim_end = min(sim_end, expected_harvest)

    # Truncate IFS+SEAS5 scenarios to the useful crop horizon.
    # This also keeps the fast branching evaluator active because member_end == sim_end.
    future_member_forcings = {
        member: forcing.slice(
            start=pd.Timestamp(forcing.dates[0]).date(),
            end=sim_end,
        )
        for member, forcing in future_member_forcings.items()
    }

    min_interval = int(parcel.irrigation.get("min_interval_days", 3))
    min_dose = float(parcel.irrigation.get("min_dose_mm", 0.0))
    max_dose = float(parcel.irrigation.get("max_dose_mm", 40.0))
    weekdays = _available_weekdays(parcel)
    anchor = today

    with _YieldEvaluator(
        base_forcing=base_forcing,
        member_forcings=future_member_forcings,
        parcel=parcel,
        terrain=terrain,
        soil_profile=soil_profile,
        sim_start=sim_start,
        sim_end=sim_end,
        anchor=anchor,
        n_workers=n_workers,
        branching_historical_forcing=historical_forcing,
        branching_arome_forcing=arome_forcing,
    ) as evaluator:
        no_candidate = IrrigationCandidate(events=[], label="No future irrigation")
        no_eval = evaluator.evaluate(
            no_candidate,
            stress_threshold_ks=stress_trigger_ks,
            stress_not_before=today + timedelta(days=1),
        )

        max_events_reference = _max_feasible_events(
            parcel,
            today=today,
            sim_end=sim_end,
        )
        reference_evaluations = [no_eval]
        wet_reference_evaluations: list[YieldScheduleEvaluation] = []
        for fraction in (0.50, 0.75, 1.00):
            wet_events = [(d, q * fraction) for d, q in max_events_reference]
            wet_eval = evaluator.evaluate(
                _candidate_from_dates(
                    wet_events,
                    anchor=anchor,
                    label=f"Wet reference {fraction:.0%}",
                )
            )
            wet_reference_evaluations.append(wet_eval)
            reference_evaluations.append(wet_eval)

        # Keep one concrete reference schedule for diagnostics, but use the
        # member-wise maximum across all wet/no-irrigation runs as the actual
        # yield constraint benchmark.
        reference_eval = max(
            wet_reference_evaluations,
            key=lambda row: row.mean_yield_t_ha,
        )
        reference_yields = _reference_yields(reference_evaluations)

        if no_eval.satisfies(reference_yields, constraint):
            return OperationalOptimizationResult(
                today=today,
                recommended_events=[],
                committed_dates=set(),
                reference_evaluation=reference_eval,
                no_irrigation_evaluation=no_eval,
                optimized_evaluation=no_eval,
                reference_yields_t_ha=reference_yields,
                constraint=constraint,
                feasible=True,
                reason="No future irrigation is needed to satisfy the yield constraint.",
            )

        commitment_limit = today + timedelta(days=max(0, commitment_days))
        committed_dates: set[date] = set()
        current_events: list[tuple[date, float]] = []
        for event_date, _dose in previous_plan or []:
            if today < event_date <= commitment_limit and event_date <= sim_end:
                committed_dates.add(event_date)
                # Keep the date, but start high and let the dose search adjust it.
                current_events.append((event_date, max_dose))

        # Remove duplicates while retaining date order.
        current_events = sorted({d: q for d, q in current_events}.items())
        search_not_before = today + timedelta(days=1)
        evaluation = evaluator.evaluate(
            _candidate_from_dates(current_events, anchor=anchor, label="event search"),
            stress_threshold_ks=stress_trigger_ks,
            stress_not_before=search_not_before,
        )

        feasible = True
        reason = "Yield constraint reached with event-triggered schedule."
        while not evaluation.satisfies(reference_yields, constraint):
            if len(current_events) >= max_events:
                feasible = False
                reason = f"Reached max_events={max_events} before satisfying the yield constraint."
                break

            stress_dates = [d for d in evaluation.first_stress_date.values() if d is not None]
            if not stress_dates:
                feasible = False
                reason = (
                    "The schedule misses the yield target but no water-stress trigger was found; "
                    "the remaining yield gap is not safely attributable to an irrigation event."
                )
                break

            ordinals = np.asarray([d.toordinal() for d in stress_dates], dtype=float)
            trigger = date.fromordinal(int(round(float(np.quantile(ordinals, trigger_quantile)))))
            desired = trigger - timedelta(days=max(0, irrigation_lead_days))

            last_actual = _last_recorded_irrigation(parcel, today)
            last_planned = max((d for d, _ in current_events), default=None)
            last_event = max(
                [d for d in (last_actual, last_planned) if d is not None], default=None
            )
            lower = today + timedelta(days=1)
            if last_event is not None:
                lower = max(lower, last_event + timedelta(days=min_interval))

            event_date = _latest_available_at_or_before(desired, lower, weekdays)
            if event_date is None:
                event_date = _next_available(lower, weekdays)
            if event_date > sim_end:
                feasible = False
                reason = "No feasible irrigation date remains inside the optimization horizon."
                break
            if any(d == event_date for d, _ in current_events):
                # Advance to the next feasible slot rather than looping on the
                # same stress trigger.
                event_date = _next_available(event_date + timedelta(days=min_interval), weekdays)
                if event_date > sim_end:
                    feasible = False
                    reason = "No additional feasible irrigation date remains."
                    break

            current_events.append((event_date, max_dose))
            current_events.sort()
            search_not_before = event_date + timedelta(days=1)
            evaluation = evaluator.evaluate(
                _candidate_from_dates(current_events, anchor=anchor, label="event search"),
                stress_threshold_ks=stress_trigger_ks,
                stress_not_before=search_not_before,
            )

        if feasible:
            current_events, optimized_eval = _minimize_event_doses(
                evaluator,
                current_events,
                anchor=anchor,
                reference_yields=reference_yields,
                constraint=constraint,
                min_dose=min_dose,
                max_dose=max_dose,
                dose_resolution_mm=dose_resolution_mm,
                committed_dates=committed_dates,
            )
            feasible = optimized_eval.satisfies(reference_yields, constraint)
            if not feasible:
                reason = "Dose reduction unexpectedly crossed the yield constraint; inspect monotonicity."
        else:
            optimized_eval = evaluation

    recommended = [(d, q) for d, q in current_events if q > 1e-9]
    return OperationalOptimizationResult(
        today=today,
        recommended_events=recommended,
        committed_dates=committed_dates,
        reference_evaluation=reference_eval,
        no_irrigation_evaluation=no_eval,
        optimized_evaluation=optimized_eval,
        reference_yields_t_ha=reference_yields,
        constraint=constraint,
        feasible=feasible,
        reason=reason,
    )
