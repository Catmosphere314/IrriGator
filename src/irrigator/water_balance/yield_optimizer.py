"""Crop-state/yield-constrained irrigation optimization for AquaCrop.

Water use remains the quantity being minimized.  Agronomic constraints depend
on the available forecast horizon:

* when weather reaches harvest, candidate schedules preserve final dry yield;
* on a partial forecast horizon (e.g. HRES + IFS without SEAS5), candidates
  preserve end-horizon biomass and canopy cover relative to the same-weather
  wet reference, and also preserve partial dry yield once it becomes material.

Irrigation dates are proposed from AquaCrop's crop-specific dynamic root-zone
depletion thresholds rather than from an arbitrary ``Tr / TrPot`` (Ks) value.
The trigger is only a search heuristic; crop-state/yield preservation determines
whether a candidate schedule is acceptable.

Two workflows are provided:

``optimize_historical_irrigation_amounts``
    Keep the farmer's recorded dates fixed and reduce/remove doses while
    preserving the recorded schedule's final dry yield.

``optimize_operational_irrigation``
    Receding-horizon forecast optimization.  A no-future-irrigation run first
    determines whether intervention is needed.  AquaCrop depletion thresholds
    propose agronomically meaningful event dates; doses are then minimized
    against either the harvest-yield or partial-horizon crop-state constraint.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

from irrigator.atmospheric.forcing import DailyForcing
from irrigator.config import ParcelConfig
from irrigator.static_layers.soil import SoilProfile
from irrigator.static_layers.terrain import TerrainParams
from irrigator.water_balance.aquacrop_adapter import (
    AquaCropResult,
    AquaCropYieldBranchingEvaluator,
    IrrigationCandidate,
    _build_candidate_irrigation,
    _soil_for_crop,
    parcel_to_crop,
    run_aquacrop,
)
from irrigator.water_balance.aquacrop_stress import (
    first_water_stress_trigger_date,
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
    """Acceptable crop degradation relative to a member-wise wet reference.

    ``max_relative_loss`` and ``absolute_tolerance_t_ha`` retain their original
    final-yield meaning.  The crop-state fields are used automatically when the
    operational forecast does not reach harvest.

    The defaults for biomass/canopy loss are intentionally configurable starting
    values, not calibrated agronomic constants.  Retrospective experiments should
    calibrate them over multiple parcel-years rather than one season.
    """

    max_relative_loss: float = 0.01
    min_member_probability: float = 0.80
    absolute_tolerance_t_ha: float = 0.01

    # Partial-horizon crop-state preservation.
    max_relative_biomass_loss: float = 0.02
    max_relative_canopy_loss: float = 0.02
    absolute_biomass_tolerance_t_ha: float = 0.05
    absolute_canopy_tolerance: float = 0.01

    # Do not form unstable ratios when a state variable is still effectively 0.
    min_reference_yield_t_ha: float = 0.05
    min_reference_biomass_t_ha: float = 0.10
    min_reference_canopy_cover: float = 0.05

    def __post_init__(self) -> None:
        relative_fields = {
            "max_relative_loss": self.max_relative_loss,
            "max_relative_biomass_loss": self.max_relative_biomass_loss,
            "max_relative_canopy_loss": self.max_relative_canopy_loss,
        }
        for name, value in relative_fields.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")

        if not 0 < self.min_member_probability <= 1:
            raise ValueError("min_member_probability must be in (0, 1].")

        nonnegative = {
            "absolute_tolerance_t_ha": self.absolute_tolerance_t_ha,
            "absolute_biomass_tolerance_t_ha": self.absolute_biomass_tolerance_t_ha,
            "absolute_canopy_tolerance": self.absolute_canopy_tolerance,
            "min_reference_yield_t_ha": self.min_reference_yield_t_ha,
            "min_reference_biomass_t_ha": self.min_reference_biomass_t_ha,
            "min_reference_canopy_cover": self.min_reference_canopy_cover,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if self.absolute_canopy_tolerance > 1:
            raise ValueError("absolute_canopy_tolerance must be <= 1.")
        if self.min_reference_canopy_cover > 1:
            raise ValueError("min_reference_canopy_cover must be <= 1.")


@dataclass(frozen=True)
class OperationalReferenceMetrics:
    """Member-wise best attainable state from the small wet-reference family."""

    member_yield_t_ha: dict[int, float]
    member_biomass_t_ha: dict[int, float]
    member_canopy_cover: dict[int, float]


@dataclass
class YieldScheduleEvaluation:
    """Member-wise AquaCrop outcome for one candidate future schedule."""

    candidate: IrrigationCandidate
    member_yield_t_ha: dict[int, float]
    member_biomass_t_ha: dict[int, float]
    member_canopy_cover: dict[int, float]
    first_stress_date: dict[int, date | None]
    first_stress_process: dict[int, str | None]

    @property
    def total_irrigation_mm(self) -> float:
        return self.candidate.total_mm

    @property
    def mean_yield_t_ha(self) -> float:
        return float(np.mean(list(self.member_yield_t_ha.values())))

    @property
    def mean_biomass_t_ha(self) -> float:
        return float(np.mean(list(self.member_biomass_t_ha.values())))

    @property
    def mean_canopy_cover(self) -> float:
        return float(np.mean(list(self.member_canopy_cover.values())))

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
        """Probability of satisfying the final/partial dry-yield criterion."""
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
        """Backward-compatible yield-only feasibility check."""
        return (
            self.success_probability(reference_yields, constraint)
            >= constraint.min_member_probability
        )

    def crop_state_success_probability(
        self,
        reference: OperationalReferenceMetrics,
        constraint: YieldConstraint,
    ) -> float:
        """Joint partial-horizon biomass/canopy/(when material) yield success.

        A member counts as successful only when every currently meaningful crop
        state is preserved.  Partial dry yield is included automatically once
        the wet-reference yield is large enough to form a stable comparison.
        """
        outcomes: list[bool] = []

        for member in self.member_yield_t_ha:
            checks: list[bool] = []

            ref_biomass = float(reference.member_biomass_t_ha[member])
            if ref_biomass >= constraint.min_reference_biomass_t_ha:
                biomass_threshold = (1.0 - constraint.max_relative_biomass_loss) * ref_biomass
                checks.append(
                    float(self.member_biomass_t_ha[member])
                    + constraint.absolute_biomass_tolerance_t_ha
                    >= biomass_threshold
                )

            ref_canopy = float(reference.member_canopy_cover[member])
            if ref_canopy >= constraint.min_reference_canopy_cover:
                canopy_threshold = (1.0 - constraint.max_relative_canopy_loss) * ref_canopy
                checks.append(
                    float(self.member_canopy_cover[member]) + constraint.absolute_canopy_tolerance
                    >= canopy_threshold
                )

            ref_yield = float(reference.member_yield_t_ha[member])
            if ref_yield >= constraint.min_reference_yield_t_ha:
                yield_threshold = (1.0 - constraint.max_relative_loss) * ref_yield
                checks.append(
                    float(self.member_yield_t_ha[member]) + constraint.absolute_tolerance_t_ha
                    >= yield_threshold
                )

            # Before emergence all reference states can legitimately be zero.
            # In that case irrigation is not required by crop-state preservation.
            outcomes.append(all(checks) if checks else True)

        return float(np.mean(outcomes)) if outcomes else 0.0

    def satisfies_crop_state(
        self,
        reference: OperationalReferenceMetrics,
        constraint: YieldConstraint,
    ) -> bool:
        return (
            self.crop_state_success_probability(reference, constraint)
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
    reference_metrics: OperationalReferenceMetrics
    constraint: YieldConstraint
    objective_mode: str
    feasible: bool
    reason: str

    @property
    def reference_yields_t_ha(self) -> dict[int, float]:
        """Backward-compatible access to the member-wise yield reference."""
        return self.reference_metrics.member_yield_t_ha

    @property
    def reference_biomass_t_ha(self) -> dict[int, float]:
        return self.reference_metrics.member_biomass_t_ha

    @property
    def reference_canopy_cover(self) -> dict[int, float]:
        return self.reference_metrics.member_canopy_cover

    @property
    def recommended_total_mm(self) -> float:
        return float(sum(q for _, q in self.recommended_events))

    @property
    def success_probability(self) -> float:
        if self.objective_mode == "crop_state":
            return self.optimized_evaluation.crop_state_success_probability(
                self.reference_metrics,
                self.constraint,
            )
        return self.optimized_evaluation.success_probability(
            self.reference_metrics.member_yield_t_ha,
            self.constraint,
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(d),
                    "day_offset": (d - self.today).days,
                    "amount_mm": float(q),
                    "committed": d in self.committed_dates,
                }
                for d, q in self.recommended_events
            ]
        )

    def to_irrigation_log(self) -> list[dict[str, Any]]:
        return [{"date": d.isoformat(), "amount_mm": float(q)} for d, q in self.recommended_events]


# ---------------------------------------------------------------------------
# AquaCrop evaluation engine
# ---------------------------------------------------------------------------


def _last_completed_crop_row(result: AquaCropResult) -> pd.Series | None:
    """Return the last completed AquaCrop crop-growth row.

    AquaCrop output tables can contain a terminal/preallocated row.  Prefer the
    largest non-negative ``time_step_counter`` when it is available rather than
    blindly using ``iloc[-1]``.
    """
    cg = result.crop_growth
    if cg is None or len(cg) == 0:
        return None

    if "time_step_counter" in cg.columns:
        counter = pd.to_numeric(cg["time_step_counter"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(counter) & (counter >= 0)
        if np.any(valid):
            valid_idx = np.flatnonzero(valid)
            idx = valid_idx[int(np.argmax(counter[valid_idx]))]
            return cg.iloc[int(idx)]

    # Retain the historical defensive behaviour for older result tables.
    return cg.iloc[-2] if len(cg) > 1 else cg.iloc[-1]


def _extract_dry_yield(result: AquaCropResult) -> float:
    """Return final yield when available, otherwise the current partial yield."""
    final = result.final_results
    if len(final) and "Dry yield (tonne/ha)" in final.columns:
        value = float(final.iloc[-1]["Dry yield (tonne/ha)"])
        if np.isfinite(value):
            return value

    row = _last_completed_crop_row(result)
    if row is None:
        return 0.0

    value = float(row.get("DryYield", np.nan))
    if np.isfinite(value):
        return value

    # Same partial-season fallback historically used by this module.
    biomass = float(row.get("biomass", 0.0))
    hi = float(row.get("harvest_index", 0.0))
    return biomass * hi / 1000.0


def _extract_end_horizon_crop_state(
    result: AquaCropResult,
) -> tuple[float, float, float]:
    """Return dry yield, biomass and canopy at the end of the real horizon.

    AquaCrop raw biomass is expressed in g/m2 in this project; dividing by 100
    gives t/ha, consistently with the plotting/summary utilities.
    """
    dry_yield = _extract_dry_yield(result)
    row = _last_completed_crop_row(result)
    if row is None:
        return dry_yield, 0.0, 0.0

    biomass = float(row.get("biomass", 0.0))
    canopy = float(row.get("canopy_cover", 0.0))

    biomass_t_ha = biomass / 100.0 if np.isfinite(biomass) else 0.0
    canopy_cover = canopy if np.isfinite(canopy) else 0.0
    return dry_yield, biomass_t_ha, canopy_cover


def _first_stress_date(
    result: AquaCropResult,
    *,
    crop,
    aquacrop_soil_profile: pd.DataFrame,
    trigger_mode: str,
    trigger_process: str,
    trigger_margin_fraction: float,
    threshold_ks: float,
    not_before: date | None,
) -> tuple[date | None, str | None]:
    """Return the first irrigation-trigger date and active stress process.

    ``dynamic_depletion`` compares root-zone depletion against AquaCrop's own
    process-specific, ET0-adjusted stress-onset threshold.  The old Tr/TrPot
    threshold is retained as ``ks`` mode for comparison/backwards compatibility.
    """
    if result.weather_daily is None:
        return None, None

    stress = result.daily_stress
    if stress.empty:
        return None, None

    dates = pd.to_datetime(result.weather_daily["Date"].iloc[: len(stress)]).dt.date.tolist()

    if trigger_mode == "ks":
        ks = stress["ks"].to_numpy(dtype=float)
        for d, value in zip(dates, ks, strict=True):
            if not_before is not None and d < not_before:
                continue
            if np.isfinite(value) and value < float(threshold_ks):
                return d, "ks"
        return None, None

    if trigger_mode != "dynamic_depletion":
        raise ValueError("stress_trigger_mode must be 'dynamic_depletion' or 'ks'.")

    return first_water_stress_trigger_date(
        dates=dates,
        crop=crop,
        soil_profile=aquacrop_soil_profile,
        wr_mm=stress["wr_mm"].to_numpy(dtype=float),
        z_root_m=stress["z_root_m"].to_numpy(dtype=float),
        et0=stress["et0_mm"].to_numpy(dtype=float),
        gdd_cum=stress["gdd_cum"].to_numpy(dtype=float),
        dap=stress["dap"].to_numpy(dtype=float),
        not_before=not_before,
        process=trigger_process,
        margin_fraction=trigger_margin_fraction,
    )


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
    initial_soil_water_profile: xr.Dataset | None = None,
) -> None:
    trigger_crop = parcel_to_crop(parcel)
    trigger_soil = _soil_for_crop(soil_profile, trigger_crop)

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
            "initial_soil_water_profile": initial_soil_water_profile,
            "trigger_crop": trigger_crop,
            "trigger_soil_profile": trigger_soil.profile.copy(),
        }
    )


def _yield_worker(
    job: tuple[
        int,
        IrrigationCandidate,
        str,
        str,
        float,
        float,
        date | None,
    ],
) -> tuple[int, float, float, float, date | None, str | None]:
    (
        member,
        candidate,
        trigger_mode,
        trigger_process,
        trigger_margin_fraction,
        stress_threshold_ks,
        stress_not_before,
    ) = job

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
        initial_soil_water_profile=ctx["initial_soil_water_profile"],  # type: ignore[arg-type]
    )

    dry_yield, biomass_t_ha, canopy_cover = _extract_end_horizon_crop_state(result)
    first_stress, first_process = _first_stress_date(
        result,
        crop=ctx["trigger_crop"],
        aquacrop_soil_profile=ctx["trigger_soil_profile"],  # type: ignore[arg-type]
        trigger_mode=trigger_mode,
        trigger_process=trigger_process,
        trigger_margin_fraction=trigger_margin_fraction,
        threshold_ks=stress_threshold_ks,
        not_before=stress_not_before,
    )

    return (
        int(member),
        float(dry_yield),
        float(biomass_t_ha),
        float(canopy_cover),
        first_stress,
        first_process,
    )


class _YieldEvaluator:
    """Cache crop-state/yield evaluations and optionally reuse a process pool."""

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
        initial_soil_water_profile: xr.Dataset | None = None,
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
        self.initial_soil_water_profile = initial_soil_water_profile
        self.n_workers = max(1, int(n_workers))
        self.cache: dict[tuple, YieldScheduleEvaluation] = {}
        self.pool: mp.pool.Pool | None = None
        self.branching: AquaCropYieldBranchingEvaluator | None = None

        # Needed by the non-branching path for AquaCrop's dynamic trigger.
        self.trigger_crop = parcel_to_crop(parcel)
        trigger_soil = _soil_for_crop(soil_profile, self.trigger_crop)
        self.trigger_soil_profile = trigger_soil.profile.copy()

        # Operational optimization has a natural deterministic/ensemble branch
        # boundary. Reuse the private AquaCrop state stepper there instead of
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
                    initial_soil_water_profile=initial_soil_water_profile,
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
                    self.initial_soil_water_profile,
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
        stress_trigger_mode: str,
        stress_trigger_process: str,
        stress_trigger_margin_fraction: float,
        stress_threshold_ks: float,
        stress_not_before: date | None,
    ) -> tuple:
        events = tuple((int(d), round(float(q), 4)) for d, q in candidate.events if q > 1e-9)
        return (
            events,
            stress_trigger_mode,
            stress_trigger_process,
            round(float(stress_trigger_margin_fraction), 4),
            round(float(stress_threshold_ks), 4),
            stress_not_before,
        )

    def evaluate(
        self,
        candidate: IrrigationCandidate,
        *,
        stress_trigger_mode: str = "dynamic_depletion",
        stress_trigger_process: str = "auto",
        stress_trigger_margin_fraction: float = 0.0,
        stress_threshold_ks: float = 0.98,
        stress_not_before: date | None = None,
    ) -> YieldScheduleEvaluation:
        key = self._key(
            candidate,
            stress_trigger_mode,
            stress_trigger_process,
            stress_trigger_margin_fraction,
            stress_threshold_ks,
            stress_not_before,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.branching is not None:
            branch_rows = self.branching.evaluate(
                candidate,
                stress_trigger_mode=stress_trigger_mode,
                stress_trigger_process=stress_trigger_process,
                stress_trigger_margin_fraction=stress_trigger_margin_fraction,
                stress_threshold_ks=stress_threshold_ks,
                stress_not_before=stress_not_before,
            )
            outputs = [
                (
                    member,
                    row.dry_yield_t_ha,
                    row.biomass_t_ha,
                    row.canopy_cover,
                    row.first_stress_date,
                    row.first_stress_process,
                )
                for member, row in sorted(branch_rows.items())
            ]

        elif self.member_forcings:
            jobs = [
                (
                    int(member),
                    candidate,
                    stress_trigger_mode,
                    stress_trigger_process,
                    float(stress_trigger_margin_fraction),
                    float(stress_threshold_ks),
                    stress_not_before,
                )
                for member in sorted(self.member_forcings)
            ]

            if self.pool is not None:
                outputs = self.pool.map(_yield_worker, jobs)
            else:
                outputs = []
                for member, *_ in jobs:
                    forcing = self.base_forcing.concat(self.member_forcings[member])
                    irrigation = _build_candidate_irrigation(
                        self.parcel,
                        candidate,
                        self.anchor,
                    )
                    result = run_aquacrop(
                        forcing=forcing,
                        parcel=self.parcel,
                        terrain=self.terrain,
                        soil_profile=self.soil_profile,
                        sim_start=self.sim_start,
                        sim_end=self.sim_end,
                        irrigation_management=irrigation,
                        initial_soil_water_profile=self.initial_soil_water_profile,
                    )
                    dry_yield, biomass_t_ha, canopy_cover = _extract_end_horizon_crop_state(result)
                    first_stress, first_process = _first_stress_date(
                        result,
                        crop=self.trigger_crop,
                        aquacrop_soil_profile=self.trigger_soil_profile,
                        trigger_mode=stress_trigger_mode,
                        trigger_process=stress_trigger_process,
                        trigger_margin_fraction=stress_trigger_margin_fraction,
                        threshold_ks=stress_threshold_ks,
                        not_before=stress_not_before,
                    )
                    outputs.append(
                        (
                            member,
                            dry_yield,
                            biomass_t_ha,
                            canopy_cover,
                            first_stress,
                            first_process,
                        )
                    )

        else:
            irrigation = _build_candidate_irrigation(
                self.parcel,
                candidate,
                self.anchor,
            )
            result = run_aquacrop(
                forcing=self.base_forcing,
                parcel=self.parcel,
                terrain=self.terrain,
                soil_profile=self.soil_profile,
                sim_start=self.sim_start,
                sim_end=self.sim_end,
                irrigation_management=irrigation,
                initial_soil_water_profile=self.initial_soil_water_profile,
            )
            dry_yield, biomass_t_ha, canopy_cover = _extract_end_horizon_crop_state(result)
            first_stress, first_process = _first_stress_date(
                result,
                crop=self.trigger_crop,
                aquacrop_soil_profile=self.trigger_soil_profile,
                trigger_mode=stress_trigger_mode,
                trigger_process=stress_trigger_process,
                trigger_margin_fraction=stress_trigger_margin_fraction,
                threshold_ks=stress_threshold_ks,
                not_before=stress_not_before,
            )
            outputs = [
                (
                    0,
                    dry_yield,
                    biomass_t_ha,
                    canopy_cover,
                    first_stress,
                    first_process,
                )
            ]

        evaluation = YieldScheduleEvaluation(
            candidate=candidate,
            member_yield_t_ha={m: float(y) for m, y, _, _, _, _ in outputs},
            member_biomass_t_ha={m: float(b) for m, _, b, _, _, _ in outputs},
            member_canopy_cover={m: float(c) for m, _, _, c, _, _ in outputs},
            first_stress_date={m: d for m, _, _, _, d, _ in outputs},
            first_stress_process={m: p for m, _, _, _, _, p in outputs},
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
    """Member-wise attainable-yield benchmark from several wet schedules."""
    rows = list(evaluations)
    if not rows:
        raise ValueError("At least one evaluation is required for the yield reference.")
    members = rows[0].member_yield_t_ha
    return {member: max(float(row.member_yield_t_ha[member]) for row in rows) for member in members}


def _reference_metrics(
    evaluations: Iterable[YieldScheduleEvaluation],
) -> OperationalReferenceMetrics:
    """Build member-wise attainable crop-state references.

    A maximum-irrigation schedule can itself be suboptimal because of
    drainage/aeration effects.  Taking each member's best state across a small
    family of wet schedules plus no-irrigation is therefore more robust than
    assuming that the wettest schedule is the reference.
    """
    rows = list(evaluations)
    if not rows:
        raise ValueError("At least one evaluation is required for the reference.")

    members = set(rows[0].member_yield_t_ha)
    for row in rows[1:]:
        if set(row.member_yield_t_ha) != members:
            raise ValueError("Reference evaluations do not contain the same members.")

    return OperationalReferenceMetrics(
        member_yield_t_ha={
            member: max(float(row.member_yield_t_ha[member]) for row in rows) for member in members
        },
        member_biomass_t_ha={
            member: max(float(row.member_biomass_t_ha[member]) for row in rows)
            for member in members
        },
        member_canopy_cover={
            member: max(float(row.member_canopy_cover[member]) for row in rows)
            for member in members
        },
    )


def _operational_success_probability(
    evaluation: YieldScheduleEvaluation,
    reference: OperationalReferenceMetrics,
    constraint: YieldConstraint,
    objective_mode: str,
) -> float:
    if objective_mode == "crop_state":
        return evaluation.crop_state_success_probability(reference, constraint)
    if objective_mode == "yield":
        return evaluation.success_probability(reference.member_yield_t_ha, constraint)
    raise ValueError(f"Unknown objective_mode={objective_mode!r}")


def _operational_satisfies(
    evaluation: YieldScheduleEvaluation,
    reference: OperationalReferenceMetrics,
    constraint: YieldConstraint,
    objective_mode: str,
) -> bool:
    return (
        _operational_success_probability(
            evaluation,
            reference,
            constraint,
            objective_mode,
        )
        >= constraint.min_member_probability
    )


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
    reference: OperationalReferenceMetrics,
    objective_mode: str,
    constraint: YieldConstraint,
    min_dose: float,
    max_dose: float,
    dose_resolution_mm: float,
    committed_dates: set[date],
    stress_trigger_mode: str,
    stress_trigger_process: str,
    stress_trigger_margin_fraction: float,
    stress_trigger_ks: float,
    max_passes: int = 2,
) -> tuple[list[tuple[date, float]], YieldScheduleEvaluation]:
    """Coordinate-wise monotone dose reduction under the active constraint."""

    def evaluate(candidate: IrrigationCandidate) -> YieldScheduleEvaluation:
        return evaluator.evaluate(
            candidate,
            stress_trigger_mode=stress_trigger_mode,
            stress_trigger_process=stress_trigger_process,
            stress_trigger_margin_fraction=stress_trigger_margin_fraction,
            stress_threshold_ks=stress_trigger_ks,
            stress_not_before=None,
        )

    current = [(d, float(q)) for d, q in events]
    current_eval = evaluate(_candidate_from_dates(current, anchor=anchor, label="dose-search"))

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
                candidate = _candidate_from_dates(
                    trial,
                    anchor=anchor,
                    label="dose-search",
                )
                evaluation = evaluate(candidate)

                if _operational_satisfies(
                    evaluation,
                    reference,
                    constraint,
                    objective_mode,
                ):
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
    final_candidate = _candidate_from_dates(
        current,
        anchor=anchor,
        label="optimized",
    )
    final_eval = evaluate(final_candidate)
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
    initial_soil_water_profile: xr.Dataset | None = None,
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
        initial_soil_water_profile=initial_soil_water_profile,
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
    previous_plan: list[dict[str, Any]] | list[tuple[date, float]] | None = None,
    commitment_days: int = 2,
    constraint: YieldConstraint = YieldConstraint(),
    stress_trigger_mode: str = "dynamic_depletion",
    stress_trigger_process: str = "auto",
    stress_trigger_margin_fraction: float = 0.0,
    stress_trigger_ks: float = 0.98,
    trigger_quantile: float = 0.25,
    irrigation_lead_days: int = 1,
    dose_resolution_mm: float = 2.0,
    max_events: int = 99,
    n_workers: int = 1,
    require_harvest_horizon: bool = True,
    initial_soil_water_profile: xr.Dataset | None = None,
) -> OperationalOptimizationResult:
    """Find a sparse forecast irrigation plan while preserving crop performance.

    Water is always minimized.  The agronomic acceptance constraint depends on
    how far the supplied weather reaches:

    * if the common horizon reaches expected harvest, preserve final dry yield;
    * otherwise preserve end-horizon biomass + canopy cover relative to the
      member-wise wet reference, and also partial dry yield once it is material.

    Irrigation dates are proposed from AquaCrop's dynamic root-zone depletion
    thresholds by default.  ``stress_trigger_process='auto'`` changes the active
    process with phenology (canopy expansion, flowering/pollination, stomatal
    limitation, senescence).  The trigger proposes dates only; it does not
    determine candidate feasibility.

    ``stress_trigger_mode='ks'`` retains the legacy Tr/TrPot trigger for
    sensitivity testing.  In that mode ``stress_trigger_ks`` is used; otherwise
    it is ignored.

    ``arome_forcing`` is the deterministic short-range forcing segment.  The
    name is retained for API compatibility and may contain either AROME (live)
    or HRES (historical backtests).
    """
    if not future_member_forcings:
        raise ValueError("future_member_forcings is empty.")
    if not 0 <= trigger_quantile <= 1:
        raise ValueError("trigger_quantile must be in [0, 1].")
    if max_events < 1:
        raise ValueError("max_events must be >= 1.")
    if stress_trigger_mode not in {"dynamic_depletion", "ks"}:
        raise ValueError("stress_trigger_mode must be 'dynamic_depletion' or 'ks'.")
    if stress_trigger_process not in {
        "auto",
        "expansion",
        "stomatal",
        "senescence",
        "pollination",
    }:
        raise ValueError(
            "stress_trigger_process must be one of: auto, expansion, "
            "stomatal, senescence, pollination."
        )
    if not 0 <= stress_trigger_margin_fraction < 1:
        raise ValueError("stress_trigger_margin_fraction must be in [0, 1).")

    base_forcing = historical_forcing.concat(arome_forcing)

    # Maximum date supported by every forecast member.
    common_end = _end_date_from_forcings(
        base_forcing,
        future_member_forcings,
    )

    # User-requested horizon, constrained by available weather.
    sim_end = common_end if sim_end is None else min(sim_end, common_end)
    expected_harvest = _expected_harvest(parcel)

    if require_harvest_horizon and expected_harvest is not None and sim_end < expected_harvest:
        raise ValueError(
            f"Yield optimization needs weather through expected harvest "
            f"{expected_harvest}, but the common forecast horizon ends "
            f"{sim_end}. Extend IFS members with seasonal scenarios first "
            "or set require_harvest_horizon=False to use the partial-horizon "
            "crop-state constraint."
        )

    # Whether crop-state or final-yield preservation is scientifically meaningful
    # is determined by the actual available horizon, not by the flag itself.
    objective_mode = (
        "crop_state" if expected_harvest is None or sim_end < expected_harvest else "yield"
    )

    # There is no reason to simulate beyond the configured harvest date.
    if expected_harvest is not None:
        sim_end = min(sim_end, expected_harvest)

    # Truncate ensemble scenarios to the useful horizon. This also keeps the
    # private AquaCrop branching evaluator active because member_end == sim_end.
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

    trigger_kwargs = {
        "stress_trigger_mode": stress_trigger_mode,
        "stress_trigger_process": stress_trigger_process,
        "stress_trigger_margin_fraction": stress_trigger_margin_fraction,
        "stress_threshold_ks": stress_trigger_ks,
    }

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
        initial_soil_water_profile=initial_soil_water_profile,
    ) as evaluator:
        no_candidate = IrrigationCandidate(
            events=[],
            label="No future irrigation",
        )
        no_eval = evaluator.evaluate(
            no_candidate,
            **trigger_kwargs,
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
            wet_events = [
                (event_date, dose * fraction) for event_date, dose in max_events_reference
            ]
            wet_eval = evaluator.evaluate(
                _candidate_from_dates(
                    wet_events,
                    anchor=anchor,
                    label=f"Wet reference {fraction:.0%}",
                ),
                **trigger_kwargs,
            )
            wet_reference_evaluations.append(wet_eval)
            reference_evaluations.append(wet_eval)

        reference_metrics = _reference_metrics(reference_evaluations)

        # Keep one real schedule for diagnostics.  Feasibility itself uses the
        # member-wise reference_metrics above, not this single schedule.
        if objective_mode == "yield":
            reference_eval = max(
                wet_reference_evaluations,
                key=lambda row: row.mean_yield_t_ha,
            )
        else:
            reference_eval = max(
                wet_reference_evaluations,
                key=lambda row: (
                    row.mean_biomass_t_ha,
                    row.mean_canopy_cover,
                    row.mean_yield_t_ha,
                ),
            )

        if _operational_satisfies(
            no_eval,
            reference_metrics,
            constraint,
            objective_mode,
        ):
            criterion = (
                "crop-state constraint" if objective_mode == "crop_state" else "yield constraint"
            )
            return OperationalOptimizationResult(
                today=today,
                recommended_events=[],
                committed_dates=set(),
                reference_evaluation=reference_eval,
                no_irrigation_evaluation=no_eval,
                optimized_evaluation=no_eval,
                reference_metrics=reference_metrics,
                constraint=constraint,
                objective_mode=objective_mode,
                feasible=True,
                reason=f"No future irrigation is needed to satisfy the {criterion}.",
            )

        commitment_limit = today + timedelta(days=max(0, commitment_days))
        committed_dates: set[date] = set()
        current_events: list[tuple[date, float]] = []

        for event in previous_plan or []:
            if isinstance(event, dict):
                event_date = pd.Timestamp(event["date"]).date()
            else:
                event_date = pd.Timestamp(event[0]).date()

            if today < event_date <= commitment_limit and event_date <= sim_end:
                committed_dates.add(event_date)
                # Keep the committed date, but start high and let the dose
                # minimizer reduce it under the active agronomic constraint.
                current_events.append((event_date, max_dose))

        # Remove duplicates while retaining chronological date order.
        current_events = sorted({d: q for d, q in current_events}.items())

        search_not_before = today + timedelta(days=1)
        evaluation = evaluator.evaluate(
            _candidate_from_dates(
                current_events,
                anchor=anchor,
                label="event search",
            ),
            **trigger_kwargs,
            stress_not_before=search_not_before,
        )

        feasible = True
        criterion = (
            "crop-state constraint" if objective_mode == "crop_state" else "yield constraint"
        )
        reason = f"{criterion.capitalize()} reached with event-triggered schedule."

        while not _operational_satisfies(
            evaluation,
            reference_metrics,
            constraint,
            objective_mode,
        ):
            if len(current_events) >= max_events:
                feasible = False
                reason = f"Reached max_events={max_events} before satisfying the {criterion}."
                break

            stress_dates = [d for d in evaluation.first_stress_date.values() if d is not None]
            if not stress_dates:
                feasible = False
                reason = (
                    f"The schedule misses the {criterion}, but no "
                    "AquaCrop water-stress trigger was found inside the "
                    "remaining horizon."
                )
                break

            ordinals = np.asarray(
                [d.toordinal() for d in stress_dates],
                dtype=float,
            )
            trigger = date.fromordinal(int(round(float(np.quantile(ordinals, trigger_quantile)))))
            desired = trigger - timedelta(days=max(0, irrigation_lead_days))

            last_actual = _last_recorded_irrigation(parcel, today)
            last_planned = max(
                (d for d, _ in current_events),
                default=None,
            )
            last_event = max(
                [d for d in (last_actual, last_planned) if d is not None],
                default=None,
            )

            lower = today + timedelta(days=1)
            if last_event is not None:
                lower = max(
                    lower,
                    last_event + timedelta(days=min_interval),
                )

            event_date = _latest_available_at_or_before(
                desired,
                lower,
                weekdays,
            )
            if event_date is None:
                event_date = _next_available(lower, weekdays)

            if event_date > sim_end:
                feasible = False
                reason = "No feasible irrigation date remains inside the optimization horizon."
                break

            if any(d == event_date for d, _ in current_events):
                # Advance rather than repeatedly reacting to the same threshold.
                event_date = _next_available(
                    event_date + timedelta(days=min_interval),
                    weekdays,
                )
                if event_date > sim_end:
                    feasible = False
                    reason = "No additional feasible irrigation date remains."
                    break

            current_events.append((event_date, max_dose))
            current_events.sort()
            search_not_before = event_date + timedelta(days=1)

            evaluation = evaluator.evaluate(
                _candidate_from_dates(
                    current_events,
                    anchor=anchor,
                    label="event search",
                ),
                **trigger_kwargs,
                stress_not_before=search_not_before,
            )

        if feasible:
            current_events, optimized_eval = _minimize_event_doses(
                evaluator,
                current_events,
                anchor=anchor,
                reference=reference_metrics,
                objective_mode=objective_mode,
                constraint=constraint,
                min_dose=min_dose,
                max_dose=max_dose,
                dose_resolution_mm=dose_resolution_mm,
                committed_dates=committed_dates,
                stress_trigger_mode=stress_trigger_mode,
                stress_trigger_process=stress_trigger_process,
                stress_trigger_margin_fraction=stress_trigger_margin_fraction,
                stress_trigger_ks=stress_trigger_ks,
            )

            feasible = _operational_satisfies(
                optimized_eval,
                reference_metrics,
                constraint,
                objective_mode,
            )
            if not feasible:
                reason = (
                    "Dose reduction unexpectedly crossed the active agronomic "
                    "constraint; inspect candidate-response monotonicity."
                )
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
        reference_metrics=reference_metrics,
        constraint=constraint,
        objective_mode=objective_mode,
        feasible=feasible,
        reason=reason,
    )
