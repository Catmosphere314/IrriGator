"""Historical AquaCrop benchmark plots.

This module compares one current AquaCrop run per parcel with an ensemble of
historical runs (for example, the previous 20 weather years). Historical
trajectories are aligned by crop day / DAP, then summarized with the 25th
percentile, median, and 75th percentile.

The module is designed for the output structure returned by the IrriGator
``run_aquacrop`` adapter:

- ``crop_growth``
- ``water_flux``
- ``daily_stress``
- an AquaCrop ``Soil.profile`` DataFrame

Important experimental design
-----------------------------
For a climatic benchmark, keep soil, crop parameters, planting month/day,
initial-water method, and irrigation-management rule fixed across historical
runs. Change only the meteorological year. If historical configurations are
allowed to vary, the envelope represents historical *systems* rather than
weather variability alone.

Historical runs must retain their real ``sim_start`` year. The plotting code
aligns them to the current season by crop day, so there is no need to relabel a
2025 run with a 2026 simulation start date.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


DEFAULT_QUANTILES = (0.25, 0.50, 0.75)
DEFAULT_STRESS_THRESHOLD = 0.99

# Metrics included in the current-versus-history table.
# ``higher_is_better`` is descriptive only; the reported percentile is always
# the ordinary lower-tail percentile P(historical <= current).
BENCHMARK_METRICS: dict[str, dict[str, Any]] = {
    "final_gdd_cum": {
        "label": "Cumulative GDD",
        "unit": "°C·day",
        "higher_is_better": None,
    },
    "maximum_root_depth_m": {
        "label": "Maximum effective root depth",
        "unit": "m",
        "higher_is_better": None,
    },
    "maximum_canopy_cover": {
        "label": "Maximum canopy cover",
        "unit": "fraction",
        "higher_is_better": True,
    },
    "current_biomass_t_ha": {
        "label": "Biomass at comparison date",
        "unit": "t/ha",
        "higher_is_better": True,
    },
    "current_dry_yield_t_ha": {
        "label": "Dry yield at comparison date",
        "unit": "t/ha",
        "higher_is_better": True,
    },
    "precipitation_mm": {
        "label": "Cumulative precipitation",
        "unit": "mm",
        "higher_is_better": None,
    },
    "irrigation_mm": {
        "label": "Cumulative irrigation",
        "unit": "mm",
        "higher_is_better": False,
    },
    "transpiration_mm": {
        "label": "Cumulative transpiration",
        "unit": "mm",
        "higher_is_better": True,
    },
    "soil_evaporation_mm": {
        "label": "Cumulative soil evaporation",
        "unit": "mm",
        "higher_is_better": False,
    },
    "runoff_mm": {
        "label": "Cumulative runoff",
        "unit": "mm",
        "higher_is_better": False,
    },
    "deep_percolation_mm": {
        "label": "Cumulative deep percolation",
        "unit": "mm",
        "higher_is_better": False,
    },
    "transpiration_stress_days": {
        "label": "Transpiration-stress days",
        "unit": "days",
        "higher_is_better": False,
    },
    "minimum_tr_over_trpot": {
        "label": "Minimum Tr/TrPot",
        "unit": "fraction",
        "higher_is_better": True,
    },
    "maximum_root_zone_depletion_fraction": {
        "label": "Maximum root-zone depletion",
        "unit": "Dr/TAW",
        "higher_is_better": False,
    },
    "minimum_stomatal_margin_mm": {
        "label": "Minimum margin to stomatal stress",
        "unit": "mm",
        "higher_is_better": True,
    },
    "stomatal_threshold_crossing_days": {
        "label": "Days beyond stomatal-stress onset",
        "unit": "days",
        "higher_is_better": False,
    },
}


@dataclass(slots=True)
class HistoricalBenchmarkResult:
    """Prepared data and summary tables returned with the figure."""

    current: dict[str, pd.DataFrame]
    historical: dict[str, pd.DataFrame]
    daily_quantiles: pd.DataFrame
    run_summaries: pd.DataFrame
    metric_benchmark: pd.DataFrame
    comparison_day_by_parcel: dict[str, int]


def make_run_config(
    model_run: Any,
    *,
    parcel: str,
    soil: pd.DataFrame,
    sim_start: str | pd.Timestamp,
    year: int | None = None,
    season: int = 0,
    scenario: str | None = None,
    phenology: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one configuration entry from an IrriGator AquaCrop run object.

    Parameters
    ----------
    model_run:
        Object exposing ``crop_growth``, ``water_flux`` and ``daily_stress``.
    parcel:
        Parcel name used for grouping and color assignment.
    soil:
        AquaCrop ``Soil.profile`` DataFrame.
    sim_start:
        The run's *real* simulation start date. For a 2012 historical run,
        provide a 2012 date rather than shifting it to the current year.
    year:
        Weather/simulation year. Inferred from ``sim_start`` when omitted.
    scenario:
        Optional free-text label retained in metadata.
    """
    start = pd.Timestamp(sim_start)
    return {
        "parcel": str(parcel),
        "scenario": scenario or ("current" if year == start.year else "historical"),
        "year": int(year if year is not None else start.year),
        "crop_growth": model_run.crop_growth,
        "water_flux": model_run.water_flux,
        "daily_stress": model_run.daily_stress,
        "soil": soil,
        "sim_start": start,
        "season": int(season),
        "phenology": dict(phenology or {}),
    }


def _drop_csv_index(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[c for c in frame.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    ).copy()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _integrated_storage(
    soil: pd.DataFrame,
    depths_m: np.ndarray,
    theta_column: str,
) -> np.ndarray:
    """Integrate a volumetric water-content threshold to each root depth."""
    profile = _drop_csv_index(soil).sort_values("dzsum")
    _require_columns(profile, ["dz", "dzsum", theta_column], "soil profile")

    top = (profile["dzsum"] - profile["dz"]).to_numpy(dtype=float)
    bottom = profile["dzsum"].to_numpy(dtype=float)
    theta = profile[theta_column].to_numpy(dtype=float)
    profile_depth = float(bottom[-1])

    result = np.full(len(depths_m), np.nan, dtype=float)
    for index, depth in enumerate(np.asarray(depths_m, dtype=float)):
        if not np.isfinite(depth) or depth <= 0:
            continue
        clipped_depth = np.clip(depth, 0.0, profile_depth)
        overlap = np.clip(
            np.minimum(bottom, clipped_depth) - top,
            0.0,
            bottom - top,
        )
        result[index] = np.sum(theta * overlap * 1000.0)
    return result


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=np.isfinite(denominator) & (np.abs(denominator) > 1e-12),
    )


def _thresholds_are_similar(
    frames_by_parcel: Mapping[str, pd.DataFrame],
    columns: Sequence[str] = ("p_exp_start", "p_sto_start", "p_sto_full"),
    *,
    tolerance: float = 0.02,
) -> bool:
    """Return whether parcel threshold curves are effectively interchangeable.

    Curves are compared on common calendar dates while the crop is active.
    ``tolerance`` is expressed as a fraction of TAW; 0.02 means two
    percentage points of total available root-zone water.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    if len(frames_by_parcel) <= 1:
        return True

    for column in columns:
        series = []
        for parcel, frame in frames_by_parcel.items():
            data = frame.loc[
                frame["active"] & frame[column].notna(),
                ["date", column],
            ].drop_duplicates("date")
            series.append(data.set_index("date")[column].rename(parcel))

        aligned = pd.concat(series, axis=1, join="inner").dropna()
        if aligned.empty:
            return False

        spread = aligned.max(axis=1) - aligned.min(axis=1)
        if float(spread.max()) > tolerance:
            return False

    return True


def _plot_crop_water_thresholds(
    axis: plt.Axes,
    data: pd.DataFrame,
    *,
    color: Any = "0.30",
    label_prefix: str = "",
    show_stomatal_zone: bool = True,
) -> None:
    """Plot the crop-water thresholds that are most useful for management.

    Canopy-expansion onset is only shown while the no-stress canopy is still
    expanding. Stomatal onset is shown through the active season. The region
    from stomatal onset to maximum stomatal stress is lightly shaded instead
    of adding another prominent line.
    """
    active = data.loc[data["active"]].copy()
    if active.empty:
        return

    # Canopy-expansion stress is only relevant before canopy closure.
    canopy_reference = active["canopy_cover_ns"]
    canopy_max = float(canopy_reference.max()) if canopy_reference.notna().any() else np.nan
    if np.isfinite(canopy_max) and canopy_max > 0:
        expansion = active.loc[
            active["p_exp_start"].notna() & active["canopy_cover_ns"].lt(0.98 * canopy_max)
        ]
        # if not expansion.empty:
            # axis.plot(
            #     expansion["date"],
            #     expansion["p_exp_start"],
            #     color=color,
            #     linestyle=":",
            #     linewidth=1.05,
            #     alpha=0.72,
            #     label=f"{label_prefix}canopy-expansion stress onset",
            # )

    stomatal = active.loc[active["p_sto_start"].notna() & active["p_sto_full"].notna()]
    if stomatal.empty:
        return

    axis.plot(
        stomatal["date"],
        stomatal["p_sto_start"],
        color=color,
        linestyle="-.",
        linewidth=1.45,
        alpha=0.95,
        label=f"{label_prefix}stomatal-stress onset",
    )

    # if show_stomatal_zone:
    #     axis.fill_between(
    #         stomatal["date"],
    #         stomatal["p_sto_start"],
    #         stomatal["p_sto_full"],
    #         color=color,
    #         alpha=0.045,
    #         linewidth=0,
    #         label=f"{label_prefix}increasing stomatal limitation",
    #     )


def prepare_aquacrop_run(
    configuration: Mapping[str, Any],
) -> pd.DataFrame:
    """Convert one raw AquaCrop output set into a clean daily table.

    Unlike the earlier seasonal plotting helper, this function retains daily
    water fluxes after crop termination. Crop-state variables are set to NaN
    outside the active season. This allows cumulative precipitation and water
    fluxes to be compared over a common crop-age horizon even when historical
    harvest dates differ.
    """
    required_keys = {
        "parcel",
        "crop_growth",
        "water_flux",
        "daily_stress",
        "soil",
        "sim_start",
    }
    missing_keys = required_keys - set(configuration)
    if missing_keys:
        raise ValueError(f"Run configuration is missing keys: {sorted(missing_keys)}")

    crop = _drop_csv_index(configuration["crop_growth"])
    flux = _drop_csv_index(configuration["water_flux"])
    stress = _drop_csv_index(configuration["daily_stress"])
    soil = _drop_csv_index(configuration["soil"])

    _require_columns(
        crop,
        [
            "time_step_counter",
            "season_counter",
            "dap",
            "gdd_cum",
            "z_root",
            "canopy_cover",
            "canopy_cover_ns",
            "biomass",
            "biomass_ns",
            "DryYield",
            "YieldPot",
        ],
        "crop_growth",
    )
    _require_columns(
        flux,
        [
            "Wr",
            "IrrDay",
            "Runoff",
            "DeepPerc",
            "Es",
            "Tr",
            "TrPot",
        ],
        "water_flux",
    )
    _require_columns(
        stress,
        [
            "precip_mm",
            "et0_mm",
            "p_exp_start",
            "p_sto_start",
            "p_sen_start",
            "p_pol_start",
            "p_exp_full",
            "p_sto_full",
            "p_sen_full",
            "p_pol_full",
        ],
        "daily_stress",
    )
    _require_columns(soil, ["dz", "dzsum", "th_fc", "th_wp"], "soil profile")

    if not (len(crop) == len(flux) == len(stress)):
        raise ValueError("crop_growth, water_flux and daily_stress must have equal row counts.")

    season = int(configuration.get("season", 0))
    active = crop["season_counter"].eq(season).to_numpy()
    if not active.any():
        raise ValueError(
            f"No rows found for season={season} in parcel {configuration['parcel']!r}."
        )

    dates = pd.Timestamp(configuration["sim_start"]) + pd.to_timedelta(
        crop["time_step_counter"].to_numpy(dtype=float), unit="D"
    )
    planting_position = int(np.flatnonzero(active)[0])
    planting_date = pd.Timestamp(
        dates.iloc[planting_position] if hasattr(dates, "iloc") else dates[planting_position]
    )
    crop_day = (pd.DatetimeIndex(dates) - planting_date).days + 1
    after_planting = crop_day >= 1

    positive_roots = crop.loc[active & crop["z_root"].gt(0), "z_root"]
    crop_zmin = configuration.get("crop_zmin")
    if crop_zmin is None:
        if positive_roots.empty:
            raise ValueError("Cannot infer a positive minimum root depth.")
        crop_zmin = float(positive_roots.min())

    root_depth = crop["z_root"].to_numpy(dtype=float)
    root_depth = np.where(active, np.maximum(root_depth, float(crop_zmin)), np.nan)
    root_depth = np.minimum(root_depth, float(soil["dzsum"].max()))

    wr_fc = _integrated_storage(soil, root_depth, "th_fc")
    wr_wp = _integrated_storage(soil, root_depth, "th_wp")
    taw = wr_fc - wr_wp
    depletion = _safe_divide(wr_fc - flux["Wr"].to_numpy(dtype=float), taw)
    tr_ratio = _safe_divide(
        flux["Tr"].to_numpy(dtype=float),
        flux["TrPot"].to_numpy(dtype=float),
    )

    tile_drain = flux["TileDrain"] if "TileDrain" in flux else 0.0
    capillary_rise = flux["CR"] if "CR" in flux else 0.0
    groundwater_inflow = flux["GwIn"] if "GwIn" in flux else 0.0

    result = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "crop_day": crop_day,
            "active": active,
            "dap": crop["dap"].to_numpy(),
            "gdd_cum": crop["gdd_cum"].to_numpy(dtype=float),
            "z_root_m": root_depth,
            "canopy_cover": crop["canopy_cover"].to_numpy(dtype=float),
            "canopy_cover_ns": crop["canopy_cover_ns"].to_numpy(dtype=float),
            # AquaCrop raw biomass is g/m²; divide by 100 for t/ha.
            "biomass_t_ha": crop["biomass"].to_numpy(dtype=float) / 100.0,
            "biomass_ns_t_ha": crop["biomass_ns"].to_numpy(dtype=float) / 100.0,
            "dry_yield_t_ha": crop["DryYield"].to_numpy(dtype=float),
            "yield_pot_t_ha": crop["YieldPot"].to_numpy(dtype=float),
            "precip_mm": stress["precip_mm"].to_numpy(dtype=float),
            "irrigation_mm": flux["IrrDay"].to_numpy(dtype=float),
            "tr_mm": flux["Tr"].to_numpy(dtype=float),
            "tr_pot_mm": flux["TrPot"].to_numpy(dtype=float),
            "tr_ratio": tr_ratio,
            "es_mm": flux["Es"].to_numpy(dtype=float),
            "runoff_mm": flux["Runoff"].to_numpy(dtype=float),
            "deep_perc_mm": flux["DeepPerc"].to_numpy(dtype=float),
            "tile_drain_mm": np.asarray(tile_drain, dtype=float),
            "capillary_rise_mm": np.asarray(capillary_rise, dtype=float),
            "groundwater_inflow_mm": np.asarray(groundwater_inflow, dtype=float),
            "wr_mm": flux["Wr"].to_numpy(dtype=float),
            "wr_fc_mm": wr_fc,
            "wr_wp_mm": wr_wp,
            "taw_mm": taw,
            "depletion_fraction": depletion,
            "et0_mm": stress["et0_mm"].to_numpy(dtype=float),
            "p_exp_start": stress["p_exp_start"].to_numpy(dtype=float),
            "p_sto_start": stress["p_sto_start"].to_numpy(dtype=float),
            "p_sen_start": stress["p_sen_start"].to_numpy(dtype=float),
            "p_pol_start": stress["p_pol_start"].to_numpy(dtype=float),
            "p_exp_full": stress["p_exp_full"].to_numpy(dtype=float),
            "p_sto_full": stress["p_sto_full"].to_numpy(dtype=float),
            "p_sen_full": stress["p_sen_full"].to_numpy(dtype=float),
            "p_pol_full": stress["p_pol_full"].to_numpy(dtype=float),
        }
    )

    result["margin_to_stomatal_stress_fraction"] = (
        result["p_sto_start"] - result["depletion_fraction"]
    )

    result["margin_to_stomatal_stress_mm"] = (
        result["margin_to_stomatal_stress_fraction"] * result["taw_mm"]
    )

    result["stomatal_threshold_crossed"] = result["depletion_fraction"] >= result["p_sto_start"]

    # Crop-state outputs outside the active season often reset to zero. NaN is
    # safer for historical quantiles and prevents a post-harvest zero from being
    # interpreted as crop collapse.
    crop_state_columns = [
        "dap",
        "gdd_cum",
        "z_root_m",
        "canopy_cover",
        "canopy_cover_ns",
        "biomass_t_ha",
        "biomass_ns_t_ha",
        "dry_yield_t_ha",
        "yield_pot_t_ha",
        "tr_ratio",
        "wr_fc_mm",
        "wr_wp_mm",
        "taw_mm",
        "depletion_fraction",
        "p_exp_start",
        "p_sto_start",
        "p_sen_start",
        "p_pol_start",
        "p_exp_full",
        "p_sto_full",
        "p_sen_full",
        "p_pol_full",
        "margin_to_stomatal_stress_fraction",
        "margin_to_stomatal_stress_mm",
        "stomatal_threshold_crossed",
    ]
    result.loc[~result["active"], crop_state_columns] = np.nan
    result = result.loc[after_planting].reset_index(drop=True)

    cumulative_columns = {
        "precip_cum_mm": "precip_mm",
        "irrigation_cum_mm": "irrigation_mm",
        "tr_cum_mm": "tr_mm",
        "es_cum_mm": "es_mm",
        "runoff_cum_mm": "runoff_mm",
        "deep_perc_cum_mm": "deep_perc_mm",
        "tile_drain_cum_mm": "tile_drain_mm",
        "capillary_rise_cum_mm": "capillary_rise_mm",
        "groundwater_inflow_cum_mm": "groundwater_inflow_mm",
    }
    for cumulative, daily in cumulative_columns.items():
        result[cumulative] = result[daily].fillna(0.0).cumsum()

    active_dates = result.loc[result["active"], "date"]
    inferred_harvest = pd.Timestamp(active_dates.max())
    phenology = dict(configuration.get("phenology", {}) or {})
    phenology.setdefault("planting", planting_date)
    phenology.setdefault("harvest", inferred_harvest)
    phenology = {key: pd.Timestamp(value) for key, value in phenology.items()}

    start = pd.Timestamp(configuration["sim_start"])
    result.attrs.update(
        {
            "parcel": str(configuration["parcel"]),
            "scenario": str(configuration.get("scenario", "")),
            "year": int(configuration.get("year", start.year)),
            "sim_start": start,
            "planting_date": planting_date,
            "harvest_date": inferred_harvest,
            "phenology": phenology,
        }
    )
    return result


def _validate_run_groups(
    current_runs: Mapping[str, Mapping[str, Any]],
    historical_runs: Mapping[str, Mapping[str, Any]],
) -> None:
    if not current_runs:
        raise ValueError("current_runs cannot be empty.")
    if not historical_runs:
        raise ValueError("historical_runs cannot be empty.")

    current_parcels = [str(cfg.get("parcel", name)) for name, cfg in current_runs.items()]
    duplicates = sorted({p for p in current_parcels if current_parcels.count(p) > 1})
    if duplicates:
        raise ValueError(
            f"Provide exactly one current run per parcel. Duplicate current parcels: {duplicates}"
        )

    historical_pairs: list[tuple[str, int]] = []
    for name, cfg in historical_runs.items():
        parcel = str(cfg.get("parcel", name))
        start = pd.Timestamp(cfg["sim_start"])
        year = int(cfg.get("year", start.year))
        historical_pairs.append((parcel, year))
    duplicate_pairs = sorted(
        {pair for pair in historical_pairs if historical_pairs.count(pair) > 1}
    )
    if duplicate_pairs:
        raise ValueError(
            "Historical runs must contain at most one run per parcel/year. "
            f"Duplicates: {duplicate_pairs}"
        )


def _prepare_groups(
    current_runs: Mapping[str, Mapping[str, Any]],
    historical_runs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    _validate_run_groups(current_runs, historical_runs)
    current = {
        name: prepare_aquacrop_run(configuration) for name, configuration in current_runs.items()
    }
    historical = {
        name: prepare_aquacrop_run(configuration) for name, configuration in historical_runs.items()
    }

    current_parcels = {frame.attrs["parcel"] for frame in current.values()}
    historical_parcels = {frame.attrs["parcel"] for frame in historical.values()}
    missing = sorted(current_parcels - historical_parcels)
    if missing:
        raise ValueError(f"No historical runs were supplied for parcels: {missing}")
    return current, historical


def _current_by_parcel(current: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {frame.attrs["parcel"]: frame for frame in current.values()}


def _historical_by_parcel(
    historical: Mapping[str, pd.DataFrame],
) -> dict[str, list[pd.DataFrame]]:
    grouped: dict[str, list[pd.DataFrame]] = {}
    for frame in historical.values():
        grouped.setdefault(frame.attrs["parcel"], []).append(frame)
    for frames in grouped.values():
        frames.sort(key=lambda frame: frame.attrs["year"])
    return grouped


def _comparison_days(
    current: Mapping[str, pd.DataFrame],
    comparison_day: int | Mapping[str, int] | None,
) -> dict[str, int]:
    by_parcel = _current_by_parcel(current)
    result: dict[str, int] = {}
    for parcel, frame in by_parcel.items():
        if isinstance(comparison_day, Mapping):
            requested = comparison_day.get(parcel)
        else:
            requested = comparison_day
        maximum = int(frame["crop_day"].max())
        result[parcel] = maximum if requested is None else min(int(requested), maximum)
        if result[parcel] < 1:
            raise ValueError(f"comparison_day must be >= 1 for parcel {parcel!r}.")
    return result


def _truncate(
    frame: pd.DataFrame,
    comparison_day: int,
) -> pd.DataFrame:
    return frame.loc[frame["crop_day"].le(comparison_day)].copy()


def _summary_row(
    frame: pd.DataFrame,
    *,
    state: str,
    comparison_day: int,
    stress_threshold: float,
) -> dict[str, Any]:
    data = _truncate(frame, comparison_day)
    if data.empty:
        raise ValueError(
            f"Run {frame.attrs.get('parcel')} {frame.attrs.get('year')} does not "
            f"reach crop day {comparison_day}."
        )

    active = data.loc[data["active"]]
    last_active = active.iloc[-1] if not active.empty else pd.Series(dtype=float)
    tr_valid = active["tr_pot_mm"].gt(1e-8) & active["tr_ratio"].notna()

    return {
        "state": state,
        "parcel": frame.attrs["parcel"],
        "year": int(frame.attrs["year"]),
        "comparison_day": int(comparison_day),
        "available_through_day": int(data["crop_day"].max()),
        "covers_comparison_day": bool(data["crop_day"].max() >= comparison_day),
        "planting_date": frame.attrs["planting_date"],
        "harvest_date": frame.attrs["harvest_date"],
        "crop_active_at_comparison": bool(data.iloc[-1]["active"]),
        "final_gdd_cum": float(last_active.get("gdd_cum", np.nan)),
        "maximum_root_depth_m": float(active["z_root_m"].max()),
        "maximum_canopy_cover": float(active["canopy_cover"].max()),
        "current_biomass_t_ha": float(last_active.get("biomass_t_ha", np.nan)),
        "current_dry_yield_t_ha": float(last_active.get("dry_yield_t_ha", np.nan)),
        "current_potential_yield_t_ha": float(last_active.get("yield_pot_t_ha", np.nan)),
        "precipitation_mm": float(data["precip_mm"].sum()),
        "irrigation_mm": float(data["irrigation_mm"].sum()),
        "transpiration_mm": float(data["tr_mm"].sum()),
        "soil_evaporation_mm": float(data["es_mm"].sum()),
        "runoff_mm": float(data["runoff_mm"].sum()),
        "deep_percolation_mm": float(data["deep_perc_mm"].sum()),
        "transpiration_stress_days": int(
            (active.loc[tr_valid, "tr_ratio"] < stress_threshold).sum()
        ),
        "minimum_tr_over_trpot": float(active.loc[tr_valid, "tr_ratio"].min())
        if tr_valid.any()
        else np.nan,
        "maximum_root_zone_depletion_fraction": float(active["depletion_fraction"].max()),
        "minimum_stomatal_margin_mm": float(active["margin_to_stomatal_stress_mm"].min()),
        "stomatal_threshold_crossing_days": int(active["stomatal_threshold_crossed"].sum()),
    }


def build_run_summaries(
    current: Mapping[str, pd.DataFrame],
    historical: Mapping[str, pd.DataFrame],
    comparison_day_by_parcel: Mapping[str, int],
    *,
    stress_threshold: float = DEFAULT_STRESS_THRESHOLD,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in current.values():
        parcel = frame.attrs["parcel"]
        rows.append(
            _summary_row(
                frame,
                state="current",
                comparison_day=comparison_day_by_parcel[parcel],
                stress_threshold=stress_threshold,
            )
        )
    for frame in historical.values():
        parcel = frame.attrs["parcel"]
        rows.append(
            _summary_row(
                frame,
                state="historical",
                comparison_day=comparison_day_by_parcel[parcel],
                stress_threshold=stress_threshold,
            )
        )
    return pd.DataFrame(rows).sort_values(["parcel", "state", "year"]).reset_index(drop=True)


def _empirical_percentile(current_value: float, historical_values: np.ndarray) -> float:
    values = np.asarray(historical_values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(current_value) or len(values) == 0:
        return np.nan
    lower = np.sum(values < current_value)
    equal = np.sum(values == current_value)
    return 100.0 * (lower + 0.5 * equal) / len(values)


def build_metric_benchmark(
    run_summaries: pd.DataFrame,
    *,
    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    metrics: Mapping[str, Mapping[str, Any]] = BENCHMARK_METRICS,
) -> pd.DataFrame:
    q25, q50, q75 = quantiles
    rows: list[dict[str, Any]] = []
    for parcel in run_summaries["parcel"].unique():
        current_row = run_summaries.loc[
            (run_summaries["parcel"] == parcel) & (run_summaries["state"] == "current")
        ].iloc[0]
        history = run_summaries.loc[
            (run_summaries["parcel"] == parcel) & (run_summaries["state"] == "historical")
        ]
        for metric, info in metrics.items():
            if metric not in run_summaries:
                continue
            historical_values = history[metric].dropna().to_numpy(dtype=float)
            current_value = float(current_row[metric])
            if len(historical_values) == 0:
                historical_quantiles = [np.nan, np.nan, np.nan]
            else:
                historical_quantiles = np.quantile(
                    historical_values,
                    [q25, q50, q75],
                )
            rows.append(
                {
                    "parcel": parcel,
                    "comparison_day": int(current_row["comparison_day"]),
                    "metric": metric,
                    "label": info["label"],
                    "unit": info["unit"],
                    "higher_is_better": info.get("higher_is_better"),
                    "current": current_value,
                    "q25": float(historical_quantiles[0]),
                    "median": float(historical_quantiles[1]),
                    "q75": float(historical_quantiles[2]),
                    "percentile": _empirical_percentile(
                        current_value,
                        historical_values,
                    ),
                    "n_historical": int(len(historical_values)),
                }
            )
    return pd.DataFrame(rows)


def build_daily_quantiles(
    historical: Mapping[str, pd.DataFrame],
    comparison_day_by_parcel: Mapping[str, int],
    *,
    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    min_year_fraction: float = 0.75,
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute pointwise historical quantiles by parcel and crop day."""
    if not 0 < min_year_fraction <= 1:
        raise ValueError("min_year_fraction must be in (0, 1].")
    q25, q50, q75 = quantiles
    variables = tuple(
        variables
        or (
            "gdd_cum",
            "z_root_m",
            "canopy_cover",
            "biomass_t_ha",
            "depletion_fraction",
            "precip_cum_mm",
            "irrigation_cum_mm",
            "p_sto_start",
            "margin_to_stomatal_stress_mm",
        )
    )

    frames: list[pd.DataFrame] = []
    for frame in historical.values():
        parcel = frame.attrs["parcel"]
        data = _truncate(frame, comparison_day_by_parcel[parcel])
        selected = data[["crop_day", *variables]].copy()
        selected["parcel"] = parcel
        selected["year"] = int(frame.attrs["year"])
        frames.append(selected)
    long_source = pd.concat(frames, ignore_index=True)

    records: list[dict[str, Any]] = []
    for parcel, parcel_data in long_source.groupby("parcel", sort=False):
        n_total_years = int(parcel_data["year"].nunique())
        minimum_count = max(1, ceil(n_total_years * min_year_fraction))
        for variable in variables:
            grouped = parcel_data.groupby("crop_day")[variable]
            for crop_day, values in grouped:
                clean = values.dropna().to_numpy(dtype=float)
                if len(clean) < minimum_count:
                    continue
                qs = np.quantile(clean, [q25, q50, q75])
                records.append(
                    {
                        "parcel": parcel,
                        "crop_day": int(crop_day),
                        "variable": variable,
                        "q25": float(qs[0]),
                        "median": float(qs[1]),
                        "q75": float(qs[2]),
                        "n_years": int(len(clean)),
                        "n_total_years": n_total_years,
                    }
                )
    return pd.DataFrame(records)


def prepare_historical_benchmark(
    current_runs: Mapping[str, Mapping[str, Any]],
    historical_runs: Mapping[str, Mapping[str, Any]],
    *,
    comparison_day: int | Mapping[str, int] | None = None,
    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    min_year_fraction: float = 0.75,
    stress_threshold: float = DEFAULT_STRESS_THRESHOLD,
) -> HistoricalBenchmarkResult:
    """Prepare current and historical AquaCrop runs without plotting."""
    if tuple(sorted(quantiles)) != tuple(quantiles) or len(quantiles) != 3:
        raise ValueError("quantiles must contain three increasing probabilities.")
    if not all(0 <= value <= 1 for value in quantiles):
        raise ValueError("quantiles must be between 0 and 1.")

    current, historical = _prepare_groups(current_runs, historical_runs)
    comparison_days = _comparison_days(current, comparison_day)

    historical_by_parcel = _historical_by_parcel(historical)
    for parcel, frames in historical_by_parcel.items():
        if len(frames) < 4:
            warnings.warn(
                f"Only {len(frames)} historical runs were supplied for {parcel}. "
                "Quartile estimates will be unstable; 15–20 years are preferable.",
                stacklevel=2,
            )
        required_day = comparison_days[parcel]
        short_years = [
            frame.attrs["year"] for frame in frames if int(frame["crop_day"].max()) < required_day
        ]
        if short_years:
            warnings.warn(
                f"Historical runs for {parcel} do not all reach comparison day "
                f"{required_day}: {short_years}. Cumulative comparisons for these "
                "years will be shorter and should be rerun to a common endpoint.",
                stacklevel=2,
            )

    run_summaries = build_run_summaries(
        current,
        historical,
        comparison_days,
        stress_threshold=stress_threshold,
    )
    daily_quantiles = build_daily_quantiles(
        historical,
        comparison_days,
        quantiles=quantiles,
        min_year_fraction=min_year_fraction,
    )
    metric_benchmark = build_metric_benchmark(
        run_summaries,
        quantiles=quantiles,
    )
    return HistoricalBenchmarkResult(
        current=current,
        historical=historical,
        daily_quantiles=daily_quantiles,
        run_summaries=run_summaries,
        metric_benchmark=metric_benchmark,
        comparison_day_by_parcel=comparison_days,
    )


def _quantile_series(
    daily_quantiles: pd.DataFrame,
    parcel: str,
    variable: str,
) -> pd.DataFrame:
    result = daily_quantiles.loc[
        (daily_quantiles["parcel"] == parcel) & (daily_quantiles["variable"] == variable)
    ].sort_values("crop_day")
    return result


def _map_to_current_dates(
    quantiles: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    mapping = current[["crop_day", "date"]].drop_duplicates("crop_day")
    return quantiles.merge(mapping, on="crop_day", how="inner")


def _benchmark_row(
    benchmark: pd.DataFrame,
    parcel: str,
    metric: str,
) -> pd.Series | None:
    rows = benchmark.loc[(benchmark["parcel"] == parcel) & (benchmark["metric"] == metric)]
    return None if rows.empty else rows.iloc[0]


def _plot_historical_band(
    axis: plt.Axes,
    *,
    quantiles: pd.DataFrame,
    current: pd.DataFrame,
    parcel: str,
    variable: str,
    color: Any,
    current_label: str,
    history_label: str,
    current_linewidth: float = 1.9,
    median_linestyle: str = "--",
    band_alpha: float = 0.16,
    current_alpha: float = 1.0,
) -> None:
    q = _map_to_current_dates(
        _quantile_series(quantiles, parcel, variable),
        current,
    )
    if not q.empty:
        axis.fill_between(
            q["date"],
            q["q25"],
            q["q75"],
            color=color,
            alpha=band_alpha,
            linewidth=0,
            label=f"{parcel} — historical Q25–Q75",
        )
        axis.plot(
            q["date"],
            q["median"],
            color=color,
            linestyle=median_linestyle,
            linewidth=1.25,
            alpha=0.9,
            label=history_label,
        )
    active_current = current.loc[current[variable].notna()]
    axis.plot(
        active_current["date"],
        active_current[variable],
        color=color,
        linewidth=current_linewidth,
        alpha=current_alpha,
        label=current_label,
    )


def plot_aquacrop_historical_benchmark(
    current_runs: Mapping[str, Mapping[str, Any]],
    historical_runs: Mapping[str, Mapping[str, Any]],
    *,
    comparison_day: int | Mapping[str, int] | None = None,
    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    min_year_fraction: float = 0.75,
    stress_threshold: float = DEFAULT_STRESS_THRESHOLD,
    figsize: tuple[float, float] = (16, 14),
    title: str | None = None,
    threshold_similarity_tolerance: float = 0.02,
    show_stomatal_zone: bool = True,
) -> tuple[plt.Figure, HistoricalBenchmarkResult]:
    """Plot a six-panel current-versus-historical AquaCrop benchmark.

    Panels
    ------
    1. Cumulative GDD and effective root depth.
    2. Actual canopy cover and current no-water-stress reference.
    3. Biomass and current no-water-stress reference.
    4. Cumulative precipitation and irrigation.
    5. Root-zone reserve use with crop-water stress thresholds.
    6. Current cumulative water fluxes against historical median and IQR.
    """
    result = prepare_historical_benchmark(
        current_runs,
        historical_runs,
        comparison_day=comparison_day,
        quantiles=quantiles,
        min_year_fraction=min_year_fraction,
        stress_threshold=stress_threshold,
    )

    current_by_parcel = _current_by_parcel(result.current)
    parcels = list(current_by_parcel)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        parcel: default_colors[index % len(default_colors)] for index, parcel in enumerate(parcels)
    }

    figure, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
    figure.set_constrained_layout_pads(
        w_pad=0.05,
        h_pad=0.05,
        wspace=0.10,
        hspace=0.14,
    )
    ax_gdd, ax_canopy, ax_biomass, ax_inputs, ax_depletion, ax_fluxes = axes.flat
    ax_root = ax_gdd.twinx()

    # 1. Phenology and effective root depth -------------------------------
    for parcel, current in current_by_parcel.items():
        color = colors[parcel]
        _plot_historical_band(
            ax_gdd,
            quantiles=result.daily_quantiles,
            current=current,
            parcel=parcel,
            variable="gdd_cum",
            color=color,
            current_label=f"{parcel} — current GDD",
            history_label=f"{parcel} — historical median GDD",
            band_alpha=0.10,
        )
        _plot_historical_band(
            ax_root,
            quantiles=result.daily_quantiles,
            current=current,
            parcel=parcel,
            variable="z_root_m",
            color=color,
            current_label=f"{parcel} — current effective root depth",
            history_label=f"{parcel} — historical median root depth",
            current_linewidth=1.25,
            median_linestyle=":",
            band_alpha=0.08,
            current_alpha=0.75,
        )

        phenology = current.attrs.get("phenology", {})
        for event, linestyle in (("planting", ":"), ("harvest", "--")):
            date = phenology.get(event)
            if date is None:
                continue
            ax_gdd.axvline(
                date,
                color=color,
                linestyle=linestyle,
                linewidth=0.9,
                alpha=0.35,
            )
            ax_gdd.annotate(
                f"{parcel} {event}",
                xy=(date, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -3),
                textcoords="offset points",
                rotation=90,
                va="top",
                fontsize=7,
                color=color,
            )

    ax_gdd.set_title("Phenology and effective root development")
    ax_gdd.set_ylabel("Cumulative GDD (°C·day)")
    ax_root.set_ylabel("Effective root depth (m)")
    handles_1, labels_1 = ax_gdd.get_legend_handles_labels()
    handles_2, labels_2 = ax_root.get_legend_handles_labels()
    # Deduplicate repeated band labels generated by the two variables.
    combined = dict(zip(labels_1 + labels_2, handles_1 + handles_2))
    ax_gdd.legend(combined.values(), combined.keys(), fontsize=6.5, ncol=2)

    # 2. Canopy ------------------------------------------------------------
    for parcel, current in current_by_parcel.items():
        color = colors[parcel]
        _plot_historical_band(
            ax_canopy,
            quantiles=result.daily_quantiles,
            current=current,
            parcel=parcel,
            variable="canopy_cover",
            color=color,
            current_label=f"{parcel} — current actual",
            history_label=f"{parcel} — historical median actual",
        )
        reference = current.loc[current["canopy_cover_ns"].notna()]
        ax_canopy.plot(
            reference["date"],
            reference["canopy_cover_ns"],
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=f"{parcel} — current no-water-stress reference",
        )
    ax_canopy.set_title("Canopy development")
    ax_canopy.set_ylabel("Green canopy cover (fraction)")
    ax_canopy.set_ylim(0, 1.02)
    ax_canopy.legend(fontsize=6.5, ncol=2)

    # 3. Biomass -----------------------------------------------------------
    biomass_notes: list[str] = []
    for parcel, current in current_by_parcel.items():
        color = colors[parcel]
        _plot_historical_band(
            ax_biomass,
            quantiles=result.daily_quantiles,
            current=current,
            parcel=parcel,
            variable="biomass_t_ha",
            color=color,
            current_label=f"{parcel} — current biomass",
            history_label=f"{parcel} — historical median biomass",
        )
        reference = current.loc[current["biomass_ns_t_ha"].notna()]
        ax_biomass.plot(
            reference["date"],
            reference["biomass_ns_t_ha"],
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=f"{parcel} — current no-water-stress biomass",
        )
        active = current.loc[current["active"]]
        if not active.empty:
            endpoint = active.iloc[-1]
            ax_biomass.scatter(
                endpoint["date"],
                endpoint["dry_yield_t_ha"],
                color=color,
                s=34,
                zorder=5,
            )
        row = _benchmark_row(
            result.metric_benchmark,
            parcel,
            "current_dry_yield_t_ha",
        )
        if row is not None:
            biomass_notes.append(
                f"{parcel}: yield {row['current']:.1f} t/ha; "
                f"historical {row['median']:.1f} "
                f"[{row['q25']:.1f}–{row['q75']:.1f}]; "
                f"P{row['percentile']:.0f}"
            )
    ax_biomass.text(
        0.99,
        0.02,
        "\n".join(biomass_notes),
        transform=ax_biomass.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.8"},
    )
    ax_biomass.set_title("Biomass accumulation and dry yield")
    ax_biomass.set_ylabel("Dry biomass or yield (t/ha)")
    ax_biomass.legend(fontsize=6.5, ncol=2)

    # 4. Cumulative water inputs ------------------------------------------
    for parcel, current in current_by_parcel.items():
        color = colors[parcel]
        _plot_historical_band(
            ax_inputs,
            quantiles=result.daily_quantiles,
            current=current,
            parcel=parcel,
            variable="precip_cum_mm",
            color=color,
            current_label=f"{parcel} — current precipitation",
            history_label=f"{parcel} — historical median precipitation",
            current_linewidth=1.5,
            band_alpha=0.10,
        )
        _plot_historical_band(
            ax_inputs,
            quantiles=result.daily_quantiles,
            current=current,
            parcel=parcel,
            variable="irrigation_cum_mm",
            color=color,
            current_label=f"{parcel} — current irrigation",
            history_label=f"{parcel} — historical median irrigation",
            current_linewidth=2.1,
            median_linestyle=":",
            band_alpha=0.07,
        )
    ax_inputs.set_title("Cumulative precipitation and irrigation")
    ax_inputs.set_ylabel("Cumulative water depth (mm)")
    handles, labels = ax_inputs.get_legend_handles_labels()
    deduplicated = dict(zip(labels, handles))
    ax_inputs.legend(deduplicated.values(), deduplicated.keys(), fontsize=6.5, ncol=2)

    # 5. Root-zone reserve and physiological thresholds -------------------
    # Dr/TAW = 0 at field capacity and 1 at wilting point. Positive depletion
    # is normal reserve use; physiological limitation begins only when one of
    # AquaCrop's crop-specific thresholds is reached.
    for parcel, current in current_by_parcel.items():
        color = colors[parcel]
        _plot_historical_band(
            ax_depletion,
            quantiles=result.daily_quantiles,
            current=current,
            parcel=parcel,
            variable="depletion_fraction",
            color=color,
            current_label=f"{parcel} — current Dr/TAW",
            history_label=f"{parcel} — historical median Dr/TAW",
        )

        # Keep the closest approach to stomatal limitation parcel-specific,
        # even when the threshold curves themselves are shared visually.
        # margin_data = current.loc[
        #     current["active"]
        #     & current["margin_to_stomatal_stress_mm"].notna()
        #     & current["depletion_fraction"].notna()
        # ]
        # if not margin_data.empty:
        #     closest_index = margin_data["margin_to_stomatal_stress_mm"].idxmin()
        #     closest = margin_data.loc[closest_index]
        #     margin_mm = float(closest["margin_to_stomatal_stress_mm"])

        #     ax_depletion.scatter(
        #         [closest["date"]],
        #         [closest["depletion_fraction"]],
        #         facecolors="none",
        #         edgecolors=color,
        #         s=40,
        #         linewidths=1.2,
        #         zorder=6,
        #     )

        #     note = (
        #         f"{parcel}: closest margin {margin_mm:.0f} mm"
        #         if margin_mm >= 0
        #         else f"{parcel}: onset exceeded by {-margin_mm:.0f} mm"
        #     )
        #     ax_depletion.annotate(
        #         note,
        #         xy=(closest["date"], closest["depletion_fraction"]),
        #         xytext=(5, 7),
        #         textcoords="offset points",
        #         fontsize=7,
        #         color=color,
        #     )

        stressed = (
            current["active"]
            & current["tr_pot_mm"].gt(1e-8)
            & current["tr_ratio"].lt(stress_threshold)
        )
        if stressed.any():
            ax_depletion.scatter(
                current.loc[stressed, "date"],
                current.loc[stressed, "depletion_fraction"],
                color=color,
                marker="x",
                s=22,
                linewidths=1,
                zorder=5,
                label=f"{parcel} — current transpiration-stress day",
            )

    # If the parcel thresholds differ by at most the chosen tolerance, plot a
    # single neutral crop-threshold set. Otherwise retain one set per parcel.
    shared_thresholds = _thresholds_are_similar(
        current_by_parcel,
        tolerance=threshold_similarity_tolerance,
    )
    if shared_thresholds:
        representative = current_by_parcel[parcels[0]]
        _plot_crop_water_thresholds(
            ax_depletion,
            representative,
            color="0.30",
            show_stomatal_zone=show_stomatal_zone,
        )
    else:
        for parcel, current in current_by_parcel.items():
            _plot_crop_water_thresholds(
                ax_depletion,
                current,
                color=colors[parcel],
                label_prefix=f"{parcel} — ",
                show_stomatal_zone=show_stomatal_zone,
            )

    ax_depletion.axhline(0, color="0.35", linestyle=":", linewidth=1, label="Field capacity")
    ax_depletion.axhline(1, color="0.55", linestyle=":", linewidth=1, label="Wilting point")
    ax_depletion.set_title("Root-zone water reserve use and crop-water stress thresholds")
    ax_depletion.set_ylabel("Fraction depleted, Dr/TAW (0 = FC, 1 = WP)")
    handles, labels = ax_depletion.get_legend_handles_labels()
    deduplicated = dict(zip(labels, handles))
    ax_depletion.legend(deduplicated.values(), deduplicated.keys(), fontsize=6.5, ncol=2)

    # 6. Current fluxes versus historical median and IQR -------------------
    flux_metrics = [
        ("Precipitation", "precipitation_mm"),
        ("Irrigation", "irrigation_mm"),
        ("Transpiration", "transpiration_mm"),
        ("Soil evaporation", "soil_evaporation_mm"),
        ("Runoff", "runoff_mm"),
        ("Deep percolation", "deep_percolation_mm"),
    ]
    positions = np.arange(len(flux_metrics), dtype=float)
    n_parcels = max(1, len(parcels))
    width = 0.72 / n_parcels
    for parcel_index, parcel in enumerate(parcels):
        color = colors[parcel]
        offset = (parcel_index - (n_parcels - 1) / 2) * width
        current_values: list[float] = []
        historical_medians: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        for _, metric in flux_metrics:
            row = _benchmark_row(result.metric_benchmark, parcel, metric)
            if row is None:
                current_values.append(np.nan)
                historical_medians.append(np.nan)
                lower_errors.append(np.nan)
                upper_errors.append(np.nan)
                continue
            current_values.append(float(row["current"]))
            historical_medians.append(float(row["median"]))
            lower_errors.append(float(row["median"] - row["q25"]))
            upper_errors.append(float(row["q75"] - row["median"]))

        x = positions + offset
        ax_fluxes.bar(
            x,
            current_values,
            width=width * 0.82,
            color=color,
            alpha=0.78,
            label=f"{parcel} — current",
        )
        ax_fluxes.errorbar(
            x,
            historical_medians,
            yerr=np.vstack([lower_errors, upper_errors]),
            fmt="D",
            markersize=4,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.2,
            capsize=3,
            linewidth=1.1,
            label=f"{parcel} — historical median and Q25–Q75",
            zorder=5,
        )
    ax_fluxes.set_xticks(
        positions,
        [label for label, _ in flux_metrics],
        rotation=30,
        ha="right",
    )
    maximum_day = max(result.comparison_day_by_parcel.values())
    ax_fluxes.set_title(f"Cumulative water fluxes through crop day {maximum_day}")
    ax_fluxes.set_ylabel("Cumulative depth (mm)")
    ax_fluxes.legend(fontsize=6.5, ncol=2)

    for axis in axes.flat:
        axis.grid(True, alpha=0.24)
    for axis in [ax_gdd, ax_canopy, ax_biomass, ax_inputs, ax_depletion]:
        axis.tick_params(axis="x", rotation=25)

    if title:
        figure.suptitle(title)
    return figure, result


def compact_benchmark_table(
    metric_benchmark: pd.DataFrame,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a website-friendly wide table from ``metric_benchmark``."""
    selected = list(
        metrics
        or (
            "final_gdd_cum",
            "current_biomass_t_ha",
            "current_dry_yield_t_ha",
            "precipitation_mm",
            "irrigation_mm",
            "transpiration_mm",
            "deep_percolation_mm",
            "transpiration_stress_days",
            "maximum_root_zone_depletion_fraction",
            "minimum_stomatal_margin_mm",
            "stomatal_threshold_crossing_days",
        )
    )
    data = metric_benchmark.loc[metric_benchmark["metric"].isin(selected)].copy()
    data["historical_q25_q75"] = data.apply(
        lambda row: f"{row['q25']:.2f}–{row['q75']:.2f}", axis=1
    )
    return (
        data[
            [
                "parcel",
                "metric",
                "label",
                "unit",
                "current",
                "median",
                "historical_q25_q75",
                "percentile",
                "n_historical",
            ]
        ]
        .sort_values(["parcel", "metric"])
        .reset_index(drop=True)
    )


__all__ = [
    "BENCHMARK_METRICS",
    "HistoricalBenchmarkResult",
    "make_run_config",
    "prepare_aquacrop_run",
    "prepare_historical_benchmark",
    "plot_aquacrop_historical_benchmark",
    "build_daily_quantiles",
    "build_run_summaries",
    "build_metric_benchmark",
    "compact_benchmark_table",
]
