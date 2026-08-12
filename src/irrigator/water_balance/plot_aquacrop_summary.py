"""Metadata audit and six-panel seasonal summary for AquaCrop-OSPy outputs.

The plotting functions accept one or more parcels. Each parcel supplies:
- crop_growth DataFrame
- water_flux DataFrame
- daily_stress DataFrame (used here for precipitation)
- soil profile DataFrame exported from Soil.Profile
- simulation start date

Raw AquaCrop biomass is converted from g/m² to t/ha by dividing by 100.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _metadata(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["variable", "unit", "meaning"])


AQUACROPMODEL_HELPER = {
    "crop_growth": _metadata(
        [
            ("time_step_counter", "day index", "Zero-based simulation-day index."),
            (
                "season_counter",
                "season index",
                "Zero-based crop-season index; -1 before the first active season.",
            ),
            ("dap", "days", "Calendar days after planting; zero outside the active crop season."),
            (
                "gdd",
                "°C·day",
                "Thermal time accumulated on the current day; mask outside the active season.",
            ),
            ("gdd_cum", "°C·day", "Cumulative thermal time since planting."),
            ("z_root", "m", "Current effective rooting depth."),
            ("canopy_cover", "fraction 0–1", "Actual green canopy cover."),
            (
                "canopy_cover_ns",
                "fraction 0–1",
                "Green canopy cover in the no-water-stress reference trajectory.",
            ),
            (
                "biomass",
                "g/m²",
                "Cumulative actual above-ground dry biomass; divide by 100 for t/ha or multiply by 10 for kg/ha.",
            ),
            ("biomass_ns", "g/m²", "Cumulative no-water-stress above-ground dry biomass."),
            (
                "harvest_index",
                "fraction 0–1",
                "Reference/unadjusted harvest-index build-up on the current day.",
            ),
            (
                "harvest_index_adj",
                "fraction 0–1",
                "Harvest index after water- and temperature-stress adjustments.",
            ),
            (
                "DryYield",
                "t/ha",
                "Cumulative actual dry yield: actual biomass × adjusted harvest index.",
            ),
            (
                "FreshYield",
                "t/ha",
                "Cumulative fresh yield, adjusted for the crop yield dry-matter fraction.",
            ),
            (
                "YieldPot",
                "t/ha",
                "Potential dry yield from no-water-stress biomass and the reference harvest index.",
            ),
        ]
    ),
    "water_flux": _metadata(
        [
            ("time_step_counter", "day index", "Zero-based simulation-day index."),
            (
                "season_counter",
                "season index",
                "Zero-based crop-season index; -1 before the first active season.",
            ),
            ("dap", "days", "Calendar days after planting."),
            ("Wr", "mm", "Actual water stored in the dynamically changing root zone."),
            (
                "z_gw",
                "m",
                "Groundwater-table depth below the soil surface; missing when no water table is configured.",
            ),
            ("surface_storage", "mm", "Water ponded or stored at the soil surface."),
            (
                "IrrDay",
                "mm/day",
                "Irrigation applied on the current day; net irrigation for irrigation method 4.",
            ),
            (
                "Infl",
                "mm/day",
                "Water infiltrating into the modelled soil profile after rainfall/irrigation partitioning.",
            ),
            ("Runoff", "mm/day", "Surface runoff."),
            (
                "DeepPerc",
                "mm/day",
                "Drainage/deep percolation leaving the bottom of the modelled soil profile.",
            ),
            ("TileDrain", "mm/day", "Optional lateral or tile-drain outflow."),
            ("CR", "mm/day", "Upward capillary rise from a shallow groundwater table."),
            (
                "GwIn",
                "mm/day",
                "Groundwater inflow that saturates compartments below a water table located within the profile.",
            ),
            ("Es", "mm/day", "Actual soil evaporation."),
            ("EsPot", "mm/day", "Potential soil evaporation."),
            ("Tr", "mm/day", "Actual crop transpiration."),
            (
                "TrPot",
                "mm/day",
                "Potential transpiration for the simulated canopy before current-day soil-water/aeration limitation.",
            ),
        ]
    ),
    "daily_stress": _metadata(
        [
            ("dap", "days", "Copy of crop_growth.dap."),
            ("gdd_cum", "°C·day", "Copy of crop_growth.gdd_cum."),
            (
                "ks",
                "fraction 0–1",
                "Custom Tr/TrPot ratio; not a universal native AquaCrop Ks coefficient.",
            ),
            ("canopy_cover", "fraction 0–1", "Copy of crop_growth.canopy_cover."),
            ("canopy_cover_ns", "fraction 0–1", "Copy of crop_growth.canopy_cover_ns."),
            (
                "biomass_kg_ha",
                "kg/ha",
                "Actual biomass after converting raw AquaCrop biomass with biomass × 10.",
            ),
            ("z_root_m", "m", "Copy of crop_growth.z_root."),
            ("harvest_index", "fraction 0–1", "Copy of the unadjusted crop_growth.harvest_index."),
            ("precip_mm", "mm/day", "Daily precipitation taken from the weather input."),
            ("irrigation_mm", "mm/day", "Copy of water_flux.IrrDay."),
            ("tr_mm", "mm/day", "Copy of water_flux.Tr."),
            ("tr_pot_mm", "mm/day", "Copy of water_flux.TrPot."),
            ("es_mm", "mm/day", "Copy of water_flux.Es."),
            ("deep_perc_mm", "mm/day", "Copy of water_flux.DeepPerc."),
            ("wr_mm", "mm", "Copy of water_flux.Wr."),
            (
                "et0_mm",
                "mm/day",
                "Daily reference evapotranspiration used by AquaCrop.",
            ),
            (
                "p_exp_start",
                "fraction 0–1",
                "Dynamic depletion threshold where canopy-expansion stress starts.",
            ),
            (
                "p_sto_start",
                "fraction 0–1",
                "Dynamic depletion threshold where stomatal stress starts.",
            ),
            (
                "p_sen_start",
                "fraction 0–1",
                "Dynamic depletion threshold where early-senescence stress starts.",
            ),
            (
                "p_pol_start",
                "fraction 0–1",
                "Depletion threshold where pollination stress starts.",
            ),
            (
                "p_exp_full",
                "fraction 0–1",
                "Depletion corresponding to maximum canopy-expansion stress.",
            ),
            (
                "p_sto_full",
                "fraction 0–1",
                "Depletion corresponding to maximum stomatal stress.",
            ),
            (
                "p_sen_full",
                "fraction 0–1",
                "Depletion corresponding to maximum senescence stress.",
            ),
            (
                "p_pol_full",
                "fraction 0–1",
                "Depletion corresponding to maximum pollination stress.",
            ),
        ]
    ),
}


def drop_csv_index(df: pd.DataFrame) -> pd.DataFrame:
    """Remove CSV index artefacts such as ``Unnamed: 0``."""
    return df.drop(
        columns=[column for column in df.columns if column.startswith("Unnamed:")],
        errors="ignore",
    ).copy()


def audit_output_metadata(
    frames: Mapping[str, pd.DataFrame],
    helper: Mapping[str, pd.DataFrame] = AQUACROPMODEL_HELPER,
) -> pd.DataFrame:
    """Report columns present in data but missing from metadata, and vice versa."""
    records: list[dict[str, str]] = []
    for name, frame in frames.items():
        actual = set(drop_csv_index(frame).columns)
        documented = set(helper[name]["variable"])
        records.append(
            {
                "dataset": name,
                "undocumented_columns": ", ".join(sorted(actual - documented)),
                "documented_but_absent": ", ".join(sorted(documented - actual)),
            }
        )
    return pd.DataFrame(records)


def fix_daily_stress_units(
    daily_stress: pd.DataFrame,
    crop_growth: pd.DataFrame,
) -> pd.DataFrame:
    """Correct the attached custom daily_stress biomass column and add useful fields."""
    ds = drop_csv_index(daily_stress)
    cg = drop_csv_index(crop_growth)
    if len(ds) != len(cg):
        raise ValueError("daily_stress and crop_growth must have the same row count.")

    # The attached file copied raw biomass directly despite naming it kg/ha.
    ds["biomass_kg_ha"] = cg["biomass"].to_numpy() * 10.0
    ds["biomass_ns_kg_ha"] = cg["biomass_ns"].to_numpy() * 10.0
    ds["harvest_index_adj"] = cg["harvest_index_adj"].to_numpy()
    ds["dry_yield_t_ha"] = cg["DryYield"].to_numpy()
    ds["yield_pot_t_ha"] = cg["YieldPot"].to_numpy()
    return ds


def _integrated_storage(
    soil: pd.DataFrame,
    depths_m: np.ndarray,
    theta_column: str,
) -> np.ndarray:
    """Integrate a volumetric water-content threshold to each root depth."""
    profile = drop_csv_index(soil).sort_values("dzsum")
    top = (profile["dzsum"] - profile["dz"]).to_numpy(float)
    bottom = profile["dzsum"].to_numpy(float)
    theta = profile[theta_column].to_numpy(float)
    profile_depth = bottom[-1]

    result = np.empty(len(depths_m), dtype=float)
    for i, depth in enumerate(np.asarray(depths_m, dtype=float)):
        depth = np.clip(depth, 0.0, profile_depth)
        overlap = np.clip(np.minimum(bottom, depth) - top, 0.0, bottom - top)
        result[i] = np.sum(theta * overlap * 1000.0)
    return result


def prepare_aquacrop_parcel(
    crop_growth: pd.DataFrame,
    water_flux: pd.DataFrame,
    daily_stress: pd.DataFrame,
    soil: pd.DataFrame,
    sim_start: str | pd.Timestamp,
    season: int = 0,
    crop_zmin: float | None = None,
) -> pd.DataFrame:
    """Create a clean, date-indexed seasonal table for one parcel."""
    cg = drop_csv_index(crop_growth)
    wf = drop_csv_index(water_flux)
    ds = drop_csv_index(daily_stress)
    soil = drop_csv_index(soil)

    if not (len(cg) == len(wf) == len(ds)):
        raise ValueError("crop_growth, water_flux and daily_stress must have equal lengths.")
    if not np.allclose(cg["dap"], wf["dap"], equal_nan=True):
        raise ValueError("crop_growth and water_flux are not aligned by DAP.")
    if not np.allclose(cg["dap"], ds["dap"], equal_nan=True):
        raise ValueError("daily_stress is not aligned with the raw AquaCrop outputs.")
    threshold_columns = [
        "p_exp_start",
        "p_sto_start",
        "p_sen_start",
        "p_pol_start",
        "p_exp_full",
        "p_sto_full",
        "p_sen_full",
        "p_pol_full",
    ]

    missing_thresholds = [c for c in threshold_columns if c not in ds.columns]

    if missing_thresholds:
        raise ValueError(
            "daily_stress is missing AquaCrop "
            "dynamic stress thresholds: "
            f"{missing_thresholds}. "
            "Re-run AquaCrop with the updated adapter."
        )
    dates = pd.Timestamp(sim_start) + pd.to_timedelta(cg["time_step_counter"], unit="D")
    active = cg["season_counter"].eq(season)

    if crop_zmin is None:
        positive_roots = cg.loc[active & cg["z_root"].gt(0), "z_root"]
        if positive_roots.empty:
            raise ValueError("Cannot infer the crop minimum rooting depth.")
        crop_zmin = float(positive_roots.min())

    root_depth = np.maximum(cg["z_root"].to_numpy(float), crop_zmin)
    root_depth = np.minimum(root_depth, float(soil["dzsum"].max()))
    wr_fc = _integrated_storage(soil, root_depth, "th_fc")
    wr_wp = _integrated_storage(soil, root_depth, "th_wp")
    taw = wr_fc - wr_wp

    depletion_fraction = np.divide(
        wr_fc - wf["Wr"].to_numpy(float),
        taw,
        out=np.full(len(wf), np.nan),
        where=taw > 0,
    )
    tr_ratio = np.divide(
        wf["Tr"].to_numpy(float),
        wf["TrPot"].to_numpy(float),
        out=np.full(len(wf), np.nan),
        where=wf["TrPot"].to_numpy(float) > 1e-8,
    )

    result = pd.DataFrame(
        {
            "date": dates,
            "active": active.to_numpy(),
            "dap": cg["dap"],
            "gdd_cum": cg["gdd_cum"],
            "z_root_m": cg["z_root"],
            "canopy_cover": cg["canopy_cover"],
            "canopy_cover_ns": cg["canopy_cover_ns"],
            "biomass_t_ha": cg["biomass"] / 100.0,
            "biomass_ns_t_ha": cg["biomass_ns"] / 100.0,
            "dry_yield_t_ha": cg["DryYield"],
            "yield_pot_t_ha": cg["YieldPot"],
            "harvest_index": cg["harvest_index"],
            "harvest_index_adj": cg["harvest_index_adj"],
            "precip_mm": ds["precip_mm"],
            "irrigation_mm": wf["IrrDay"],
            "tr_mm": wf["Tr"],
            "tr_pot_mm": wf["TrPot"],
            "tr_ratio": tr_ratio,
            "es_mm": wf["Es"],
            "es_pot_mm": wf["EsPot"],
            "runoff_mm": wf["Runoff"],
            "deep_perc_mm": wf["DeepPerc"],
            "tile_drain_mm": wf["TileDrain"],
            "capillary_rise_mm": wf["CR"],
            "groundwater_inflow_mm": wf["GwIn"],
            "wr_mm": wf["Wr"],
            "wr_fc_mm": wr_fc,
            "wr_wp_mm": wr_wp,
            "taw_mm": taw,
            "depletion_fraction": depletion_fraction,
            "available_water_fraction": 1.0 - depletion_fraction,
            "et0_mm": ds["et0_mm"],
            "p_exp_start": ds["p_exp_start"],
            "p_sto_start": ds["p_sto_start"],
            "p_sen_start": ds["p_sen_start"],
            "p_pol_start": ds["p_pol_start"],
            "p_exp_full": ds["p_exp_full"],
            "p_sto_full": ds["p_sto_full"],
            "p_sen_full": ds["p_sen_full"],
            "p_pol_full": ds["p_pol_full"],
        }
    )

    result["margin_to_stomatal_stress_fraction"] = (
        result["p_sto_start"] - result["depletion_fraction"]
    )
    result["margin_to_stomatal_stress_mm"] = (
        result["margin_to_stomatal_stress_fraction"] * result["taw_mm"]
    )

    result["stomatal_threshold_crossed"] = result["depletion_fraction"] >= result["p_sto_start"]

    return result.loc[result["active"]].reset_index(drop=True)


def _thresholds_are_similar(
    frames_by_parcel: Mapping[str, pd.DataFrame],
    columns: tuple[str, ...] = ("p_exp_start", "p_sto_start", "p_sto_full"),
    *,
    tolerance: float = 0.02,
) -> bool:
    """Return whether parcel threshold curves differ by at most ``tolerance``.

    Curves are compared on common active-season calendar dates. The tolerance
    is expressed as a fraction of TAW.
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
    """Plot canopy-expansion and stomatal water-stress thresholds."""
    active = data.loc[data["active"]].copy()
    if active.empty:
        return

    canopy_reference = active["canopy_cover_ns"]
    canopy_max = float(canopy_reference.max()) if canopy_reference.notna().any() else np.nan
    if np.isfinite(canopy_max) and canopy_max > 0:
        expansion = active.loc[
            active["p_exp_start"].notna() & active["canopy_cover_ns"].lt(0.98 * canopy_max)
        ]
        if not expansion.empty:
            axis.plot(
                expansion["date"],
                expansion["p_exp_start"],
                color=color,
                linestyle=":",
                linewidth=1.05,
                alpha=0.72,
                label=f"{label_prefix}canopy-expansion stress onset",
            )

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

    if show_stomatal_zone:
        axis.fill_between(
            stomatal["date"],
            stomatal["p_sto_start"],
            stomatal["p_sto_full"],
            color=color,
            alpha=0.045,
            linewidth=0,
            label=f"{label_prefix}increasing stomatal limitation",
        )


def _scenario_linestyle(scenario: str) -> str:
    """Return a stable line style for common irrigation scenario names."""
    scenario_lower = scenario.lower()
    if any(
        token in scenario_lower
        for token in ("no irrigation", "without irrigation", "rainfed", "unirrigated", "past")
    ):
        return "--"
    if any(token in scenario_lower for token in ("irrigated", "irrigation", "baseline", "current")):
        return "-"
    return "-."


def _scenario_hatch(scenario: str) -> str:
    """Return a stable hatch for grouped bars."""
    scenario_lower = scenario.lower()
    if any(
        token in scenario_lower
        for token in ("no irrigation", "without irrigation", "rainfed", "unirrigated", "past")
    ):
        return "//"
    return ""


def _display_name(parcel: str, scenario: str) -> str:
    return f"{parcel} — {scenario}"


def _extract_plot_metadata(
    name: str,
    configuration: Mapping[str, Any],
) -> tuple[str, str, dict[str, pd.Timestamp]]:
    """Read plotting metadata without passing it to prepare_aquacrop_parcel."""
    parcel = str(configuration.get("parcel", name))
    scenario = str(configuration.get("scenario", "Irrigated"))

    phenology_cfg = configuration.get("phenology", {}) or {}
    phenology: dict[str, pd.Timestamp] = {}
    for event in ("planting", "flowering", "harvest"):
        value = phenology_cfg.get(event)
        if value is not None:
            phenology[event] = pd.Timestamp(value)
    return parcel, scenario, phenology


def _prepare_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only arguments accepted by prepare_aquacrop_parcel."""
    allowed = {
        "crop_growth",
        "water_flux",
        "daily_stress",
        "soil",
        "sim_start",
        "season",
        "crop_zmin",
    }
    return {key: value for key, value in configuration.items() if key in allowed}


def summarize_aquacrop_seasons(
    prepared: Mapping[str, pd.DataFrame],
    series_metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """Return one seasonal summary row per parcel/scenario series."""
    rows: list[dict[str, Any]] = []
    for name, data in prepared.items():
        metadata = series_metadata.get(name, {}) if series_metadata else {}
        tr_pot_positive = data["tr_pot_mm"].gt(1e-8)
        stress_days = int((data.loc[tr_pot_positive, "tr_ratio"] < 0.99).sum())
        rows.append(
            {
                "series": name,
                "parcel": metadata.get("parcel", name),
                "scenario": metadata.get("scenario", ""),
                "planting_date": data["date"].min(),
                "harvest_date": data["date"].max(),
                "season_length_days": int(len(data)),
                "final_gdd_cum": float(data["gdd_cum"].iloc[-1]),
                "maximum_root_depth_m": float(data["z_root_m"].max()),
                "maximum_canopy_cover": float(data["canopy_cover"].max()),
                "final_biomass_t_ha": float(data["biomass_t_ha"].iloc[-1]),
                "final_dry_yield_t_ha": float(data["dry_yield_t_ha"].iloc[-1]),
                "potential_dry_yield_t_ha": float(data["yield_pot_t_ha"].iloc[-1]),
                "precipitation_mm": float(data["precip_mm"].sum()),
                "irrigation_mm": float(data["irrigation_mm"].sum()),
                "transpiration_mm": float(data["tr_mm"].sum()),
                "soil_evaporation_mm": float(data["es_mm"].sum()),
                "runoff_mm": float(data["runoff_mm"].sum()),
                "deep_percolation_mm": float(data["deep_perc_mm"].sum()),
                "transpiration_stress_days": stress_days,
                "minimum_tr_over_trpot": float(data.loc[tr_pot_positive, "tr_ratio"].min())
                if tr_pot_positive.any()
                else np.nan,
                "maximum_root_zone_depletion_fraction": float(data["depletion_fraction"].max()),
                "minimum_stomatal_margin_mm": float(data["margin_to_stomatal_stress_mm"].min()),
                "stomatal_threshold_crossing_days": int(data["stomatal_threshold_crossed"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_aquacrop_season(
    parcels: Mapping[str, Mapping[str, Any]],
    figsize: tuple[float, float] = (16, 14),
    weekly_anchor: str = "W-MON",
    threshold_similarity_tolerance: float = 0.02,
    show_stomatal_zone: bool = True,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Plot a six-panel comparison for one or more parcel/scenario runs.

    Color encodes the parcel and line style encodes the scenario. This is
    specifically designed for comparisons such as irrigated versus no-irrigation
    runs for Billac and Guide.

    Optional plotting metadata can be added to each configuration:

    ``parcel``
        Parcel name used for color grouping.
    ``scenario``
        Scenario name used for line style and hatch grouping.
    ``phenology``
        Optional mapping with explicit ``planting``, ``flowering`` and/or
        ``harvest`` dates. Planting and harvest are inferred from the active
        season when omitted.

    Example
    -------
    parcels = {
        "Billac irrigated": {
            "parcel": "Billac",
            "scenario": "Irrigated",
            "crop_growth": billac_crop,
            "water_flux": billac_flux,
            "daily_stress": billac_stress,
            "soil": billac_soil,
            "sim_start": "2025-01-01",
            "season": 0,
        },
        "Billac no irrigation": {
            "parcel": "Billac",
            "scenario": "No irrigation",
            ...
        },
    }
    """
    prepared: dict[str, pd.DataFrame] = {}
    series_metadata: dict[str, dict[str, Any]] = {}

    for name, configuration in parcels.items():
        prepared[name] = prepare_aquacrop_parcel(**_prepare_configuration(configuration))
        parcel, scenario, phenology = _extract_plot_metadata(name, configuration)
        data = prepared[name]
        phenology.setdefault("planting", pd.Timestamp(data["date"].min()))
        phenology.setdefault("harvest", pd.Timestamp(data["date"].max()))
        series_metadata[name] = {
            "parcel": parcel,
            "scenario": scenario,
            "phenology": phenology,
        }

    figure, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
    figure.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.08, hspace=0.12)
    ax_gdd, ax_canopy, ax_biomass, ax_inputs, ax_water, ax_totals = axes.flat

    parcel_order = list(dict.fromkeys(meta["parcel"] for meta in series_metadata.values()))
    scenario_order = list(dict.fromkeys(meta["scenario"] for meta in series_metadata.values()))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    parcel_colors = {
        parcel: default_colors[index % len(default_colors)]
        for index, parcel in enumerate(parcel_order)
    }
    scenario_styles = {scenario: _scenario_linestyle(scenario) for scenario in scenario_order}

    # 1. Phenology and root development -------------------------------------
    ax_root = ax_gdd.twinx()

    # GDD is weather-driven and should be identical between irrigation
    # scenarios for one parcel, so plot it only once per parcel.
    for parcel in parcel_order:
        representative_name = next(
            name for name, meta in series_metadata.items() if meta["parcel"] == parcel
        )
        data = prepared[representative_name]
        ax_gdd.plot(
            data["date"],
            data["gdd_cum"],
            color=parcel_colors[parcel],
            linewidth=1.8,
            label=f"{parcel} — cumulative GDD",
        )

    # Root development may differ under water stress, so retain one curve per
    # parcel/scenario run.
    for name, data in prepared.items():
        meta = series_metadata[name]
        parcel = meta["parcel"]
        scenario = meta["scenario"]
        color = parcel_colors[parcel]
        linestyle = scenario_styles[scenario]
        display = _display_name(parcel, scenario)
        ax_root.plot(
            data["date"],
            data["z_root_m"],
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            alpha=0.75,
            label=f"{display} — root depth",
        )

    # One set of phenology markers per parcel, not per irrigation scenario.
    for parcel in parcel_order:
        representative_name = next(
            name for name, meta in series_metadata.items() if meta["parcel"] == parcel
        )
        phenology = series_metadata[representative_name]["phenology"]
        color = parcel_colors[parcel]
        event_styles = {"planting": ":", "flowering": "-.", "harvest": "--"}
        for event, date in phenology.items():
            ax_gdd.axvline(
                date,
                color=color,
                linestyle=event_styles[event],
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
                ha="left",
                fontsize=7,
                color=color,
                alpha=0.8,
            )

    ax_gdd.set_title("Phenology and root development")
    ax_gdd.set_ylabel("Cumulative GDD (°C·day)")
    ax_root.set_ylabel("Root depth (m)")
    handles_1, labels_1 = ax_gdd.get_legend_handles_labels()
    handles_2, labels_2 = ax_root.get_legend_handles_labels()
    ax_gdd.legend(handles_1 + handles_2, labels_1 + labels_2, fontsize=7, ncol=2)

    # 2. Canopy development -------------------------------------------------
    # Plot the no-stress reference once per parcel to avoid duplicate curves.
    for parcel in parcel_order:
        representative_name = next(
            name for name, meta in series_metadata.items() if meta["parcel"] == parcel
        )
        data = prepared[representative_name]
        ax_canopy.plot(
            data["date"],
            data["canopy_cover_ns"],
            color=parcel_colors[parcel],
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=f"{parcel} — no-water-stress reference",
        )

    for name, data in prepared.items():
        meta = series_metadata[name]
        parcel = meta["parcel"]
        scenario = meta["scenario"]
        ax_canopy.plot(
            data["date"],
            data["canopy_cover"],
            color=parcel_colors[parcel],
            linestyle=scenario_styles[scenario],
            linewidth=1.8,
            label=f"{_display_name(parcel, scenario)} — actual",
        )
    ax_canopy.set_title("Canopy development")
    ax_canopy.set_ylabel("Green canopy cover (fraction)")
    ax_canopy.set_ylim(0, 1.02)
    ax_canopy.legend(fontsize=7, ncol=2)

    # 3. Biomass and final yield --------------------------------------------
    for parcel in parcel_order:
        representative_name = next(
            name for name, meta in series_metadata.items() if meta["parcel"] == parcel
        )
        data = prepared[representative_name]
        ax_biomass.plot(
            data["date"],
            data["biomass_ns_t_ha"],
            color=parcel_colors[parcel],
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=f"{parcel} — no-water-stress biomass",
        )

    biomass_summary_lines: list[str] = []
    for name, data in prepared.items():
        meta = series_metadata[name]
        parcel = meta["parcel"]
        scenario = meta["scenario"]
        color = parcel_colors[parcel]
        display = _display_name(parcel, scenario)
        final_biomass = float(data["biomass_t_ha"].iloc[-1])
        final_yield = float(data["dry_yield_t_ha"].iloc[-1])

        ax_biomass.plot(
            data["date"],
            data["biomass_t_ha"],
            color=color,
            linestyle=scenario_styles[scenario],
            linewidth=1.8,
            label=f"{display} — biomass",
        )
        ax_biomass.scatter(
            data["date"].iloc[-1],
            final_yield,
            color=color,
            marker="o" if scenario_styles[scenario] == "-" else "s",
            s=28,
            zorder=4,
        )
        biomass_summary_lines.append(
            f"{display}: biomass {final_biomass:.1f}, yield {final_yield:.1f} t/ha"
        )

    ax_biomass.text(
        0.99,
        0.02,
        "\n".join(biomass_summary_lines),
        transform=ax_biomass.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
    )
    ax_biomass.set_title("Biomass accumulation and final dry yield")
    ax_biomass.set_ylabel("Dry biomass (t/ha)")
    ax_biomass.legend(fontsize=7, ncol=2)

    # 4. Weekly water inputs as bars ----------------------------------------
    # Precipitation is plotted once per parcel because it should be identical
    # between irrigation scenarios. Irrigation is plotted for each non-zero run.
    parcel_weekly: dict[str, pd.DataFrame] = {}
    all_weekly_dates: set[pd.Timestamp] = set()

    for parcel in parcel_order:
        names = [name for name, meta in series_metadata.items() if meta["parcel"] == parcel]
        representative = prepared[names[0]].set_index("date")
        weekly_precip = representative[["precip_mm"]].resample(weekly_anchor).sum()

        # Detect accidental weather mismatch between scenarios for one parcel.
        # for other_name in names[1:]:
        #     other = prepared[other_name].set_index("date")[["precip_mm"]].resample(weekly_anchor).sum()
        #     aligned = weekly_precip.join(other, how="outer", lsuffix="_ref", rsuffix="_other").fillna(0)
        #     print(aligned)
        #     if not np.allclose(aligned["precip_mm_ref"], aligned["precip_mm_other"], atol=1e-6):
        #         raise ValueError(
        #             f"Precipitation differs between scenarios for parcel {parcel!r}. "
        #             "Use the same weather forcing when comparing irrigation strategies."
        #         )

        parcel_weekly[parcel] = weekly_precip
        all_weekly_dates.update(weekly_precip.index.to_pydatetime())

    n_parcels = max(len(parcel_order), 1)
    group_width_days = 5.4
    parcel_bar_width = group_width_days / n_parcels
    for parcel_index, parcel in enumerate(parcel_order):
        color = parcel_colors[parcel]
        offset_days = (parcel_index - (n_parcels - 1) / 2) * parcel_bar_width
        weekly_precip = parcel_weekly[parcel]
        x_precip = weekly_precip.index + pd.to_timedelta(offset_days, unit="D")
        ax_inputs.bar(
            x_precip,
            weekly_precip["precip_mm"],
            width=parcel_bar_width * 0.9,
            color=color,
            alpha=0.25,
            label=f"{parcel} — precipitation",
            align="center",
        )

        parcel_names = [name for name, meta in series_metadata.items() if meta["parcel"] == parcel]
        nonzero_irrigation_names = [
            name for name in parcel_names if prepared[name]["irrigation_mm"].sum() > 1e-8
        ]
        n_irrigation = max(len(nonzero_irrigation_names), 1)
        irrigation_width = parcel_bar_width * 0.55 / n_irrigation
        for irrigation_index, name in enumerate(nonzero_irrigation_names):
            data = prepared[name].set_index("date")
            weekly_irrigation = data[["irrigation_mm"]].resample(weekly_anchor).sum()
            scenario = series_metadata[name]["scenario"]
            within_offset = (irrigation_index - (n_irrigation - 1) / 2) * irrigation_width
            x_irrigation = weekly_irrigation.index + pd.to_timedelta(
                offset_days + within_offset, unit="D"
            )
            ax_inputs.bar(
                x_irrigation,
                weekly_irrigation["irrigation_mm"],
                width=irrigation_width * 0.9,
                color=color,
                alpha=0.85,
                hatch=_scenario_hatch(scenario),
                label=f"{_display_name(parcel, scenario)} — irrigation",
                align="center",
            )

    ax_inputs.set_title("Weekly precipitation and irrigation")
    ax_inputs.set_ylabel("Water depth (mm/week)")
    ax_inputs.legend(fontsize=7, ncol=2)

    # 5. Root-zone reserve and physiological thresholds --------------------
    # Positive Dr/TAW is normal use of the soil-water reserve. Water stress
    # starts only once process-specific AquaCrop depletion thresholds are met.
    for name, data in prepared.items():
        meta = series_metadata[name]
        parcel = meta["parcel"]
        scenario = meta["scenario"]
        color = parcel_colors[parcel]
        stress_mask = data["tr_pot_mm"].gt(1e-8) & data["tr_ratio"].lt(0.99)
        stress_days = int(stress_mask.sum())
        label = f"{_display_name(parcel, scenario)} — Dr/TAW ({stress_days} stress d)"

        ax_water.plot(
            data["date"],
            data["depletion_fraction"],
            color=color,
            linestyle=scenario_styles[scenario],
            linewidth=1.7,
            label=label,
        )
        if stress_mask.any():
            ax_water.scatter(
                data.loc[stress_mask, "date"],
                data.loc[stress_mask, "depletion_fraction"],
                color=color,
                marker="x",
                s=18,
                linewidths=0.9,
                zorder=4,
            )

    # Thresholds depend on crop parameters and ET0, not irrigation scenario.
    # Use one representative run per parcel and collapse to one neutral set if
    # parcel curves differ by no more than the requested tolerance.
    thresholds_by_parcel: dict[str, pd.DataFrame] = {}
    for parcel in parcel_order:
        representative_name = next(
            name for name, meta in series_metadata.items() if meta["parcel"] == parcel
        )
        thresholds_by_parcel[parcel] = prepared[representative_name]

    shared_thresholds = _thresholds_are_similar(
        thresholds_by_parcel,
        tolerance=threshold_similarity_tolerance,
    )
    if shared_thresholds:
        representative = thresholds_by_parcel[parcel_order[0]]
        _plot_crop_water_thresholds(
            ax_water,
            representative,
            color="0.30",
            show_stomatal_zone=show_stomatal_zone,
        )
    else:
        for parcel, data in thresholds_by_parcel.items():
            _plot_crop_water_thresholds(
                ax_water,
                data,
                color=parcel_colors[parcel],
                label_prefix=f"{parcel} — ",
                show_stomatal_zone=show_stomatal_zone,
            )

    ax_water.axhline(0, linestyle=":", linewidth=1, color="0.35", label="Field capacity")
    ax_water.axhline(1, linestyle=":", linewidth=1, color="0.55", label="Wilting point")
    ax_water.set_title("Root-zone water reserve use and crop-water stress thresholds")
    ax_water.set_ylabel("Fraction depleted, Dr/TAW (0 = FC, 1 = WP)")
    handles, labels = ax_water.get_legend_handles_labels()
    deduplicated = dict(zip(labels, handles))
    ax_water.legend(deduplicated.values(), deduplicated.keys(), fontsize=7, ncol=2)
    # 6. Cumulative seasonal water fluxes -----------------------------------
    categories = [
        ("Precipitation", "precip_mm"),
        ("Irrigation", "irrigation_mm"),
        ("Transpiration", "tr_mm"),
        ("Soil evaporation", "es_mm"),
        ("Runoff", "runoff_mm"),
        ("Deep percolation", "deep_perc_mm"),
    ]
    optional_categories = [
        ("Tile drainage", "tile_drain_mm"),
        ("Capillary rise", "capillary_rise_mm"),
        ("Groundwater inflow", "groundwater_inflow_mm"),
    ]
    for label, column in optional_categories:
        if any(data[column].sum() > 1e-6 for data in prepared.values()):
            categories.append((label, column))

    positions = np.arange(len(categories))
    n_series = max(len(prepared), 1)
    bar_width = 0.84 / n_series
    for index, (name, data) in enumerate(prepared.items()):
        meta = series_metadata[name]
        parcel = meta["parcel"]
        scenario = meta["scenario"]
        values = [float(data[column].sum()) for _, column in categories]
        offset = (index - (n_series - 1) / 2) * bar_width
        ax_totals.bar(
            positions + offset,
            values,
            width=bar_width,
            color=parcel_colors[parcel],
            alpha=0.9,
            hatch=_scenario_hatch(scenario),
            edgecolor="white",
            linewidth=0.5,
            label=_display_name(parcel, scenario),
        )
    ax_totals.set_xticks(
        positions,
        [label for label, _ in categories],
        rotation=30,
        ha="right",
    )
    ax_totals.set_title("Cumulative seasonal water fluxes")
    ax_totals.set_ylabel("Cumulative depth (mm)")
    ax_totals.legend(fontsize=7, ncol=2)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    for axis in [ax_gdd, ax_canopy, ax_biomass, ax_inputs, ax_water]:
        axis.tick_params(axis="x", rotation=25)

    # Keep metadata available without changing the historical return signature.
    for name, data in prepared.items():
        data.attrs.update(series_metadata[name])

    return figure, prepared
