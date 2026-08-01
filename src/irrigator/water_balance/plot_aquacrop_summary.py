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
            ("season_counter", "season index", "Zero-based crop-season index; -1 before the first active season."),
            ("dap", "days", "Calendar days after planting; zero outside the active crop season."),
            ("gdd", "°C·day", "Thermal time accumulated on the current day; mask outside the active season."),
            ("gdd_cum", "°C·day", "Cumulative thermal time since planting."),
            ("z_root", "m", "Current effective rooting depth."),
            ("canopy_cover", "fraction 0–1", "Actual green canopy cover."),
            ("canopy_cover_ns", "fraction 0–1", "Green canopy cover in the no-water-stress reference trajectory."),
            ("biomass", "g/m²", "Cumulative actual above-ground dry biomass; divide by 100 for t/ha or multiply by 10 for kg/ha."),
            ("biomass_ns", "g/m²", "Cumulative no-water-stress above-ground dry biomass."),
            ("harvest_index", "fraction 0–1", "Reference/unadjusted harvest-index build-up on the current day."),
            ("harvest_index_adj", "fraction 0–1", "Harvest index after water- and temperature-stress adjustments."),
            ("DryYield", "t/ha", "Cumulative actual dry yield: actual biomass × adjusted harvest index."),
            ("FreshYield", "t/ha", "Cumulative fresh yield, adjusted for the crop yield dry-matter fraction."),
            ("YieldPot", "t/ha", "Potential dry yield from no-water-stress biomass and the reference harvest index."),
        ]
    ),
    "water_flux": _metadata(
        [
            ("time_step_counter", "day index", "Zero-based simulation-day index."),
            ("season_counter", "season index", "Zero-based crop-season index; -1 before the first active season."),
            ("dap", "days", "Calendar days after planting."),
            ("Wr", "mm", "Actual water stored in the dynamically changing root zone."),
            ("z_gw", "m", "Groundwater-table depth below the soil surface; missing when no water table is configured."),
            ("surface_storage", "mm", "Water ponded or stored at the soil surface."),
            ("IrrDay", "mm/day", "Irrigation applied on the current day; net irrigation for irrigation method 4."),
            ("Infl", "mm/day", "Water infiltrating into the modelled soil profile after rainfall/irrigation partitioning."),
            ("Runoff", "mm/day", "Surface runoff."),
            ("DeepPerc", "mm/day", "Drainage/deep percolation leaving the bottom of the modelled soil profile."),
            ("TileDrain", "mm/day", "Optional lateral or tile-drain outflow."),
            ("CR", "mm/day", "Upward capillary rise from a shallow groundwater table."),
            ("GwIn", "mm/day", "Groundwater inflow that saturates compartments below a water table located within the profile."),
            ("Es", "mm/day", "Actual soil evaporation."),
            ("EsPot", "mm/day", "Potential soil evaporation."),
            ("Tr", "mm/day", "Actual crop transpiration."),
            ("TrPot", "mm/day", "Potential transpiration for the simulated canopy before current-day soil-water/aeration limitation."),
        ]
    ),
    "daily_stress": _metadata(
        [
            ("dap", "days", "Copy of crop_growth.dap."),
            ("gdd_cum", "°C·day", "Copy of crop_growth.gdd_cum."),
            ("ks", "fraction 0–1", "Custom Tr/TrPot ratio; not a universal native AquaCrop Ks coefficient."),
            ("canopy_cover", "fraction 0–1", "Copy of crop_growth.canopy_cover."),
            ("canopy_cover_ns", "fraction 0–1", "Copy of crop_growth.canopy_cover_ns."),
            ("biomass_kg_ha", "kg/ha", "Actual biomass after converting raw AquaCrop biomass with biomass × 10."),
            ("z_root_m", "m", "Copy of crop_growth.z_root."),
            ("harvest_index", "fraction 0–1", "Copy of the unadjusted crop_growth.harvest_index."),
            ("precip_mm", "mm/day", "Daily precipitation taken from the weather input."),
            ("irrigation_mm", "mm/day", "Copy of water_flux.IrrDay."),
            ("tr_mm", "mm/day", "Copy of water_flux.Tr."),
            ("tr_pot_mm", "mm/day", "Copy of water_flux.TrPot."),
            ("es_mm", "mm/day", "Copy of water_flux.Es."),
            ("deep_perc_mm", "mm/day", "Copy of water_flux.DeepPerc."),
            ("wr_mm", "mm", "Copy of water_flux.Wr."),
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

    dates = pd.Timestamp(sim_start) + pd.to_timedelta(
        cg["time_step_counter"], unit="D"
    )
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
        }
    )
    return result.loc[result["active"]].reset_index(drop=True)


def plot_aquacrop_season(
    parcels: Mapping[str, Mapping[str, Any]],
    figsize: tuple[float, float] = (15, 13),
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Plot a six-panel seasonal comparison for one or more parcels.

    Example
    -------
    parcels = {
        "Billac": {
            "crop_growth": pd.read_csv("billac_crop_growth.csv"),
            "water_flux": pd.read_csv("billac_water_flux.csv"),
            "daily_stress": pd.read_csv("billac_daily_stress.csv"),
            "soil": pd.read_csv("billac_soil.csv"),
            "sim_start": "2025-01-01",
            "season": 0,
        },
        "Guide": {
            ...
        },
    }
    fig, prepared = plot_aquacrop_season(parcels)
    plt.show()
    """
    prepared = {
        name: prepare_aquacrop_parcel(**configuration)
        for name, configuration in parcels.items()
    }

    figure, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
    ax_gdd, ax_canopy, ax_biomass, ax_inputs, ax_water, ax_totals = axes.flat

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        name: default_colors[index % len(default_colors)]
        for index, name in enumerate(prepared)
    }

    # 1. Phenology and rooting
    ax_root = ax_gdd.twinx()
    for name, data in prepared.items():
        color = colors[name]
        ax_gdd.plot(
            data["date"], data["gdd_cum"], color=color,
            label=f"{name} — cumulative GDD",
        )
        ax_root.plot(
            data["date"], data["z_root_m"], color=color, linestyle="--",
            label=f"{name} — root depth",
        )
    ax_gdd.set_title("Phenology and root development")
    ax_gdd.set_ylabel("Cumulative GDD (°C·day)")
    ax_root.set_ylabel("Root depth (m)")
    handles_1, labels_1 = ax_gdd.get_legend_handles_labels()
    handles_2, labels_2 = ax_root.get_legend_handles_labels()
    ax_gdd.legend(handles_1 + handles_2, labels_1 + labels_2, fontsize=8)

    # 2. Canopy actual versus no-water-stress reference
    for name, data in prepared.items():
        color = colors[name]
        ax_canopy.plot(
            data["date"], data["canopy_cover"], color=color,
            label=f"{name} — actual",
        )
        ax_canopy.plot(
            data["date"], data["canopy_cover_ns"], color=color, linestyle="--",
            label=f"{name} — no water stress",
        )
    ax_canopy.set_title("Canopy development")
    ax_canopy.set_ylabel("Green canopy cover (fraction)")
    ax_canopy.set_ylim(0, 1.02)
    ax_canopy.legend(fontsize=8)

    # 3. Biomass and yield formation; all quantities are t/ha
    for name, data in prepared.items():
        color = colors[name]
        ax_biomass.plot(
            data["date"], data["biomass_t_ha"], color=color,
            label=f"{name} — biomass",
        )
        ax_biomass.plot(
            data["date"], data["biomass_ns_t_ha"], color=color,
            linestyle="--", label=f"{name} — no-stress biomass",
        )
        ax_biomass.plot(
            data["date"], data["dry_yield_t_ha"], color=color,
            linestyle=":", label=f"{name} — dry yield",
        )
    ax_biomass.set_title("Biomass accumulation and dry yield")
    ax_biomass.set_ylabel("Dry matter or yield (t/ha)")
    ax_biomass.legend(fontsize=8)

    # 4. Weekly inputs are easier to read than overlapping daily bars
    for name, data in prepared.items():
        color = colors[name]
        weekly = (
            data.set_index("date")[["precip_mm", "irrigation_mm"]]
            .resample("7D")
            .sum()
        )
        ax_inputs.step(
            weekly.index, weekly["precip_mm"], where="mid", color=color,
            label=f"{name} — precipitation",
        )
        ax_inputs.step(
            weekly.index, weekly["irrigation_mm"], where="mid", color=color,
            linestyle="--", label=f"{name} — irrigation",
        )
    ax_inputs.set_title("Weekly water inputs")
    ax_inputs.set_ylabel("Water depth (mm/week)")
    ax_inputs.legend(fontsize=8)

    # 5. Put soil-water state and transpiration response on the same 0–1 direction:
    # 1 means water at field capacity / no transpiration limitation; 0 means WP / full limitation.
    for name, data in prepared.items():
        color = colors[name]
        ax_water.plot(
            data["date"], data["available_water_fraction"], color=color,
            label=f"{name} — relative available water",
        )
        ax_water.plot(
            data["date"], data["tr_ratio"], color=color, linestyle="--",
            label=f"{name} — Tr/TrPot",
        )
    ax_water.axhline(1, linestyle=":", linewidth=1, label="Field capacity / no limitation")
    ax_water.axhline(0, linestyle=":", linewidth=1, label="Wilting point / full limitation")
    ax_water.set_title("Root-zone water status and transpiration response")
    ax_water.set_ylabel("Relative level (fraction)")
    ax_water.legend(fontsize=8)

    # 6. Cumulative components; not labelled as exact balance closure because
    # a storage-change term would require the full water_storage output.
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
    bar_width = 0.8 / max(len(prepared), 1)
    for index, (name, data) in enumerate(prepared.items()):
        values = [data[column].sum() for _, column in categories]
        offset = (index - (len(prepared) - 1) / 2) * bar_width
        ax_totals.bar(
            positions + offset,
            values,
            width=bar_width,
            label=name,
        )
    ax_totals.set_xticks(
        positions,
        [label for label, _ in categories],
        rotation=30,
        ha="right",
    )
    ax_totals.set_title("Cumulative seasonal water components")
    ax_totals.set_ylabel("Cumulative depth (mm)")
    ax_totals.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    for axis in [ax_gdd, ax_canopy, ax_biomass, ax_inputs, ax_water]:
        axis.tick_params(axis="x", rotation=25)

    return figure, prepared
