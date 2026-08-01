from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = {
    "z_top", "zBot", "zMid", "dz",
    "th_dry", "th_wp", "th_fc", "th_s", "Ksat",
}


def read_soil_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate an AquaCrop Soil.profile CSV export."""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.sort_values(["z_top", "zBot"]).reset_index(drop=True)

    if (df["zBot"] <= df["z_top"]).any():
        raise ValueError(f"{path} contains invalid compartment depths.")
    if not (df["th_dry"] <= df["th_wp"]).all():
        raise ValueError(f"{path}: expected th_dry <= th_wp.")
    if not (df["th_wp"] <= df["th_fc"]).all():
        raise ValueError(f"{path}: expected th_wp <= th_fc.")
    if not (df["th_fc"] <= df["th_s"]).all():
        raise ValueError(f"{path}: expected th_fc <= th_s.")
    if (df["Ksat"] <= 0).any():
        raise ValueError(f"{path}: Ksat must be strictly positive for log scaling.")

    return df


def _step_xy(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert compartment values to vertical step-profile coordinates."""
    x = np.repeat(df[column].to_numpy(float), 2)
    y = df[["z_top", "zBot"]].to_numpy(float).ravel()
    return x, y


def _effective_thickness(df: pd.DataFrame, depth: float) -> np.ndarray:
    """Thickness of each compartment lying between the surface and depth."""
    return np.clip(
        np.minimum(df["zBot"].to_numpy(float), depth)
        - df["z_top"].to_numpy(float),
        0.0,
        None,
    )


def soil_summary(df: pd.DataFrame, depth: float) -> dict[str, float]:
    """Depth-integrated hydraulic summary down to `depth` metres."""
    dz = _effective_thickness(df, depth)
    included = dz > 0
    if not included.any():
        raise ValueError("Comparison depth does not intersect the soil profile.")

    taw = np.sum((df["th_fc"] - df["th_wp"]).to_numpy() * dz) * 1000
    drainable = np.sum((df["th_s"] - df["th_fc"]).to_numpy() * dz) * 1000
    saturation_storage = np.sum(df["th_s"].to_numpy() * dz) * 1000
    median_ksat = np.median(df.loc[included, "Ksat"])

    return {
        "profile_depth_m": float(df["zBot"].max()),
        "comparison_depth_m": depth,
        "TAW_mm": taw,
        "drainable_water_mm": drainable,
        "saturation_storage_mm": saturation_storage,
        "median_Ksat_mm_day": median_ksat,
    }


def _plot_water_profile(
    ax: plt.Axes,
    df: pd.DataFrame,
    label: str,
    xlim: tuple[float, float],
) -> None:
    dry, depth = _step_xy(df, "th_dry")
    wp, _ = _step_xy(df, "th_wp")
    fc, _ = _step_xy(df, "th_fc")
    sat, _ = _step_xy(df, "th_s")

    ax.fill_betweenx(depth, dry, wp, alpha=0.18, label="Dry → wilting point")
    ax.fill_betweenx(depth, wp, fc, alpha=0.32, label="Plant-available water")
    ax.fill_betweenx(depth, fc, sat, alpha=0.18, label="FC → saturation")

    ax.plot(wp, depth, linewidth=1.7, label="Wilting point")
    ax.plot(fc, depth, linewidth=2.0, label="Field capacity")
    ax.plot(sat, depth, linewidth=1.7, label="Saturation")

    ax.set_title(label)
    ax.set_xlim(*xlim)
    ax.set_ylim(df["zBot"].max(), 0)
    ax.set_xlabel("Volumetric water content (m³/m³)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True, alpha=0.25)


def _cumulative_taw(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    taw_by_compartment = (
        (df["th_fc"] - df["th_wp"]) * (df["zBot"] - df["z_top"]) * 1000
    )
    depth = np.r_[0.0, df["zBot"].to_numpy(float)]
    cumulative = np.r_[0.0, taw_by_compartment.cumsum().to_numpy(float)]
    return cumulative, depth


def compare_soils(
    soil_a_path: str | Path,
    soil_b_path: str | Path,
    labels: tuple[str, str] = ("Parcel A", "Parcel B"),
    comparison_depth: float | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Compare two AquaCrop soil-profile CSV files.

    Parameters
    ----------
    comparison_depth:
        Depth in metres used for the summary metrics. If None, use the deepest
        depth shared by both profiles.
    """
    soils = [read_soil_csv(soil_a_path), read_soil_csv(soil_b_path)]

    common_depth = min(float(df["zBot"].max()) for df in soils)
    if comparison_depth is None:
        comparison_depth = common_depth
    if comparison_depth <= 0 or comparison_depth > common_depth:
        raise ValueError(
            f"comparison_depth must be > 0 and <= shared profile depth "
            f"({common_depth:.2f} m)."
        )

    water_min = min(float(df["th_dry"].min()) for df in soils)
    water_max = max(float(df["th_s"].max()) for df in soils)
    margin = 0.04 * (water_max - water_min)
    shared_xlim = (max(0, water_min - margin), water_max + margin)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    _plot_water_profile(axes[0, 0], soils[0], labels[0], shared_xlim)
    _plot_water_profile(axes[0, 1], soils[1], labels[1], shared_xlim)
    axes[0, 1].set_ylabel("")

    # One shared legend, outside the data panels.
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    ax = axes[1, 0]
    for df, label in zip(soils, labels):
        ksat, depth = _step_xy(df, "Ksat")
        ax.plot(ksat, depth, linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_ylim(max(df["zBot"].max() for df in soils), 0)
    ax.set_xlabel("Saturated hydraulic conductivity, Ksat (mm/day, log scale)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Drainage / infiltration capacity")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for df, label in zip(soils, labels):
        cumulative_taw, depth = _cumulative_taw(df)
        ax.plot(cumulative_taw, depth, linewidth=2, label=label)
    ax.axhline(comparison_depth, linestyle="--", linewidth=1.2, label="Summary depth")
    ax.set_ylim(max(df["zBot"].max() for df in soils), 0)
    ax.set_xlabel("Cumulative plant-available water (mm)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Root-zone water storage")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    summaries = pd.DataFrame(
        [soil_summary(df, comparison_depth) for df in soils],
        index=labels,
    )
    summaries.index.name = "parcel"

    fig.suptitle(
        f"AquaCrop soil comparison — metrics integrated to {comparison_depth:.2f} m",
        y=1.06,
        fontsize=14,
    )
    return fig, summaries


if __name__ == "__main__":
    fig, summary = compare_soils(
        "soil_1.csv",
        "soil_2.csv",
        labels=("Parcel 1", "Parcel 2"),
        comparison_depth=1.5,
    )
    print(summary.round(1))
    plt.show()
