"""Parcel-level soil profile extraction.

Loads processed EU-SoilHydroGrids (soil_hydro.nc from Block 0) and extracts
soil hydraulic parameters at each parcel location.  Farmer-provided soil
analyses override gridded values when available.

Pedotransfer functions (Rosetta-style) convert texture to hydraulic
properties when farmers provide sand/silt/clay percentages but not
FC/WP directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer

from irrigator.config import ParcelConfig, RegionConfig

logger = logging.getLogger(__name__)

# Transformer WGS84 → Lambert-93 (reused across calls)
_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)


@dataclass
class SoilProfile:
    """Soil hydraulic parameters for one parcel.

    All values are for the effective root zone.

    Attributes
    ----------
    theta_fc : field capacity per depth layer [cm³/cm³]
    theta_wp : wilting point per depth layer [cm³/cm³]
    k_sat : saturated hydraulic conductivity per depth layer [cm/day]
    z_layers_cm : depth boundaries [(top, bottom), ...] in cm
    total_awc_mm : total available water capacity over root zone [mm]
    max_root_depth_m : max rooting depth (from crop config or soil constraint)
    source : "eu_soilhydrogrids", "farmer_analysis", or "probe_calibrated"
    """

    theta_fc: list[float]
    theta_wp: list[float]
    k_sat: list[float]
    z_layers_cm: list[tuple[int, int]]
    total_awc_mm: float
    max_root_depth_m: float
    source: str

    @property
    def n_layers(self) -> int:
        return len(self.z_layers_cm)

    def awc_for_root_depth(self, root_depth_m: float) -> float:
        """Compute AWC [mm] for a given root depth, interpolating layers."""
        root_cm = root_depth_m * 100
        awc_mm = 0.0
        for i, (ztop, zbot) in enumerate(self.z_layers_cm):
            if ztop >= root_cm:
                break
            effective_bot = min(zbot, root_cm)
            thickness_cm = effective_bot - ztop
            awc_cm3 = (self.theta_fc[i] - self.theta_wp[i]) * thickness_cm
            awc_mm += max(0, awc_cm3) * 10  # cm → mm
        return awc_mm

    def effective_fc_wp(self, root_depth_m: float) -> tuple[float, float]:
        """Depth-weighted average FC and WP over root zone."""
        root_cm = root_depth_m * 100
        total_thickness = 0.0
        weighted_fc = 0.0
        weighted_wp = 0.0
        for i, (ztop, zbot) in enumerate(self.z_layers_cm):
            if ztop >= root_cm:
                break
            effective_bot = min(zbot, root_cm)
            t = effective_bot - ztop
            weighted_fc += self.theta_fc[i] * t
            weighted_wp += self.theta_wp[i] * t
            total_thickness += t
        if total_thickness == 0:
            return (self.theta_fc[0], self.theta_wp[0])
        return (weighted_fc / total_thickness, weighted_wp / total_thickness)


# ---------------------------------------------------------------------------
# Pedotransfer functions (Wösten et al. 1999, continuous PTFs)
# ---------------------------------------------------------------------------


def _pedotransfer_fc_wp(
    sand_pct: float,
    clay_pct: float,
    om_pct: float,
) -> tuple[float, float]:
    """Estimate FC and WP from texture using simplified Rawls & Brakensiek.

    These are approximate — better than nothing when only texture is known.
    For production use, consider the Rosetta Python package.

    Parameters
    ----------
    sand_pct, clay_pct : sand and clay percentages (0-100)
    om_pct : organic matter percentage

    Returns
    -------
    (theta_fc, theta_wp) in cm³/cm³
    """
    # Simplified Saxton & Rawls (2006) — widely used, reasonably accurate
    s = sand_pct / 100
    c = clay_pct / 100
    om = om_pct / 100

    # Wilting point (1500 kPa)
    wp_1 = (
        -0.024 * s
        + 0.487 * c
        + 0.006 * om
        + 0.005 * s * om
        - 0.013 * c * om
        + 0.068 * s * c
        + 0.031
    )
    theta_wp = wp_1 + 0.14 * wp_1 - 0.02

    # Field capacity (33 kPa)
    fc_1 = (
        -0.251 * s
        + 0.195 * c
        + 0.011 * om
        + 0.006 * s * om
        - 0.027 * c * om
        + 0.452 * s * c
        + 0.299
    )
    theta_fc = fc_1 + 1.283 * fc_1 * fc_1 - 0.374 * fc_1 - 0.015

    # Clamp to physical range
    theta_fc = max(0.05, min(0.55, theta_fc))
    theta_wp = max(0.02, min(0.35, theta_wp))
    theta_wp = min(theta_wp, theta_fc - 0.02)

    return (theta_fc, theta_wp)


# ---------------------------------------------------------------------------
# Extraction from gridded data
# ---------------------------------------------------------------------------


def _load_soil_netcdf(cfg: RegionConfig) -> xr.Dataset:
    """Load processed soil_hydro.nc."""
    path = cfg.processed_dir / "static" / "soil_hydro.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed soil data not found: {path}\n"
            f"Run 'irrigator build-static' or the Block 0 notebook first."
        )
    return xr.open_dataset(path)


def _parcel_to_l93(parcel: ParcelConfig) -> tuple[float, float]:
    """Convert parcel WGS84 coordinates to Lambert-93."""
    x, y = _TO_L93.transform(parcel.lon, parcel.lat)
    return x, y


def _extract_grid_values(
    ds: xr.Dataset,
    x: float,
    y: float,
    variable: str,
) -> np.ndarray:
    """Extract values at (x, y) from a gridded dataset using nearest neighbor."""
    da = ds[variable]
    if "depth" in da.dims:
        return da.sel(x=x, y=y, method="nearest").values
    else:
        return np.array([float(da.sel(x=x, y=y, method="nearest"))])


def extract_soil_from_grid(
    cfg: RegionConfig,
    parcel: ParcelConfig,
) -> SoilProfile:
    """Extract soil profile from EU-SoilHydroGrids at parcel location."""
    ds = _load_soil_netcdf(cfg)
    x, y = _parcel_to_l93(parcel)

    fc = _extract_grid_values(ds, x, y, "fc")
    wp = _extract_grid_values(ds, x, y, "wp")
    ks = _extract_grid_values(ds, x, y, "ks")
    total_awc = float(ds["total_awc_mm"].sel(x=x, y=y, method="nearest"))

    # Reconstruct depth layers from the depth coordinate
    depths = ds["depth"].values  # midpoints
    # Standard EU-SoilHydroGrids layers
    layer_bounds = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]
    # Match to available depth levels
    z_layers = layer_bounds[: len(fc)]

    ds.close()

    return SoilProfile(
        theta_fc=fc.tolist(),
        theta_wp=wp.tolist(),
        k_sat=ks.tolist(),
        z_layers_cm=z_layers,
        total_awc_mm=total_awc,
        max_root_depth_m=2.0,  # will be constrained by crop later
        source="eu_soilhydrogrids",
    )


# ---------------------------------------------------------------------------
# Farmer overrides
# ---------------------------------------------------------------------------


def _build_from_farmer_analysis(parcel: ParcelConfig) -> SoilProfile:
    """Build SoilProfile from farmer-provided soil analysis."""
    soil_cfg = parcel.soil
    layers = soil_cfg.get("layers", [])

    if not layers:
        raise ValueError(
            f"Parcel {parcel.id}: soil source is 'farmer_analysis' "
            f"but no layers provided in config."
        )

    theta_fc_list = []
    theta_wp_list = []
    k_sat_list = []
    z_layers = []

    for layer in layers:
        depth = layer["depth_cm"]
        z_layers.append(tuple(depth))

        # If FC/WP given directly, use them
        if "theta_fc" in layer and "theta_wp" in layer:
            theta_fc_list.append(layer["theta_fc"])
            theta_wp_list.append(layer["theta_wp"])
        else:
            # Derive from texture via pedotransfer
            fc, wp = _pedotransfer_fc_wp(
                layer["sand_pct"],
                layer["clay_pct"],
                layer.get("organic_matter_pct", 1.5),
            )
            theta_fc_list.append(fc)
            theta_wp_list.append(wp)
            logger.info(
                "Parcel %s layer %s: PTF → FC=%.3f, WP=%.3f",
                parcel.id,
                depth,
                fc,
                wp,
            )

        # K_sat from PTF if not given (very rough)
        k_sat_list.append(layer.get("k_sat", 50.0))  # default 50 cm/day

    # Compute total AWC
    total_awc = 0.0
    for i, (zt, zb) in enumerate(z_layers):
        total_awc += max(0, theta_fc_list[i] - theta_wp_list[i]) * (zb - zt) * 10

    return SoilProfile(
        theta_fc=theta_fc_list,
        theta_wp=theta_wp_list,
        k_sat=k_sat_list,
        z_layers_cm=z_layers,
        total_awc_mm=total_awc,
        max_root_depth_m=z_layers[-1][1] / 100,
        source="farmer_analysis",
    )


def _build_from_probe(parcel: ParcelConfig, grid_profile: SoilProfile) -> SoilProfile:
    """Override grid FC/WP with probe-measured values."""
    soil_cfg = parcel.soil
    fc_measured = soil_cfg.get("theta_fc_measured")
    wp_estimated = soil_cfg.get("theta_wp_estimated")

    if fc_measured is None:
        raise ValueError(
            f"Parcel {parcel.id}: source is 'probe_calibrated' but theta_fc_measured not provided."
        )

    # Scale grid profile layers proportionally to match measured FC
    grid_mean_fc = np.mean(grid_profile.theta_fc)
    scale_fc = fc_measured / grid_mean_fc if grid_mean_fc > 0 else 1.0

    new_fc = [min(0.55, fc * scale_fc) for fc in grid_profile.theta_fc]

    if wp_estimated is not None:
        grid_mean_wp = np.mean(grid_profile.theta_wp)
        scale_wp = wp_estimated / grid_mean_wp if grid_mean_wp > 0 else 1.0
        new_wp = [max(0.02, wp * scale_wp) for wp in grid_profile.theta_wp]
    else:
        new_wp = grid_profile.theta_wp

    total_awc = 0.0
    for i, (zt, zb) in enumerate(grid_profile.z_layers_cm):
        total_awc += max(0, new_fc[i] - new_wp[i]) * (zb - zt) * 10

    logger.info(
        "Parcel %s: probe override FC=%.3f (grid was %.3f), AWC=%.0f mm",
        parcel.id,
        fc_measured,
        grid_mean_fc,
        total_awc,
    )

    return SoilProfile(
        theta_fc=new_fc,
        theta_wp=new_wp,
        k_sat=grid_profile.k_sat,
        z_layers_cm=grid_profile.z_layers_cm,
        total_awc_mm=total_awc,
        max_root_depth_m=grid_profile.max_root_depth_m,
        source="probe_calibrated",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def get_soil_profile(
    cfg: RegionConfig,
    parcel: ParcelConfig,
) -> SoilProfile:
    """Get soil profile for a parcel, applying farmer overrides if configured.

    Priority:
    1. farmer_analysis → build from texture/direct values
    2. probe_calibrated → scale grid values to match probe measurement
    3. eu_soilhydrogrids → extract from grid (default)
    """
    source = parcel.soil.get("source", "eu_soilhydrogrids")

    if source == "farmer_analysis":
        profile = _build_from_farmer_analysis(parcel)
        logger.info(
            "Parcel %s: using farmer analysis — AWC=%.0f mm",
            parcel.id,
            profile.total_awc_mm,
        )
        return profile

    # Grid extraction needed for both default and probe override
    grid_profile = extract_soil_from_grid(cfg, parcel)

    if source == "probe_calibrated":
        profile = _build_from_probe(parcel, grid_profile)
        return profile

    logger.info(
        "Parcel %s: using EU-SoilHydroGrids — AWC=%.0f mm, FC=[%s]",
        parcel.id,
        grid_profile.total_awc_mm,
        ", ".join(f"{v:.3f}" for v in grid_profile.theta_fc),
    )
    return grid_profile
