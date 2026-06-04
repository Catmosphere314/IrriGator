"""Terrain-based solar radiation correction.

ERA5-Land surface solar radiation assumes a smooth orography.  On sloped
terrain, the actual radiation differs due to:
1. Slope and aspect affecting the angle of incidence
2. Horizon shading (not modelled here — would require ray tracing on DEM)

For most irrigated maize parcels in Dordogne (valley floors, gentle slopes),
this correction is small (<5%).  It matters more for hillside parcels.

The correction factor is computed as the ratio of radiation on a tilted
surface to radiation on a horizontal surface, averaged over the day.
"""

from __future__ import annotations

import logging

import numpy as np

from irrigator.static_layers.terrain import TerrainParams

logger = logging.getLogger(__name__)


def _solar_declination(day_of_year: int) -> float:
    """Solar declination angle [radians]."""
    return 0.4093 * np.sin(2 * np.pi / 365 * day_of_year - 1.405)


def _sunset_hour_angle(lat_rad: float, decl: float) -> float:
    """Sunset hour angle [radians]."""
    x = -np.tan(lat_rad) * np.tan(decl)
    x = np.clip(x, -1.0, 1.0)
    return np.arccos(x)


def _extraterrestrial_radiation(day_of_year: int, lat_rad: float) -> float:
    """Daily extraterrestrial radiation Ra [MJ/m²/day] on horizontal surface.

    FAO-56 equation 21.
    """
    Gsc = 0.0820  # solar constant [MJ/m²/min]
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * day_of_year)  # inverse relative distance
    decl = _solar_declination(day_of_year)
    ws = _sunset_hour_angle(lat_rad, decl)

    Ra = (
        (24 * 60 / np.pi)
        * Gsc
        * dr
        * (ws * np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.sin(ws))
    )
    return max(0, Ra)


def slope_radiation_factor(
    day_of_year: int,
    latitude_deg: float,
    slope_deg: float,
    aspect_deg: float,
) -> float:
    """Compute daily radiation correction factor for a sloped surface.

    Returns ratio: Rs_slope / Rs_horizontal.
    Values > 1 mean the slope receives more radiation than flat ground.

    Parameters
    ----------
    day_of_year : 1-366
    latitude_deg : latitude in degrees
    slope_deg : slope angle in degrees
    aspect_deg : aspect in degrees (0=N, 90=E, 180=S, 270=W)

    Returns
    -------
    Correction factor (typically 0.8–1.2 for moderate slopes).
    """
    if slope_deg < 0.5:
        return 1.0

    lat_rad = np.radians(latitude_deg)
    slope_rad = np.radians(slope_deg)
    # Convert aspect: 0=N, 90=E → azimuth from south: 0=S, -90=E, 90=W
    aspect_from_south = np.radians(aspect_deg - 180)

    decl = _solar_declination(day_of_year)

    # Effective latitude on a slope (Liu & Jordan approximation)
    # For a south-facing slope, effective latitude decreases by slope angle
    lat_eff = lat_rad - slope_rad * np.cos(aspect_from_south)

    Ra_flat = _extraterrestrial_radiation(day_of_year, lat_rad)
    Ra_slope = _extraterrestrial_radiation(day_of_year, lat_eff)

    if Ra_flat <= 0:
        return 1.0

    factor = Ra_slope / Ra_flat
    # Clamp to reasonable range
    factor = np.clip(factor, 0.5, 1.5)

    return float(factor)


def correct_radiation(
    rs_mj: float,
    day_of_year: int,
    latitude_deg: float,
    terrain: TerrainParams,
) -> float:
    """Apply terrain correction to daily solar radiation.

    Parameters
    ----------
    rs_mj : ERA5-Land surface solar radiation [MJ/m²/day]
    day_of_year : 1-366
    latitude_deg : parcel latitude [degrees]
    terrain : TerrainParams with slope and aspect

    Returns
    -------
    Corrected radiation [MJ/m²/day].
    """
    if np.isnan(terrain.aspect_deg):
        # Flat terrain — no correction
        return rs_mj

    factor = slope_radiation_factor(
        day_of_year,
        latitude_deg,
        terrain.slope_deg,
        terrain.aspect_deg,
    )

    return rs_mj * factor
