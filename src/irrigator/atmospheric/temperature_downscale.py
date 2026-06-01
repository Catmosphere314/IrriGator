"""Temperature downscaling via elevation lapse-rate correction.

ERA5-Land operates on a ~9 km grid with a smoothed orography.  The actual
elevation at a specific parcel can differ by tens to hundreds of meters
from the ERA5-Land grid cell elevation.  This module corrects temperature
using the standard environmental lapse rate:

    T_local = T_ERA5 + Γ × (z_ERA5 − z_local)

where Γ ≈ −6.5 °C/km (temperature decreases with altitude).

In Dordogne, elevation differences between ERA5-Land cells and actual
parcels are typically 50–150 m, giving corrections of 0.3–1.0 °C.
Small but meaningful for GDD accumulation over a growing season.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from irrigator.static_layers.terrain import TerrainParams

logger = logging.getLogger(__name__)

# Standard environmental lapse rate [°C/m]
# Negative: temperature decreases with altitude
DEFAULT_LAPSE_RATE = -6.5 / 1000  # °C per meter


def correct_temperature(
    t_era5: float | np.ndarray,
    z_era5: float,
    z_local: float,
    lapse_rate: float = DEFAULT_LAPSE_RATE,
) -> float | np.ndarray:
    """Apply lapse-rate correction to temperature.

    Parameters
    ----------
    t_era5 : temperature from ERA5-Land [°C]
    z_era5 : elevation of the ERA5-Land grid cell [m]
    z_local : actual elevation at the parcel [m]
    lapse_rate : lapse rate [°C/m], default −6.5 °C/km

    Returns
    -------
    Corrected temperature [°C].
    """
    dz = z_era5 - z_local  # positive if ERA5 cell is higher
    # Lapse rate: dT/dz = Γ = −6.5 °C/km
    # If parcel is 100m LOWER than ERA5 cell: (z_local - z_ERA5) = -100
    #   correction = Γ × (−100) = (−0.0065)(−100) = +0.65°C → warmer. Correct.
    # If parcel is 100m HIGHER: correction = −0.65°C → cooler. Correct.
    return t_era5 + lapse_rate * (z_local - z_era5)


def downscale_daily_temperature(
    ds_daily: xr.Dataset,
    terrain: TerrainParams,
    lapse_rate: float = DEFAULT_LAPSE_RATE,
) -> xr.Dataset:
    """Apply lapse-rate correction to all daily temperature fields.

    Modifies t_min, t_max, t_mean, dewpoint in place.

    Parameters
    ----------
    ds_daily : daily ERA5-Land dataset at the ERA5 grid cell
    terrain : TerrainParams with both parcel and ERA5 cell elevations
    lapse_rate : °C/m, default −6.5 °C/km

    Returns
    -------
    Dataset with corrected temperature fields.
    """
    z_era5 = terrain.era5_elevation_m
    z_local = terrain.elevation_m

    if z_era5 is None:
        logger.warning("ERA5 cell elevation unknown — skipping lapse correction")
        return ds_daily

    dz = z_local - z_era5
    correction = lapse_rate * dz

    if abs(correction) < 0.01:
        logger.debug("Lapse correction < 0.01°C — skipping")
        return ds_daily

    logger.info(
        "Lapse-rate correction: Δz=%.0f m, ΔT=%.2f°C (Γ=%.1f °C/km)",
        dz,
        correction,
        lapse_rate * 1000,
    )

    result = ds_daily.copy()
    for var in ("t_min", "t_max", "t_mean", "dewpoint"):
        if var in result:
            result[var] = result[var] + correction

    return result
