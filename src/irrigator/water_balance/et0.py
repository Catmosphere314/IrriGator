"""Penman Monteith model using pyet.

Ref:pyet
"""

import pyet
import numpy as np

from irrigator.atmospheric import DailyForcing
from irrigator.static_layers import TerrainParams


def compute_et0(daily_forcing: DailyForcing, terrain: TerrainParams, lat_deg : float):
    """Compute the FAO-56 Penman-Monteith reference ET.
    
    Parameters
    ---------
    daily_forcing : DailyForcing from Block 2 (Atmospheric)
    terrain       : TerrainParams from Block 1 (Static)
    lat_deg       : parcel latitude in degrees (from parcel config)

    Returns
    ---------
    ET0 in mm/day, same length as forcing.dates
    """

    # Actual vapor pressure from dewpoint (FAO-56 eq. 14)
    # ea = 0.6108 × exp(17.27 × Tdew / (Tdew + 237.3))
    ea = 0.6108 * np.exp(17.27 * daily_forcing.dewpoint / (daily_forcing.dewpoint + 237.3))

    # Latitude in radians (pyet expects radians)
    lat_rad = np.radians(lat_deg)

    et0 = pyet.pm_fao56(
        tmean=daily_forcing.t_mean,
        wind=daily_forcing.wind_speed_2m,
        rs=daily_forcing.rs_mj,
        rn=None,
        ea=ea,
        lat=lat_rad,
        pressure=daily_forcing.pressure_kpa,
        tmax=daily_forcing.t_max,
        tmin=daily_forcing.t_min,
        elevation=terrain.elevation_m,
    )

    return et0
