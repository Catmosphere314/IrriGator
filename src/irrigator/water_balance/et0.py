"""Penman Monteith model using pyet.

Ref:pyet
"""

import numpy as np
import pandas as pd
import pyet

from irrigator.atmospheric import DailyForcing
from irrigator.static_layers import TerrainParams


def compute_et0(daily_forcing: DailyForcing, terrain: TerrainParams, lat_deg: float):
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
    # pyet expects pd.Series with DatetimeIndex
    index = pd.DatetimeIndex(daily_forcing.dates)

    # Actual vapor pressure from dewpoint (FAO-56 eq. 14)
    # ea = 0.6108 × exp(17.27 × Tdew / (Tdew + 237.3))
    ea = 0.6108 * np.exp(17.27 * daily_forcing.dewpoint / (daily_forcing.dewpoint + 237.3))

    # Latitude in radians (pyet expects radians)
    lat_rad = np.radians(lat_deg)

    et0 = pyet.pm_fao56(
        tmean=pd.Series(daily_forcing.t_mean, index=index),
        wind=pd.Series(daily_forcing.wind_speed_2m, index=index),
        rs=pd.Series(daily_forcing.rs_mj, index=index),
        rn=None,
        ea=pd.Series(ea, index=index),
        lat=lat_rad,
        pressure=pd.Series(daily_forcing.pressure_kpa, index=index),
        tmax=pd.Series(daily_forcing.t_max, index=index),
        tmin=pd.Series(daily_forcing.t_min, index=index),
        elevation=terrain.elevation_m,
    )

    return et0
