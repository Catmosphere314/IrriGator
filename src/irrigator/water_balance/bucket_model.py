"""Soil water balance: single-layer FAO-56 bucket.

Daily update:
```
W(t) = W(t-1) + P(t) + I(t) − ETc(t) − D(t) − R(t)
```

Where:
- `W(t)` = root zone soil water content [mm]
- `P(t)` = precipitation [mm] (from Block 2)
- `I(t)` = irrigation [mm] (from farmer input or decision layer)
- `ETc(t)` = crop ET = Kc × ET0 (Kc from Block 4, ET0 from this block)
- `D(t)` = deep drainage: max(0, W(t-1) + P + I − ETc − θ_FC × Z_r) — water above field capacity drains
- `R(t)` = runoff. Start with a simple threshold (SCS curve number or fraction of excess rainfall). Refine later if needed.

Stress coefficient:
```
Ks = (W − θ_WP × Zr) / ((1 − p) × TAW)   when W < (1-p) × TAW + θ_WP × Zr
Ks = 1.0                                     otherwise
```
Where TAW = (θ_FC − θ_WP) × Z_r, and p ≈ 0.55 for maize (FAO-56 Table 22). Under stress, ETc_adj = Ks × Kc × ET0.
"""

import pandas as pd
import numpy as np

from irrigator.static_layers import TerrainParams
from irrigator.atmospheric import DailyForcing
from irrigator.water_balance import compute_et0
from irrigator.config import ParcelConfig
from irrigator.static_layers import SoilProfile

def compute_water_balance(parcel : ParcelConfig, forcing : DailyForcing, terrain : TerrainParams, static : SoilProfile, Kc : float, runoff : float,
                          W_prev : WaterBalanceState, p : float = 0.55):
    """Compute the single-layer FAO-56 bucket soil water balance.
    
    Parameters
    ----------
    config     :
    forcing    :
    terrain    :
    static     :
    k_c        :
    runoff     :
    W_prev        : Previous water balance state

    Returns
    ----------
    WaterBalanceState object
    """

    # Compute the ET0 from Penman-Monteith
    et0 = compute_et0(daily_forcing=forcing, terrain=terrain, lat_deg=parcel.lat)

    # compute new water balance
    w_t = W_prev + forcing.precip_mm - W_prev.etc - max(0, W_prev.soil_water + forcing.precip_mm - W_prev.etc - static.theta_fc ) # No I(t) yet and what Z_r is?

    # compute stress params
    Taw = (static.theta_fc - static.theta_wp) * Z_r # Z_r ???
    Ks = (w_t - static.theta_wp * 0) / ((1  - p ) * Taw) #What Z_r again? what TAW?

    ETc_adj = Ks * Kc * et0

    return WaterBalanceState(
            date = ??,
            et0 = et0,            # reference ET [mm]
            etc = ETc_adj,            # crop ET (adjusted for stress) [mm]
            precip = forcing.precip_mm,         # effective precipitation [mm]
            irrigation = ??,     # applied irrigation [mm]
            drainage = ??,       # deep percolation [mm]
            runoff = ,         # surface runoff [mm]
            soil_water = w_t,     # root zone water content [mm]
            depletion = ??,     # current depletion [mm]
            stress_coeff = Ks   # Ks [0-1]
            fraction_awc = w_t/Taw   # W as fraction of TAW [0-1]
            )


