"""WaterBalanceState dataclass."""
import datetime as dt


class WaterBalanceState:
    date: dt.date
    et0: float  # reference ET [mm]
    etc: float  # crop ET (adjusted for stress) [mm]
    precip: float  # effective precipitation [mm]
    irrigation: float  # applied irrigation [mm]
    drainage: float  # deep percolation [mm]
    runoff: float  # surface runoff [mm]
    soil_water: float  # root zone water content [mm]
    depletion: float  # current depletion [mm]
    stress_coeff: float  # Ks [0-1]
    fraction_awc: float  # W as fraction of TAW [0-1]