"""Framework for computing Reference ET and daily soil water balance.

It relies on the FAO-56 Penman-Monteith model using pyet.

"""

#TO CLEAN 
from irrigator.water_balance.et0 import compute_et0
from irrigator.water_balance.state import WaterBalanceState

__all__ = ["compute_et0", "WaterBalanceState"]
