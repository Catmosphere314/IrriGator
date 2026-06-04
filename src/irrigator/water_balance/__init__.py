"""IrriGator Block 3 — Soil water balance.

Coupled daily simulation of crop growth (Block 4) and FAO-56 bucket
model soil water balance.

The simulation loop in ``bucket_model.run_simulation`` is the main
entry point.  It advances both the crop state and water balance daily.

Typical usage::

    from irrigator.water_balance.bucket_model import run_simulation, states_to_dataframe
    from irrigator.crop.phenology import CropParams

    crop_params = CropParams.from_config(parcel.crop)
    states = run_simulation(forcing, parcel, terrain, soil, crop_params)
    df = states_to_dataframe(states)
"""

from irrigator.water_balance.et0 import compute_et0
from irrigator.water_balance.state import WaterBalanceState

__all__ = ["WaterBalanceState", "compute_et0"]
