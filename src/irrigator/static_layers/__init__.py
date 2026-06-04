"""IrriGator Block 1 - Static layer extraction at parcel level.

Loads processed gridded data (soil_hydro.nc, terrain.nc from Block 0)
and extracts parcel-level parameters with farmer override support.

Typical usage::

    from irrigator.config import load_region_config, load_parcel_config
    from irrigator.static_layers.soil import get_soil_profile
    from irrigator.static_layers.terrain import get_terrain_params

    cfg = load_region_config("configs/dordogne.yaml")
    parcel = load_parcel_config("configs/parcels/example.yaml")

    soil = get_soil_profile(cfg, parcel)
    terrain = get_terrain_params(cfg, parcel)
"""

from irrigator.static_layers.soil import SoilProfile, get_soil_profile
from irrigator.static_layers.terrain import TerrainParams, get_terrain_params

__all__ = ["SoilProfile", "TerrainParams", "get_soil_profile", "get_terrain_params"]
