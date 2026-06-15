"""IrriGator Block 2 — Atmospheric forcing at parcel level.

Processes ERA5-Land reanalysis into downscaled daily forcing suitable
for FAO-56 Penman-Monteith ET0 and soil water balance.

Typical workflow::

    from irrigator.config import load_region_config, load_parcel_config
    from irrigator.static_layers import get_soil_profile, get_terrain_params
    from irrigator.atmospheric.era5_processor import load_daily
    from irrigator.atmospheric.forcing import extract_parcel_forcing

    cfg = load_region_config("configs/dordogne.yaml")
    parcel = load_parcel_config("configs/parcels/example.yaml")
    terrain = get_terrain_params(cfg, parcel)
    era5_daily = load_daily()  # uses default data/processed path

    forcing = extract_parcel_forcing(era5_daily, parcel, terrain)
    df = forcing.to_dataframe()
"""

from irrigator.atmospheric.forcing import DailyForcing, extract_parcel_forcing

__all__ = ["DailyForcing", "extract_parcel_forcing"]
