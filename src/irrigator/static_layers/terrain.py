"""Parcel-level terrain parameter extraction.

Loads processed terrain.nc (from Block 0) and extracts elevation, slope,
and aspect at each parcel location.  These feed into Block 2 for
temperature lapse-rate correction and radiation adjustment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import xarray as xr
from pyproj import Transformer

from irrigator.config import ParcelConfig, RegionConfig

logger = logging.getLogger(__name__)

_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)


@dataclass
class TerrainParams:
    """Terrain parameters for one parcel.

    Attributes
    ----------
    elevation_m : mean elevation of the grid cell [m]
    slope_deg : mean slope [degrees]
    aspect_deg : mean aspect [degrees, 0=N, 90=E, 180=S, 270=W], NaN if flat
    era5_elevation_m : elevation of the ERA5-Land grid cell covering this parcel [m]
                       (needed for lapse-rate correction in Block 2)
    """

    elevation_m: float
    slope_deg: float
    aspect_deg: float
    era5_elevation_m: float | None = None


def _load_terrain_netcdf(cfg: RegionConfig) -> xr.Dataset:
    """Load processed terrain.nc."""
    path = cfg.processed_dir / "static" / "terrain.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed terrain data not found: {path}\n"
            f"Run 'irrigator build-static' or the Block 0 notebook first."
        )
    return xr.open_dataset(path)


def get_terrain_params(
    cfg: RegionConfig,
    parcel: ParcelConfig,
) -> TerrainParams:
    """Extract terrain parameters at parcel location.

    Parameters
    ----------
    cfg : RegionConfig
    parcel : ParcelConfig with lat/lon

    Returns
    -------
    TerrainParams for the parcel.
    """
    ds = _load_terrain_netcdf(cfg)
    x, y = _TO_L93.transform(parcel.lon, parcel.lat)

    elevation = float(ds["elevation"].sel(x=x, y=y, method="nearest"))
    slope = float(ds["slope"].sel(x=x, y=y, method="nearest"))
    aspect = float(ds["aspect"].sel(x=x, y=y, method="nearest"))

    ds.close()

    logger.info(
        "Parcel %s terrain: elev=%.0f m, slope=%.1f°, aspect=%.0f°",
        parcel.id,
        elevation,
        slope,
        aspect,
    )

    return TerrainParams(
        elevation_m=elevation,
        slope_deg=slope,
        aspect_deg=aspect,
    )


def get_era5_elevation(cfg: RegionConfig, parcel: ParcelConfig) -> float:
    """Get ERA5-Land grid cell elevation at the parcel location.

    ERA5-Land provides its own orography field.  The difference between
    this and the high-resolution DEM elevation is what drives the
    temperature lapse-rate correction in Block 2.
    """
    from irrigator.ingestion.cds_client import open_era5_land

    try:
        ds = open_era5_land()
        # ERA5-Land geopotential / orography is typically not in the standard
        # variable set.  Use the mean elevation from the terrain DEM at ERA5
        # resolution as an approximation: average the DEM over the ~9km cell.
        # This is computed in era5_processor.py during grid setup.
        ds.close()
    except Exception:
        pass

    # Fallback: estimate ERA5 cell elevation from the terrain grid
    # by averaging over a ~9km window around the parcel
    terrain_ds = _load_terrain_netcdf(cfg)
    x, y = _TO_L93.transform(parcel.lon, parcel.lat)

    # ERA5-Land is ~9km ≈ 9000m, so ±4500m window
    half_window = 4500
    elev = terrain_ds["elevation"]
    cell = elev.sel(
        x=slice(x - half_window, x + half_window),
        y=slice(y + half_window, y - half_window),  # y is top-to-bottom
    )
    era5_elev = float(cell.mean())
    terrain_ds.close()

    logger.debug(
        "ERA5 cell elevation at parcel %s: %.0f m (parcel DEM: %.0f m)",
        parcel.id,
        era5_elev,
        float(elev.sel(x=x, y=y, method="nearest")),
    )
    return era5_elev
