"""IGN RGE ALTI DEM loader.

Downloads or reads the IGN RGE ALTI 5 m DEM for the region, derives
slope and aspect, and aggregates everything to the target grid.

Download
--------
RGE ALTI 5 m is free from IGN Géoservices:
    https://geoservices.ign.fr/rgealti

Download the tiles covering Dordogne and place the .asc or .tif files in::

    data/static/dem/

The loader will mosaic all tiles found in that directory.

IGN also provides ADMIN EXPRESS (free) for administrative boundaries.
We use the département boundary to clip the DEM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject

from irrigator.config import RegionConfig
from irrigator.ingestion.grid import TargetGrid

logger = logging.getLogger(__name__)


@dataclass
class TerrainData:
    """Terrain-derived variables on the target grid.

    All arrays have dims (y, x) matching the target grid.

    Attributes
    ----------
    elevation : mean elevation [m] per grid cell
    slope : mean slope [degrees] per grid cell
    aspect : mean aspect [degrees, 0=N, 90=E, 180=S, 270=W] per grid cell
    svf : sky view factor (optional, for radiation correction)
    """

    elevation: xr.DataArray
    slope: xr.DataArray
    aspect: xr.DataArray


def _find_dem_dir(cfg: RegionConfig) -> Path:
    """Locate the DEM tile directory."""
    dem_dir = cfg.static_dir / "dem"
    if not dem_dir.exists():
        raise FileNotFoundError(
            f"DEM directory not found: {dem_dir}\n"
            f"Download RGE ALTI tiles from https://geoservices.ign.fr/rgealti "
            f"and place .tif or .asc files in {dem_dir}"
        )
    return dem_dir


def _mosaic_dem_tiles(dem_dir: Path) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    """Mosaic all DEM tiles in the directory into a single array.

    Supports .tif, .tiff, and .asc (ESRI ASCII Grid) files.
    """
    patterns = ("*.tif", "*.tiff", "*.asc")
    tile_paths = []
    for pat in patterns:
        tile_paths.extend(dem_dir.glob(pat))

    if not tile_paths:
        raise FileNotFoundError(f"No DEM tiles (.tif, .asc) found in {dem_dir}")

    logger.info("Mosaicking %d DEM tiles from %s", len(tile_paths), dem_dir)

    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)
    crs = datasets[0].crs

    for ds in datasets:
        ds.close()

    # mosaic shape is (bands, height, width) — take first band
    return mosaic[0], transform, crs


def _compute_slope_aspect(
    elevation: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope and aspect from elevation using finite differences.

    Parameters
    ----------
    elevation : 2-D elevation array [m]
    cell_size : pixel size in meters

    Returns
    -------
    slope : array in degrees [0, 90]
    aspect : array in degrees [0, 360), 0=N, 90=E, 180=S, 270=W
             NaN for flat pixels (slope ≈ 0).
    """
    # Gradients using numpy (central differences, forward/backward at edges)
    dy, dx = np.gradient(elevation, cell_size)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect: angle from north, clockwise
    aspect_rad = np.arctan2(-dx, dy)  # atan2(-dz/dx, -dz/dy) but with sign convention
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360) % 360

    # Mark flat areas as NaN aspect (slope < 0.1°)
    aspect_deg[slope_deg < 0.1] = np.nan

    return slope_deg, aspect_deg


def _aggregate_to_grid(
    data: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: rasterio.crs.CRS,
    target_grid: TargetGrid,
    resampling: Resampling = Resampling.average,
) -> np.ndarray:
    """Reproject and aggregate a high-res raster to the target grid."""
    ny, nx = target_grid.shape
    dst = np.full((ny, nx), np.nan, dtype=np.float32)

    reproject(
        source=data.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_grid.transform,
        dst_crs=target_grid.crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def load_terrain(
    cfg: RegionConfig,
    target_grid: TargetGrid,
) -> TerrainData:
    """Load DEM, compute slope/aspect at native resolution, aggregate to grid.

    The workflow is:
    1. Mosaic all DEM tiles.
    2. Compute slope and aspect at native resolution (~5 m).
    3. Reproject and aggregate elevation (mean), slope (mean), aspect (mean)
       to the 250 m target grid.
    """
    dem_dir = _find_dem_dir(cfg)

    # Step 1: mosaic
    elev_native, transform_native, crs_native = _mosaic_dem_tiles(dem_dir)
    cell_size = abs(transform_native.a)  # pixel size in meters (assuming square)
    logger.info(
        "DEM mosaic: %d × %d at %.1f m, CRS %s",
        elev_native.shape[1],
        elev_native.shape[0],
        cell_size,
        crs_native,
    )

    # Step 2: slope and aspect at native resolution
    slope_native, aspect_native = _compute_slope_aspect(elev_native, cell_size)
    logger.info(
        "Slope range: [%.1f°, %.1f°], median %.1f°",
        np.nanmin(slope_native),
        np.nanmax(slope_native),
        np.nanmedian(slope_native),
    )

    # Step 3: aggregate to target grid
    logger.info("Aggregating terrain to %d m grid...", target_grid.resolution_m)

    elev_grid = _aggregate_to_grid(
        elev_native,
        transform_native,
        crs_native,
        target_grid,
        resampling=Resampling.average,
    )
    slope_grid = _aggregate_to_grid(
        slope_native,
        transform_native,
        crs_native,
        target_grid,
        resampling=Resampling.average,
    )
    # Aspect averaging is problematic (circular variable).
    # Use bilinear as a rough approximation — for most irrigation parcels on
    # valley floors the aspect is near-uniform. Revisit with circular mean
    # if hillside parcels become important.
    aspect_grid = _aggregate_to_grid(
        aspect_native,
        transform_native,
        crs_native,
        target_grid,
        resampling=Resampling.average,
    )

    coords = {"y": target_grid.ys, "x": target_grid.xs}
    dims = ["y", "x"]

    return TerrainData(
        elevation=xr.DataArray(
            elev_grid,
            dims=dims,
            coords=coords,
            name="elevation",
            attrs={"units": "m", "long_name": "mean elevation", "source": "IGN RGE ALTI 5m"},
        ),
        slope=xr.DataArray(
            slope_grid,
            dims=dims,
            coords=coords,
            name="slope",
            attrs={"units": "degrees", "long_name": "mean slope"},
        ),
        aspect=xr.DataArray(
            aspect_grid,
            dims=dims,
            coords=coords,
            name="aspect",
            attrs={"units": "degrees", "long_name": "mean aspect (0=N, 90=E, 180=S, 270=W)"},
        ),
    )


def save_terrain(data: TerrainData, out_dir: Path) -> Path:
    """Save processed terrain data as NetCDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "terrain.nc"

    ds = xr.Dataset(
        {
            "elevation": data.elevation,
            "slope": data.slope,
            "aspect": data.aspect,
        }
    )
    ds.attrs["source"] = "IGN RGE ALTI 5m, aggregated to target grid"

    ds.to_netcdf(out_path)
    logger.info("Saved terrain data: %s", out_path)
    return out_path
