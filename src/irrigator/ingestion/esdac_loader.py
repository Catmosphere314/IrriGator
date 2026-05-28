"""EU-SoilHydroGrids loader — tiled structure.

EU-SoilHydroGrids (Tóth et al., 2017) provides pre-computed soil hydraulic
properties at 250 m resolution across Europe, derived from SoilGrids texture
maps via pedotransfer functions.

Data structure
--------------
The dataset is distributed as ~1300 tile folders, each containing 42 GeoTIFF
files (6 variables × 7 depth layers).  File naming convention::

    {grid_id}/
        FC_M_sl1_{grid_id}.tif     # field capacity, median, layer 1
        FC_M_sl2_{grid_id}.tif
        ...
        WP_M_sl1_{grid_id}.tif     # wilting point
        ...
        KS_M_sl1_{grid_id}.tif     # saturated hydraulic conductivity
        ...
        THS_M_sl1_{grid_id}.tif    # saturated water content
        ...

Download
--------
1. Register (free) and download from ESDAC:
       https://esdac.jrc.ec.europa.eu/content/3d-soil-hydraulic-database-europe-1-km-and-250-m-resolution
   Extract into ``data/static/<region>/eu_soilhydrogrids/``

2. Download the tile navigation GeoPackage from:
       https://github.com/LandscapeGeoinformatics/EU-SoilHydroGrids_tiles_nav
   Place ``grid_cells_250m_wgs84.gpkg`` (or .shp) in the same directory.
   This spatial index tells the loader which tile folders cover your region
   without scanning all 1300 folders.

Units
-----
- FC, WP, THS: cm³/cm³ (volumetric water content)
- KS: cm/day (saturated hydraulic conductivity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

from irrigator.config import RegionConfig
from irrigator.ingestion.grid import TargetGrid

logger = logging.getLogger(__name__)

# Variables we need for the water balance
SOIL_VARIABLES = ("FC", "WP", "KS")

# Depth layers in EU-SoilHydroGrids (sl1..sl7)
# sl7 (200-x cm) often has sparse coverage; we use sl1-sl6 by default
DEPTH_LAYERS = {
    "sl1": (0, 5),
    "sl2": (5, 15),
    "sl3": (15, 30),
    "sl4": (30, 60),
    "sl5": (60, 100),
    "sl6": (100, 200),
}

# Filename pattern inside each tile folder
# {var}_M_{layer}_{grid_id}.tif  where M = Median estimate
_TILE_FILENAME = "{var}_M_{layer}_{grid_id}.tif"


@dataclass
class SoilHydroData:
    """Soil hydraulic properties on the target grid.

    Attributes
    ----------
    fc : xr.DataArray — field capacity [cm³/cm³], dims (depth, y, x)
    wp : xr.DataArray — wilting point [cm³/cm³], dims (depth, y, x)
    ks : xr.DataArray — saturated hydraulic conductivity [cm/day], dims (depth, y, x)
    awc : xr.DataArray — available water capacity FC-WP [cm³/cm³], dims (depth, y, x)
    total_awc_mm : xr.DataArray — depth-integrated AWC [mm], dims (y, x)
    depths_cm : list of (top, bottom) tuples per layer
    tile_ids : list of grid_id strings used
    """

    fc: xr.DataArray
    wp: xr.DataArray
    ks: xr.DataArray
    awc: xr.DataArray
    total_awc_mm: xr.DataArray
    depths_cm: list[tuple[int, int]]
    tile_ids: list[str]


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------


def _find_soil_dir(cfg: RegionConfig) -> Path:
    """Locate the EU-SoilHydroGrids root directory."""
    soil_dir = cfg.static_dir / "eu_soilhydrogrids"
    if not soil_dir.exists():
        raise FileNotFoundError(
            f"EU-SoilHydroGrids directory not found: {soil_dir}\n"
            f"Download from https://esdac.jrc.ec.europa.eu/ and extract into {soil_dir}\n"
            f"Expected structure: {soil_dir}/<grid_id>/FC_M_sl1_<grid_id>.tif ..."
        )
    return soil_dir


def _find_tile_index(soil_dir: Path) -> Path | None:
    """Find the tile navigation GeoPackage or Shapefile."""
    candidates = [
        soil_dir / "grid_cells_250m_wgs84.gpkg",
        soil_dir / "grid_cells_250m_wgs84.shp",
        soil_dir / "grid_cells_250m_etrs89.gpkg",
        soil_dir / "grid_cells_250m_etrs89.shp",
        soil_dir / "tiles_nav" / "grid_cells_250m_wgs84.gpkg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_tiles_with_index(
    soil_dir: Path,
    bbox_wgs84: tuple[float, float, float, float],
    index_path: Path,
) -> list[str]:
    """Use the navigation GeoPackage/Shapefile to find relevant tile IDs.

    Parameters
    ----------
    soil_dir : root directory containing tile folders
    bbox_wgs84 : (west, south, east, north) in WGS84
    index_path : path to the tile navigation file

    Returns
    -------
    List of grid_id strings whose tiles intersect the bounding box.
    """
    logger.info("Using tile index: %s", index_path)
    tiles_gdf = gpd.read_file(index_path)

    # Ensure WGS84
    if tiles_gdf.crs and tiles_gdf.crs.to_epsg() != 4326:
        tiles_gdf = tiles_gdf.to_crs("EPSG:4326")

    region_box = box(*bbox_wgs84)
    intersecting = tiles_gdf[tiles_gdf.intersects(region_box)]

    # Extract grid_id from the column (typically "grid_id")
    id_col = None
    for col in ("grid_id", "GRID_ID", "id", "ID", "name", "NAME"):
        if col in intersecting.columns:
            id_col = col
            break

    if id_col is None:
        raise ValueError(
            f"Cannot find grid_id column in tile index. "
            f"Available columns: {list(intersecting.columns)}"
        )

    tile_ids = intersecting[id_col].astype(str).tolist()

    # Filter to tiles that actually exist on disk
    existing = [tid for tid in tile_ids if (soil_dir / tid).is_dir()]
    missing = set(tile_ids) - set(existing)
    if missing:
        logger.warning(
            "%d tiles from index not found on disk (not downloaded?): %s",
            len(missing),
            list(missing)[:5],
        )

    logger.info(
        "Tile index: %d tiles intersect region, %d exist on disk",
        len(tile_ids),
        len(existing),
    )
    return existing


def find_tiles_by_scanning(
    soil_dir: Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> list[str]:
    """Fallback: scan all tile folders and check spatial overlap with rasterio.

    Slower than using the index but works without it.
    """
    logger.info("No tile index found — scanning tile folders (this may take a minute)...")
    region_box = box(*bbox_wgs84)
    matching = []

    # List all subdirectories that look like tile folders
    candidates = [d for d in soil_dir.iterdir() if d.is_dir()]
    logger.info("Scanning %d folders...", len(candidates))

    for folder in candidates:
        grid_id = folder.name
        # Try to open any file to get bounds
        sample_file = None
        for f in folder.glob("*.tif"):
            sample_file = f
            break
        if sample_file is None:
            continue

        try:
            with rasterio.open(sample_file) as src:
                # Get bounds in WGS84
                from rasterio.warp import transform_bounds

                tile_bounds = transform_bounds(
                    src.crs,
                    "EPSG:4326",
                    *src.bounds,
                )
                tile_box = box(*tile_bounds)
                if tile_box.intersects(region_box):
                    matching.append(grid_id)
        except Exception as exc:
            logger.debug("Could not read %s: %s", sample_file, exc)

    logger.info("Found %d tiles covering the region by scanning", len(matching))
    return matching


def find_relevant_tiles(cfg: RegionConfig) -> tuple[Path, list[str]]:
    """Find which EU-SoilHydroGrids tile folders cover the region.

    Uses the tile navigation index if available, otherwise falls back
    to scanning all folders.

    Returns
    -------
    (soil_dir, list_of_grid_ids)
    """
    soil_dir = _find_soil_dir(cfg)
    bbox = cfg.bbox_wgs84.as_tuple()  # (west, south, east, north)

    index_path = _find_tile_index(soil_dir)
    if index_path is not None:
        tile_ids = find_tiles_with_index(soil_dir, bbox, index_path)
    else:
        tile_ids = find_tiles_by_scanning(soil_dir, bbox)

    if not tile_ids:
        raise FileNotFoundError(
            f"No EU-SoilHydroGrids tiles found covering the region.\n"
            f"Check that tile folders exist in {soil_dir} and that the\n"
            f"bounding box {bbox} is correct."
        )

    return soil_dir, tile_ids


# ---------------------------------------------------------------------------
# Tile reading and mosaicking
# ---------------------------------------------------------------------------


def _collect_tile_paths(
    soil_dir: Path,
    tile_ids: list[str],
    variable: str,
    layer: str,
) -> list[Path]:
    """Find all tile files for one variable + layer combination.

    Tries multiple filename patterns since ESDAC distributions vary.
    """
    paths = []
    for grid_id in tile_ids:
        tile_dir = soil_dir / grid_id

        # Primary pattern: FC_M_sl1_0001.tif
        candidates = [
            tile_dir / f"{variable}_M_{layer}_{grid_id}.tif",
            tile_dir / f"{variable}_M_{layer}_{grid_id}.tiff",
            # Some distributions omit the _M_ (median) suffix
            tile_dir / f"{variable}_{layer}_{grid_id}.tif",
        ]

        found = False
        for c in candidates:
            if c.exists():
                paths.append(c)
                found = True
                break

        if not found:
            # Try glob as last resort
            pattern = f"{variable}*{layer}*{grid_id}*.tif*"
            matches = list(tile_dir.glob(pattern))
            if matches:
                paths.append(matches[0])
            else:
                logger.debug("Missing %s %s in tile %s", variable, layer, grid_id)

    return paths


def _mosaic_and_reproject(
    tile_paths: list[Path],
    target_grid: TargetGrid,
) -> np.ndarray:
    """Mosaic multiple tiles and reproject to the target grid.

    Returns a 2-D array aligned to the target grid.
    """
    ny, nx = target_grid.shape

    if not tile_paths:
        return np.full((ny, nx), np.nan, dtype=np.float32)

    if len(tile_paths) == 1:
        # Single tile — skip mosaic step
        with rasterio.open(tile_paths[0]) as src:
            dst = np.full((ny, nx), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_grid.transform,
                dst_crs=target_grid.crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
        return dst

    # Multiple tiles — mosaic first, then reproject
    datasets = []
    try:
        for p in tile_paths:
            datasets.append(rasterio.open(p))

        mosaic_arr, mosaic_transform = merge(datasets)
        mosaic_crs = datasets[0].crs
        mosaic_nodata = datasets[0].nodata
    finally:
        for ds in datasets:
            ds.close()

    # mosaic_arr shape: (bands, height, width)
    src_data = mosaic_arr[0].astype(np.float32)
    if mosaic_nodata is not None:
        src_data[src_data == mosaic_nodata] = np.nan

    dst = np.full((ny, nx), np.nan, dtype=np.float32)
    reproject(
        source=src_data,
        destination=dst,
        src_transform=mosaic_transform,
        src_crs=mosaic_crs,
        dst_transform=target_grid.transform,
        dst_crs=target_grid.crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    return dst


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------


def load_soil_hydro(
    cfg: RegionConfig,
    target_grid: TargetGrid,
    *,
    max_depth_cm: int = 200,
    tile_ids: list[str] | None = None,
) -> SoilHydroData:
    """Load EU-SoilHydroGrids tiles, mosaic, reproject, and compute AWC.

    Parameters
    ----------
    cfg : RegionConfig
    target_grid : TargetGrid to reproject onto
    max_depth_cm : only load layers down to this depth (default 200 cm)
    tile_ids : explicit list of tile IDs to use (skips discovery)

    Returns
    -------
    SoilHydroData with all variables mosaicked onto the target grid.
    """
    soil_dir = _find_soil_dir(cfg)

    # Discover tiles if not provided
    if tile_ids is None:
        _, tile_ids = find_relevant_tiles(cfg)

    ny, nx = target_grid.shape

    # Filter layers by max depth
    layers = {k: v for k, v in DEPTH_LAYERS.items() if v[0] < max_depth_cm}
    n_layers = len(layers)
    depth_labels = list(layers.keys())
    depth_intervals = list(layers.values())

    logger.info(
        "Loading EU-SoilHydroGrids: %d variables × %d layers from %d tiles",
        len(SOIL_VARIABLES),
        n_layers,
        len(tile_ids),
    )

    # Read, mosaic, and reproject for each variable × layer
    arrays: dict[str, np.ndarray] = {}
    for var in SOIL_VARIABLES:
        var_stack = np.full((n_layers, ny, nx), np.nan, dtype=np.float32)

        for i, layer_key in enumerate(depth_labels):
            tile_paths = _collect_tile_paths(soil_dir, tile_ids, var, layer_key)

            if not tile_paths:
                logger.warning("No tiles found for %s %s — layer will be NaN", var, layer_key)
                continue

            logger.debug("Mosaicking %s %s: %d tiles", var, layer_key, len(tile_paths))
            var_stack[i] = _mosaic_and_reproject(tile_paths, target_grid)

        arrays[var] = var_stack
        logger.info("Loaded %s: %d layers", var, n_layers)

    # Build xarray DataArrays
    depth_mids = [(d[0] + d[1]) / 2 for d in depth_intervals]
    dims = ["depth", "y", "x"]
    coords = {
        "depth": depth_mids,
        "y": target_grid.ys,
        "x": target_grid.xs,
    }

    fc = xr.DataArray(
        arrays["FC"],
        dims=dims,
        coords=coords,
        name="fc",
        attrs={"units": "cm3/cm3", "long_name": "field capacity"},
    )
    wp = xr.DataArray(
        arrays["WP"],
        dims=dims,
        coords=coords,
        name="wp",
        attrs={"units": "cm3/cm3", "long_name": "wilting point"},
    )
    ks = xr.DataArray(
        arrays["KS"],
        dims=dims,
        coords=coords,
        name="ks",
        attrs={"units": "cm/day", "long_name": "saturated hydraulic conductivity"},
    )

    # Available water capacity
    awc = (fc - wp).clip(min=0)
    awc.name = "awc"
    awc.attrs = {"units": "cm3/cm3", "long_name": "available water capacity (FC - WP)"}

    # Depth-integrated AWC in mm
    # AWC [cm³/cm³] × layer thickness [cm] × 10 [mm/cm] = mm
    layer_thicknesses = np.array([d[1] - d[0] for d in depth_intervals])
    total_awc_mm = (awc * layer_thicknesses[:, np.newaxis, np.newaxis] * 10).sum(dim="depth")
    total_awc_mm.name = "total_awc_mm"
    total_awc_mm.attrs = {"units": "mm", "long_name": "total available water capacity"}

    result = SoilHydroData(
        fc=fc,
        wp=wp,
        ks=ks,
        awc=awc,
        total_awc_mm=total_awc_mm,
        depths_cm=depth_intervals,
        tile_ids=tile_ids,
    )

    logger.info(
        "Soil data loaded: grid %s, %d tiles, median AWC = %.0f mm, range [%.0f, %.0f] mm",
        target_grid.shape,
        len(tile_ids),
        float(total_awc_mm.median()),
        float(total_awc_mm.min()),
        float(total_awc_mm.max()),
    )
    return result


def save_soil_hydro(data: SoilHydroData, out_dir: Path) -> Path:
    """Save processed soil data as a single NetCDF for re-use."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "soil_hydro.nc"

    ds = xr.Dataset(
        {
            "fc": data.fc,
            "wp": data.wp,
            "ks": data.ks,
            "awc": data.awc,
            "total_awc_mm": data.total_awc_mm,
        }
    )
    ds.attrs["source"] = "EU-SoilHydroGrids (Tóth et al., 2017)"
    ds.attrs["description"] = (
        "Soil hydraulic properties mosaicked and reprojected to IrriGator target grid"
    )
    ds.attrs["tile_ids"] = ",".join(data.tile_ids)

    ds.to_netcdf(out_path)
    logger.info("Saved soil data: %s", out_path)
    return out_path
