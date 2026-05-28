"""GEODES / THEIA Sentinel-2 client for vegetation indices.

THEIA Sentinel-2 L2A products (atmospherically corrected with MAJA) are now
distributed through the GEODES platform (https://geodes.cnes.fr), which
replaced the old MUSCATE/PEPS servers in early 2025.

This module handles:
- Searching GEODES for cloud-free Sentinel-2 L2A scenes over the region
- Downloading scene archives and extracting bands (B4 Red, B8 NIR)
- Computing NDVI at native 10 m resolution
- Aggregating NDVI to the target grid

NDVI is used as an independent cross-check on the crop module (Block 4):
if observed canopy greenness diverges from what the GDD model predicts,
something is off (wrong planting date, pest damage, replanting, etc.).

Prerequisites
-------------
- Install ``pygeodes``:  pip install pygeodes
- Register at https://geodes.cnes.fr
- For **search**: no API key needed.
- For **download**: generate an API key from your GEODES profile page
  (log in → click your username top-right → Profile → Generate API key).
  Then either:
    (a) set environment variable ``GEODES_API_KEY``
    (b) create a JSON config file: {"api_key": "...", "download_dir": "..."}
"""

from __future__ import annotations

import glob
import logging
import os
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from rasterio.warp import Resampling, reproject

from irrigator.config import RegionConfig
from irrigator.ingestion.grid import TargetGrid

logger = logging.getLogger(__name__)

# GEODES collection for THEIA Sentinel-2 L2A
_COLLECTION = "THEIA_REFLECTANCE_SENTINEL2_L2A"

# Sentinel-2 bands for NDVI
_BAND_RED = "B4"  # 665 nm, 10 m
_BAND_NIR = "B8"  # 842 nm, 10 m


# ---------------------------------------------------------------------------
# GEODES client setup
# ---------------------------------------------------------------------------


def _get_geodes_client():
    """Initialise a pygeodes Geodes client, with API key if available.

    Search works without an API key.  Download requires one.
    """
    try:
        from pygeodes import Geodes, Config
    except ImportError:
        raise ImportError(
            "Install pygeodes: pip install pygeodes\n"
            "Documentation: https://cnes.github.io/pyGeodes/"
        )

    api_key = os.environ.get("GEODES_API_KEY")

    if api_key:
        conf = Config(api_key=api_key)
        client = Geodes(conf=conf)
        logger.debug("GEODES client initialised with API key")
    else:
        client = Geodes()
        logger.debug("GEODES client initialised without API key (search only)")

    return client


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_scenes(
    cfg: RegionConfig,
    start_date: date,
    end_date: date,
    max_cloud_cover: int | None = None,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """Search GEODES for THEIA Sentinel-2 L2A scenes over the region.

    Parameters
    ----------
    cfg : RegionConfig
    start_date, end_date : temporal range
    max_cloud_cover : override config default (percent, 0-100)
    max_results : max scenes to return

    Returns
    -------
    List of dicts with keys: id, datetime, cloud_cover, grid_code, item
    (the raw pygeodes Item object is kept for downloading).
    """
    theia_cfg = cfg.data.get("theia", {})
    cloud_limit = max_cloud_cover or theia_cfg.get("max_cloud_cover", 20)
    bbox = cfg.bbox_wgs84

    logger.info(
        "Searching GEODES for Sentinel-2 L2A: %s to %s, cloud ≤ %d%%",
        start_date,
        end_date,
        cloud_limit,
    )

    client = _get_geodes_client()

    query = {
        "start_datetime": {"gte": f"{start_date.isoformat()}T00:00:00Z"},
        "end_datetime": {"lte": f"{end_date.isoformat()}T23:59:59Z"},
        "eo:cloud_cover": {"lte": cloud_limit},
    }

    try:
        items, df = client.search_items(
            query=query,
            bbox=[bbox.west, bbox.south, bbox.east, bbox.north],
            collections=[_COLLECTION],
        )
    except Exception as exc:
        logger.error("GEODES search failed: %s", exc)
        logger.info("Check your connection or browse manually at https://geodes.cnes.fr")
        return []

    if not items:
        logger.info("No scenes found matching criteria")
        return []

    # Extract metadata from pygeodes Items
    from pygeodes.utils.formatting import format_items

    df_full = format_items(
        df,
        columns_to_add={"eo:cloud_cover", "start_datetime", "grid:code"},
    )

    results = []
    for i, row in df_full.iterrows():
        item_obj = row["item"]

        # Parse datetime
        dt_str = row.get("start_datetime", "")
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            dt = None

        results.append(
            {
                "id": row["id"],
                "datetime": dt,
                "cloud_cover": row.get("eo:cloud_cover", None),
                "grid_code": row.get("grid:code", ""),
                "item": item_obj,  # keep for download
            }
        )

    # Limit results
    results = results[:max_results]

    logger.info("Found %d scenes", len(results))
    return results


# ---------------------------------------------------------------------------
# Download and extract
# ---------------------------------------------------------------------------


def download_scene(
    scene: dict[str, Any],
    out_dir: Path,
    *,
    api_key: str | None = None,
) -> Path | None:
    """Download a scene archive from GEODES.

    Parameters
    ----------
    scene : dict from search_scenes (must contain 'item' key)
    out_dir : directory to save the downloaded archive
    api_key : override GEODES_API_KEY env var

    Returns
    -------
    Path to the extracted scene directory, or None on failure.
    """
    from pygeodes import Geodes, Config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    item = scene.get("item")
    if item is None:
        logger.error("Scene dict has no 'item' — cannot download")
        return None

    key = api_key or os.environ.get("GEODES_API_KEY")
    if not key:
        logger.error(
            "No GEODES API key found. Set GEODES_API_KEY env var or pass api_key.\n"
            "Generate one at https://geodes.cnes.fr (Profile → Generate API key)."
        )
        return None

    conf = Config(api_key=key, download_dir=str(out_dir))
    client = Geodes(conf=conf)

    scene_id = scene["id"]
    logger.info("Downloading scene %s from GEODES...", scene_id)

    try:
        client.download_item_archive(item)
    except Exception as exc:
        logger.error("Download failed for %s: %s", scene_id, exc)
        return None

    # Find the downloaded ZIP file
    zips = list(out_dir.glob("*.zip"))
    if not zips:
        logger.error("No ZIP file found after download in %s", out_dir)
        return None

    # Use the most recently modified ZIP
    zip_path = max(zips, key=lambda p: p.stat().st_mtime)
    logger.info("Downloaded: %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    # Extract
    extract_dir = out_dir / zip_path.stem
    if not extract_dir.exists():
        logger.info("Extracting %s...", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    return extract_dir


def find_band_file(scene_dir: Path, band: str) -> Path | None:
    """Find a specific band file within an extracted THEIA L2A scene.

    THEIA L2A products have nested directory structures.  Band files are
    typically named like: *_FRE_B4.tif or *_SRE_B4.tif
    (FRE = flat reflectance, SRE = surface reflectance).
    We prefer FRE when available.
    """
    # Recursive search for the band
    # FRE (terrain-corrected) is preferred over SRE
    patterns = [
        f"**/*FRE_{band}.tif",
        f"**/*FRE_{band}.tiff",
        f"**/*SRE_{band}.tif",
        f"**/*SRE_{band}.tiff",
        f"**/*{band}*.tif",
    ]

    for pattern in patterns:
        matches = list(scene_dir.glob(pattern))
        if matches:
            logger.debug("Found %s: %s", band, matches[0])
            return matches[0]

    logger.warning("Band %s not found in %s", band, scene_dir)
    return None


# ---------------------------------------------------------------------------
# NDVI computation (unchanged from original)
# ---------------------------------------------------------------------------


def compute_ndvi(
    red_path: Path,
    nir_path: Path,
) -> tuple[np.ndarray, rasterio.Affine, Any]:
    """Compute NDVI from Red (B4) and NIR (B8) bands.

    Returns
    -------
    ndvi : 2-D float32 array, range [-1, 1]
    transform : affine transform of the output
    crs : CRS of the output
    """
    with rasterio.open(red_path) as red_ds, rasterio.open(nir_path) as nir_ds:
        red = red_ds.read(1).astype(np.float32)
        nir = nir_ds.read(1).astype(np.float32)
        transform = red_ds.transform
        crs = red_ds.crs

    # Handle THEIA's nodata and scaling
    # THEIA L2A reflectances are typically int16 scaled by 10000
    # with nodata = -10000
    nodata_mask = (red <= -10000) | (nir <= -10000)

    # Convert to 0-1 reflectance if scaled
    if red.max() > 100:  # clearly scaled
        red = red / 10000.0
        nir = nir / 10000.0

    # NDVI
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, np.nan)

    # Mask invalid
    ndvi = np.where(nodata_mask, np.nan, ndvi)
    ndvi = np.where((red >= 0) & (nir >= 0), ndvi, np.nan)
    ndvi = np.clip(ndvi, -1.0, 1.0).astype(np.float32)

    return ndvi, transform, crs


def aggregate_ndvi_to_grid(
    ndvi: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: Any,
    target_grid: TargetGrid,
) -> xr.DataArray:
    """Reproject NDVI from native 10 m to the target grid."""
    ny, nx = target_grid.shape
    dst = np.full((ny, nx), np.nan, dtype=np.float32)

    reproject(
        source=ndvi,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_grid.transform,
        dst_crs=target_grid.crs,
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    return xr.DataArray(
        dst,
        dims=["y", "x"],
        coords={"y": target_grid.ys, "x": target_grid.xs},
        name="ndvi",
        attrs={
            "units": "dimensionless",
            "long_name": "NDVI",
            "source": "Sentinel-2 L2A via GEODES/THEIA",
        },
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline — multi-tile
# ---------------------------------------------------------------------------


def _group_scenes_by_date(
    scenes: list[dict[str, Any]],
) -> dict[date, list[dict[str, Any]]]:
    """Group scenes by acquisition date (ignoring time)."""
    groups: dict[date, list[dict[str, Any]]] = {}
    for s in scenes:
        dt = s.get("datetime")
        if dt is None:
            continue
        d = dt.date()
        groups.setdefault(d, []).append(s)
    return groups


def _pick_best_date(
    groups: dict[date, list[dict[str, Any]]],
    target_date: date,
) -> date | None:
    """Pick the acquisition date with best coverage and lowest cloud.

    Scoring:  primary = closeness to target_date
              secondary = number of tiles (more = better coverage)
              tertiary = mean cloud cover (lower = better)
    """
    if not groups:
        return None

    def score(d: date) -> tuple:
        tiles = groups[d]
        day_distance = abs((d - target_date).days)
        n_tiles = len(tiles)
        mean_cloud = np.mean([t.get("cloud_cover") or 100 for t in tiles])
        # Sort: closest date first, then most tiles, then least cloud
        return (day_distance, -n_tiles, mean_cloud)

    return min(groups.keys(), key=score)


def _mosaic_ndvi_tiles(
    ndvi_arrays: list[tuple[np.ndarray, rasterio.Affine, Any]],
    target_grid: TargetGrid,
) -> xr.DataArray:
    """Mosaic multiple NDVI tiles onto the target grid.

    Each tile is reprojected to the target grid independently, then
    combined by nanmean (for overlapping areas between tiles).
    """
    ny, nx = target_grid.shape
    stack = []

    for ndvi, transform, crs in ndvi_arrays:
        dst = np.full((ny, nx), np.nan, dtype=np.float32)
        reproject(
            source=ndvi,
            destination=dst,
            src_transform=transform,
            src_crs=crs,
            dst_transform=target_grid.transform,
            dst_crs=target_grid.crs,
            resampling=Resampling.average,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        stack.append(dst)

    # Combine: nanmean handles overlaps and preserves single-tile areas
    mosaic = np.nanmean(np.stack(stack, axis=0), axis=0)

    return xr.DataArray(
        mosaic,
        dims=["y", "x"],
        coords={"y": target_grid.ys, "x": target_grid.xs},
        name="ndvi",
        attrs={
            "units": "dimensionless",
            "long_name": "NDVI",
            "source": "Sentinel-2 L2A via GEODES/THEIA",
            "n_tiles": len(ndvi_arrays),
        },
    )


def fetch_and_process_ndvi(
    cfg: RegionConfig,
    target_grid: TargetGrid,
    target_date: date,
    *,
    search_window_days: int = 10,
) -> xr.DataArray | None:
    """End-to-end: search, download ALL tiles for best date, mosaic NDVI.

    Unlike a single-tile approach, this downloads every tile from the
    selected acquisition date so the NDVI covers the full region.
    Sentinel-2 tiles are ~110 × 110 km; a département like Dordogne
    spans 4+ tiles.

    Parameters
    ----------
    cfg : RegionConfig
    target_grid : TargetGrid
    target_date : desired observation date
    search_window_days : look ±N days around target_date

    Returns
    -------
    NDVI DataArray on the target grid, or None if no usable scene found.
    """
    start = target_date - timedelta(days=search_window_days)
    end = target_date + timedelta(days=search_window_days)

    scenes = search_scenes(cfg, start, end, max_results=100)
    if not scenes:
        logger.warning("No Sentinel-2 scenes found near %s", target_date)
        return None

    # Group by date and pick the best one
    groups = _group_scenes_by_date(scenes)
    best_date = _pick_best_date(groups, target_date)
    if best_date is None:
        logger.warning("No valid dates found")
        return None

    tiles = groups[best_date]
    tile_codes = [t.get("grid_code", "?") for t in tiles]
    mean_cloud = np.mean([t.get("cloud_cover") or 0 for t in tiles])

    logger.info(
        "Selected date %s: %d tiles (%s), mean cloud %.0f%%",
        best_date,
        len(tiles),
        ", ".join(tile_codes),
        mean_cloud,
    )

    # Download and process each tile
    s2_dir = cfg.raw_dir / "sentinel2"
    ndvi_arrays = []

    for tile in tiles:
        tile_code = tile.get("grid_code", "?")
        logger.info("Processing tile %s...", tile_code)

        scene_dir = download_scene(tile, s2_dir)
        if scene_dir is None:
            logger.warning("Skipping tile %s — download failed", tile_code)
            continue

        red_path = find_band_file(scene_dir, _BAND_RED)
        nir_path = find_band_file(scene_dir, _BAND_NIR)

        if red_path is None or nir_path is None:
            logger.warning(
                "Skipping tile %s — B4/B8 not found. Files: %s",
                tile_code,
                [p.name for p in scene_dir.rglob("*.tif")][:10],
            )
            continue

        ndvi, transform, crs = compute_ndvi(red_path, nir_path)
        ndvi_arrays.append((ndvi, transform, crs))
        logger.info("Tile %s: NDVI median=%.2f", tile_code, float(np.nanmedian(ndvi)))

    if not ndvi_arrays:
        logger.warning("No tiles could be processed for %s", best_date)
        return None

    # Mosaic all tiles onto the target grid
    logger.info("Mosaicking %d tiles onto target grid...", len(ndvi_arrays))
    ndvi_grid = _mosaic_ndvi_tiles(ndvi_arrays, target_grid)

    # Coverage check
    valid_frac = float((~np.isnan(ndvi_grid.values)).mean())
    logger.info(
        "NDVI mosaic: %d tiles, coverage=%.0f%%, median=%.2f, range=[%.2f, %.2f]",
        len(ndvi_arrays),
        valid_frac * 100,
        float(ndvi_grid.median()),
        float(np.nanmin(ndvi_grid.values)),
        float(np.nanmax(ndvi_grid.values)),
    )

    if valid_frac < 0.5:
        logger.warning(
            "Low coverage (%.0f%%) — some tiles may be missing. "
            "Consider widening the search window.",
            valid_frac * 100,
        )

    # Attach time coordinate
    ndvi_grid = ndvi_grid.expand_dims(
        time=[datetime(best_date.year, best_date.month, best_date.day)]
    )

    return ndvi_grid
