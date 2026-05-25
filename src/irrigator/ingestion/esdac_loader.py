"""EU-SoilHydroGrids loader.

EU-SoilHydroGrids (Tóth et al., 2017) provides pre-computed soil hydraulic
properties at 250 m resolution across Europe, derived from SoilGrids texture
maps via pedotransfer functions.

Download
--------
Register (free) and download from ESDAC:
    https://esdac.jrc.ec.europa.eu/content/3d-soil-hydraulic-database-europe-1-km-and-250-m-resolution

You'll get GeoTIFF rasters per variable and depth layer.  Place them in::

    data/static/eu_soilhydrogrids/
        FC_sl1.tif    # field capacity, layer 1 (0-5 cm)
        FC_sl2.tif    # field capacity, layer 2 (5-15 cm)
        ...
        WP_sl1.tif    # wilting point, layer 1
        ...
        KS_sl1.tif    # saturated hydraulic conductivity, layer 1
        ...

The exact filenames depend on the ESDAC distribution.  Adjust
``_FILENAME_PATTERN`` if needed.

Units
-----
- FC, WP: cm³/cm³ (volumetric water content)
- KS: cm/day (saturated hydraulic conductivity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box

from irrigator.config import RegionConfig
from irrigator.ingestion.grid import TargetGrid

logger = logging.getLogger(__name__)

# Variable names as used in ESDAC filenames
SOIL_VARIABLES = ("FC", "WP", "KS")

# Depth layers in EU-SoilHydroGrids (sl1..sl7)
# Corresponding depth intervals in cm
DEPTH_LAYERS = {
    "sl1": (0, 5),
    "sl2": (5, 15),
    "sl3": (15, 30),
    "sl4": (30, 60),
    "sl5": (60, 100),
    "sl6": (100, 200),
}

# Filename pattern: {variable}_{layer}.tif
# Adjust if your downloaded files use a different convention
_FILENAME_PATTERN = "{variable}_{layer}.tif"


@dataclass
class SoilHydroData:
    """Soil hydraulic properties on the target grid.

    Attributes
    ----------
    fc : xr.DataArray
        Field capacity [cm³/cm³], dims (depth, y, x)
    wp : xr.DataArray
        Wilting point [cm³/cm³], dims (depth, y, x)
    ks : xr.DataArray
        Saturated hydraulic conductivity [cm/day], dims (depth, y, x)
    awc : xr.DataArray
        Available water capacity = FC - WP [cm³/cm³], dims (depth, y, x)
    total_awc_mm : xr.DataArray
        Depth-integrated AWC in mm over root zone, dims (y, x)
    depths_cm : list of tuples
        Depth interval for each layer, e.g. [(0,5), (5,15), ...]
    """

    fc: xr.DataArray
    wp: xr.DataArray
    ks: xr.DataArray
    awc: xr.DataArray
    total_awc_mm: xr.DataArray
    depths_cm: list[tuple[int, int]]


def _find_soil_dir(cfg: RegionConfig) -> Path:
    """Locate the EU-SoilHydroGrids directory."""
    soil_dir = cfg.static_dir / "eu_soilhydrogrids"
    if not soil_dir.exists():
        raise FileNotFoundError(
            f"EU-SoilHydroGrids directory not found: {soil_dir}\n"
            f"Download from https://esdac.jrc.ec.europa.eu/ and extract into {soil_dir}"
        )
    return soil_dir


def _read_and_reproject(
    src_path: Path,
    target_grid: TargetGrid,
) -> np.ndarray:
    """Read a GeoTIFF, reproject to the target grid CRS and resolution.

    Returns a 2-D numpy array aligned to the target grid.
    """
    profile = target_grid.rasterio_profile()
    ny, nx = target_grid.shape

    with rasterio.open(src_path) as src:
        dst_array = np.full((ny, nx), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_grid.transform,
            dst_crs=target_grid.crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    return dst_array


def load_soil_hydro(
    cfg: RegionConfig,
    target_grid: TargetGrid,
    *,
    max_depth_cm: int = 200,
) -> SoilHydroData:
    """Load EU-SoilHydroGrids data, reproject to target grid, and compute AWC.

    Parameters
    ----------
    cfg : RegionConfig
    target_grid : TargetGrid to reproject onto
    max_depth_cm : only load layers down to this depth (default 200 cm)

    Returns
    -------
    SoilHydroData with all variables on the target grid.
    """
    soil_dir = _find_soil_dir(cfg)
    ny, nx = target_grid.shape

    # Filter layers by max depth
    layers = {k: v for k, v in DEPTH_LAYERS.items() if v[0] < max_depth_cm}
    n_layers = len(layers)
    depth_labels = list(layers.keys())
    depth_intervals = list(layers.values())

    logger.info(
        "Loading EU-SoilHydroGrids: %d variables × %d layers, max depth %d cm",
        len(SOIL_VARIABLES),
        n_layers,
        max_depth_cm,
    )

    # Read all variables × layers
    arrays: dict[str, np.ndarray] = {}
    for var in SOIL_VARIABLES:
        var_stack = np.full((n_layers, ny, nx), np.nan, dtype=np.float32)
        for i, layer_key in enumerate(depth_labels):
            fname = _FILENAME_PATTERN.format(variable=var, layer=layer_key)
            fpath = soil_dir / fname

            if not fpath.exists():
                # Try alternative naming conventions
                alternatives = [
                    f"{var}_{layer_key}.tiff",
                    f"{var.lower()}_{layer_key}.tif",
                    f"{var}_{i + 1}.tif",
                ]
                found = False
                for alt in alternatives:
                    alt_path = soil_dir / alt
                    if alt_path.exists():
                        fpath = alt_path
                        found = True
                        break
                if not found:
                    logger.warning("Missing: %s — will be NaN", fname)
                    continue

            var_stack[i] = _read_and_reproject(fpath, target_grid)
            logger.debug("Loaded %s %s", var, layer_key)

        arrays[var] = var_stack

    # Build depth coordinate (use midpoint of each interval)
    depth_mids = [(d[0] + d[1]) / 2 for d in depth_intervals]

    # Create xarray DataArrays
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

    # Compute available water capacity
    awc = (fc - wp).clip(min=0)
    awc.name = "awc"
    awc.attrs = {"units": "cm3/cm3", "long_name": "available water capacity (FC - WP)"}

    # Depth-integrate AWC to get total in mm
    # AWC [cm³/cm³] × layer thickness [cm] × 10 [mm/cm] = mm
    layer_thicknesses = np.array([d[1] - d[0] for d in depth_intervals])
    # Weight each layer by its thickness, sum, convert to mm
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
    )

    logger.info(
        "Soil data loaded: grid %s, median AWC = %.0f mm, range [%.0f, %.0f] mm",
        target_grid.shape,
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
    ds.attrs["description"] = "Soil hydraulic properties reprojected to IrriGator target grid"

    ds.to_netcdf(out_path)
    logger.info("Saved soil data: %s", out_path)
    return out_path
