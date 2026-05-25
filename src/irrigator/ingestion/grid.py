"""Target grid definition for IrriGator.

All ingested data gets aligned to this grid before entering the pipeline.
The grid is defined in Lambert-93 at 250 m resolution to match EU-SoilHydroGrids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from irrigator.config import RegionConfig

logger = logging.getLogger(__name__)


@dataclass
class TargetGrid:
    """Regular rectangular grid in Lambert-93.

    Attributes
    ----------
    xs : 1-D array of cell center x-coordinates (easting)
    ys : 1-D array of cell center y-coordinates (northing), top-to-bottom
    resolution_m : cell size in meters
    crs : coordinate reference system
    transform : affine transform for rasterio compatibility
    """

    xs: np.ndarray
    ys: np.ndarray
    resolution_m: int
    crs: CRS
    transform: rasterio.Affine

    @property
    def shape(self) -> tuple[int, int]:
        """(n_rows, n_cols)."""
        return (len(self.ys), len(self.xs))

    @property
    def n_cells(self) -> int:
        return len(self.ys) * len(self.xs)

    def rasterio_profile(self, dtype: str = "float32", count: int = 1) -> dict:
        """Build a rasterio write profile for this grid."""
        ny, nx = self.shape
        return {
            "driver": "GTiff",
            "dtype": dtype,
            "width": nx,
            "height": ny,
            "count": count,
            "crs": self.crs,
            "transform": self.transform,
            "compress": "deflate",
            "nodata": np.nan,
        }

    def empty_dataarray(
        self,
        name: str = "data",
        fill: float = np.nan,
    ) -> xr.DataArray:
        """Create an empty xarray DataArray aligned to this grid."""
        data = np.full(self.shape, fill, dtype=np.float32)
        return xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": self.ys, "x": self.xs},
            name=name,
            attrs={"crs": str(self.crs), "resolution_m": self.resolution_m},
        )

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return 2-D coordinate arrays (xx, yy)."""
        return np.meshgrid(self.xs, self.ys)


def build_grid(cfg: RegionConfig) -> TargetGrid:
    """Construct the target grid from a region config.

    Grid cell coordinates are at cell *centers*. The rasterio transform
    maps to the upper-left corner of the upper-left cell.
    """
    res = cfg.grid.resolution_m
    bbox = cfg.bbox_l93

    # Snap bounds outward to whole multiples of resolution
    xmin = np.floor(bbox.xmin / res) * res
    xmax = np.ceil(bbox.xmax / res) * res
    ymin = np.floor(bbox.ymin / res) * res
    ymax = np.ceil(bbox.ymax / res) * res

    # Cell centers
    xs = np.arange(xmin + res / 2, xmax, res)
    # Top-to-bottom (northing decreasing) for raster convention
    ys = np.arange(ymax - res / 2, ymin, -res)

    crs = CRS.from_user_input(cfg.crs)
    transform = from_bounds(xmin, ymin, xmax, ymax, len(xs), len(ys))

    grid = TargetGrid(
        xs=xs,
        ys=ys,
        resolution_m=res,
        crs=crs,
        transform=transform,
    )
    logger.info(
        "Grid built: %d x %d (%d cells) at %d m in %s",
        grid.shape[1],
        grid.shape[0],
        grid.n_cells,
        res,
        crs,
    )
    return grid
