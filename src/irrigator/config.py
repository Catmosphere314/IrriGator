"""Configuration loader for IrriGator.

Loads YAML configs and provides typed accessors so the rest of the codebase
doesn't scatter dict-key lookups everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Lightweight typed wrappers — intentionally flat, no over-engineering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBoxWGS84:
    north: float
    south: float
    west: float
    east: float

    def as_cds_area(self) -> list[float]:
        """CDS API expects [N, W, S, E]."""
        return [self.north, self.west, self.south, self.east]

    def as_tuple(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) — standard for rasterio/shapely."""
        return (self.west, self.south, self.east, self.north)


@dataclass(frozen=True)
class BBoxL93:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass(frozen=True)
class GridConfig:
    resolution_m: int
    dem_aggregation: str
    slope_aggregation: str


@dataclass(frozen=True)
class RegionConfig:
    """Fully parsed region configuration."""

    name: str
    code_departement: str
    crs: str
    bbox_wgs84: BBoxWGS84
    bbox_l93: BBoxL93
    grid: GridConfig
    data: dict[str, Any]
    downscaling: dict[str, Any]
    validation_stations: list[dict[str, Any]]
    raw: dict[str, Any] = field(repr=False)

    # Convenience paths -------------------------------------------------
    # All data directories are namespaced by region slug so that multiple
    # regions can coexist without file collisions.

    @property
    def _slug(self) -> str:
        """Filesystem-safe region identifier."""
        return self.name.lower().replace(" ", "_").replace("-", "_")

    @property
    def raw_dir(self) -> Path:
        return Path(self.data["raw_dir"]) / self._slug

    @property
    def processed_dir(self) -> Path:
        return Path(self.data["processed_dir"]) / self._slug

    @property
    def static_dir(self) -> Path:
        return Path(self.data["static_dir"]) / self._slug


def load_region_config(path: str | Path) -> RegionConfig:
    """Load and parse a region YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Region config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    region = raw["region"]
    return RegionConfig(
        name=region["name"],
        code_departement=region["code_departement"],
        crs=region["crs"],
        bbox_wgs84=BBoxWGS84(**region["bbox_wgs84"]),
        bbox_l93=BBoxL93(**region["bbox_l93"]),
        grid=GridConfig(**raw["grid"]),
        data=raw["data"],
        downscaling=raw.get("downscaling", {}),
        validation_stations=raw.get("validation_stations", []),
        raw=raw,
    )


@dataclass(frozen=True)
class ParcelConfig:
    """Parsed parcel configuration."""

    id: str
    name: str
    farmer: str
    lat: float
    lon: float
    area_ha: float
    soil: dict[str, Any]
    crop: dict[str, Any]
    irrigation: dict[str, Any]
    sensors: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    @property
    def polygon_file(self) -> Path | None:
        pf = self.raw.get("parcel", {}).get("location", {}).get("polygon_file")
        return Path(pf) if pf else None


def load_parcel_config(path: str | Path) -> ParcelConfig:
    """Load and parse a parcel YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parcel config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    parcel = raw["parcel"]
    loc = parcel["location"]
    return ParcelConfig(
        id=parcel["id"],
        name=parcel["name"],
        farmer=parcel["farmer"],
        lat=loc["lat"],
        lon=loc["lon"],
        area_ha=parcel["area_ha"],
        soil=raw.get("soil", {}),
        crop=raw.get("crop", {}),
        irrigation=raw.get("irrigation", {}),
        sensors=raw.get("sensors", {}),
        raw=raw,
    )
