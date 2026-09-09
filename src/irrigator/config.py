"""Configuration loader for IrriGator.

Loads YAML configs and provides typed accessors so the rest of the codebase
doesn't scatter dict-key lookups everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datetime import date

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

    def as_cds_area(self, grid_step: float = 0.1) -> list[float]:
        """CDS API expects [N, W, S, E], snapped outward to grid_step.

        ERA5-Land uses a 0.1° grid.  Rounding outward ensures the entire
        region of interest is covered and no edge cells are missed.
        """
        import math

        n = math.ceil(self.north / grid_step) * grid_step
        s = math.floor(self.south / grid_step) * grid_step
        w = math.floor(self.west / grid_step) * grid_step
        e = math.ceil(self.east / grid_step) * grid_step
        return [round(n, 4), round(w, 4), round(s, 4), round(e, 4)]

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
        return Path(self.data["static_dir"])


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
    irrigation_log: list[dict[str, Any]]
    sensors: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    @property
    def polygon_file(self) -> Path | None:
        pf = self.raw.get("parcel", {}).get("location", {}).get("polygon_file")
        return Path(pf) if pf else None

    def add_irrigation_event(self, event: dict[str, Any]) -> ParcelConfig:
        """Return a new ParcelConfig with an additional irrigation event."""
        new_log = [*self.irrigation_log, *event]
        return ParcelConfig(
            id=self.id,
            name=self.name,
            farmer=self.farmer,
            lat=self.lat,
            lon=self.lon,
            area_ha=self.area_ha,
            soil=self.soil,
            crop=self.crop,
            irrigation=self.irrigation,
            irrigation_log=new_log,
            sensors=self.sensors,
            raw=self.raw,
        )


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
        irrigation_log=raw.get("irrigation_log",{}),
        sensors=raw.get("sensors", {}),
        raw=raw,
    )


from copy import deepcopy


def parcel_config_to_dict(config: ParcelConfig) -> dict[str, Any]:
    """Convert ParcelConfig back to YAML while preserving unknown fields."""
    raw = deepcopy(config.raw)

    parcel = raw.setdefault("parcel", {})
    location = parcel.setdefault("location", {})

    parcel["id"] = config.id
    parcel["name"] = config.name
    parcel["farmer"] = config.farmer
    parcel["area_ha"] = config.area_ha

    location["lat"] = config.lat
    location["lon"] = config.lon

    if config.polygon_file is not None:
        location["polygon_file"] = str(config.polygon_file)
    else:
        location.pop("polygon_file", None)

    raw["soil"] = config.soil
    raw["crop"] = config.crop
    raw["irrigation"] = config.irrigation
    raw["irrigation_log"] = config.irrigation_log
    raw["sensors"] = config.sensors

    return raw


def save_parcel_config(
    config: ParcelConfig,
    path: str | Path,
) -> None:
    """Write a ParcelConfig to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw = parcel_config_to_dict(config)

    with open(path, "w") as f:
        yaml.safe_dump(
            raw,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
