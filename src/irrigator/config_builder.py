"""Region config generator for IrriGator.

Generates a region YAML config file from a French département code.
Uses IGN ADMIN EXPRESS for boundaries, and looks up nearby Météo-France
synoptic stations for validation.

Usage
-----
As a script::

    python -m irrigator.region_builder --dept 24 --output configs/dordogne.yaml

Or from Python::

    from irrigator.region_builder import build_region_config
    build_region_config("24", output_path="configs/dordogne.yaml")

Prerequisites
-------------
- Download ADMIN EXPRESS from https://geoservices.ign.fr/adminexpress
  and place the DEPARTEMENT shapefile in ``data/static/admin_express/``
- Or pass the shapefile path explicitly with ``--admin-shp``
"""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent

import click
import geopandas as gpd
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Default search paths for ADMIN EXPRESS shapefile
_ADMIN_SEARCH_PATHS = [
    "data/static/admin_express/DEPARTEMENT.shp",
    "data/static/admin_express/DEPARTEMENT.shx",
    "data/static/ADMIN-EXPRESS/DEPARTEMENT.shp",
]

# Known Météo-France synoptic stations by département (subset)
# In a real deployment you'd query the Météo-France station catalog.
# This is a starter set for southwestern France.
_KNOWN_STATIONS = {
    "24": [
        {"name": "Bergerac", "id": "24037001", "lat": 44.853, "lon": 0.52},
        {"name": "Sarlat", "id": "24520001", "lat": 44.89, "lon": 1.22},
    ],
    "47": [
        {"name": "Agen", "id": "47001002", "lat": 44.175, "lon": 0.60},
    ],
    "33": [
        {"name": "Bordeaux-Mérignac", "id": "33281001", "lat": 44.83, "lon": -0.69},
    ],
    "32": [
        {"name": "Auch", "id": "32013001", "lat": 43.65, "lon": 0.58},
    ],
    "46": [
        {"name": "Cahors", "id": "46042001", "lat": 44.45, "lon": 1.47},
    ],
    "19": [
        {"name": "Brive-la-Gaillarde", "id": "19031002", "lat": 45.15, "lon": 1.47},
    ],
}


def _find_admin_shp(explicit_path: str | None = None) -> Path:
    """Locate the ADMIN EXPRESS DEPARTEMENT shapefile."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Admin shapefile not found: {p}")

    for candidate in _ADMIN_SEARCH_PATHS:
        p = Path(candidate)
        if p.exists():
            return p

    raise FileNotFoundError(
        "ADMIN EXPRESS DEPARTEMENT shapefile not found.\n"
        "Download from https://geoservices.ign.fr/adminexpress\n"
        "Place DEPARTEMENT.shp (and .shx, .dbf, .prj) in data/static/admin_express/"
    )


def _get_dept_geometry(
    dept_code: str,
    admin_shp: str | None = None,
) -> gpd.GeoDataFrame:
    """Load and filter the département boundary."""
    shp_path = _find_admin_shp(admin_shp)
    logger.info("Loading admin boundaries from %s", shp_path)

    admin = gpd.read_file(shp_path)

    # ADMIN EXPRESS uses "INSEE_DEP" for département code
    # Try common column names
    code_col = None
    for candidate in ("INSEE_DEP", "CODE_DEPT", "code_dept", "DEP"):
        if candidate in admin.columns:
            code_col = candidate
            break

    if code_col is None:
        raise ValueError(
            f"Cannot find département code column. Available columns: {list(admin.columns)}"
        )

    dept = admin[admin[code_col] == dept_code]
    if dept.empty:
        available = sorted(admin[code_col].unique())
        raise ValueError(f"Département {dept_code} not found. Available codes: {available[:20]}...")

    return dept


def _get_nearby_stations(dept_code: str, bbox_wgs84: dict) -> list[dict]:
    """Return known validation stations for the département.

    Falls back to an empty list with a log message if none are known.
    """
    stations = _KNOWN_STATIONS.get(dept_code, [])

    if not stations:
        # Check neighbouring départements
        logger.info(
            "No pre-configured stations for département %s. "
            "Add stations manually to the generated config.",
            dept_code,
        )

    return stations


def build_region_config(
    dept_code: str,
    output_path: str | Path | None = None,
    admin_shp: str | None = None,
    resolution_m: int = 250,
) -> dict:
    """Generate a region config dict from a département code.

    Parameters
    ----------
    dept_code : French département code, e.g. "24" for Dordogne
    output_path : if provided, write YAML to this path
    admin_shp : explicit path to ADMIN EXPRESS DEPARTEMENT.shp
    resolution_m : target grid resolution in meters

    Returns
    -------
    Config dict (same structure as dordogne.yaml).
    """
    dept = _get_dept_geometry(dept_code, admin_shp)
    dept_name = dept.iloc[0].get("NOM_DEPT", dept.iloc[0].get("NOM", f"dept_{dept_code}"))

    # Bounding boxes in both CRS
    dept_wgs84 = dept.to_crs("EPSG:4326")
    w, s, e, n = dept_wgs84.total_bounds
    bbox_wgs84 = {
        "north": round(float(n), 4),
        "south": round(float(s), 4),
        "west": round(float(w), 4),
        "east": round(float(e), 4),
    }

    dept_l93 = dept.to_crs("EPSG:2154")
    xmin, ymin, xmax, ymax = dept_l93.total_bounds
    bbox_l93 = {
        "xmin": int(np.floor(xmin)),
        "ymin": int(np.floor(ymin)),
        "xmax": int(np.ceil(xmax)),
        "ymax": int(np.ceil(ymax)),
    }

    stations = _get_nearby_stations(dept_code, bbox_wgs84)

    config = {
        "region": {
            "name": dept_name,
            "code_departement": dept_code,
            "code_region": str(dept.iloc[0].get("INSEE_REG", "")),
            "bbox_wgs84": bbox_wgs84,
            "bbox_l93": bbox_l93,
            "crs": "EPSG:2154",
        },
        "grid": {
            "resolution_m": resolution_m,
            "dem_aggregation": "mean",
            "slope_aggregation": "mean",
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "static_dir": "data/static",
            "era5_land": {
                "product": "reanalysis-era5-land",
                "variables": [
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "surface_pressure",
                    "surface_solar_radiation_downwards",
                    "total_precipitation",
                ],
                "validation_variables": [
                    "volumetric_soil_water_layer_1",
                    "volumetric_soil_water_layer_2",
                    "volumetric_soil_water_layer_3",
                    "volumetric_soil_water_layer_4",
                ],
                "temporal_resolution": "hourly",
                "calibration_period": {
                    "start": "2010-01-01",
                    "end": "2023-12-31",
                },
            },
            "comephore": {
                "api_base_url": "https://public-api.meteofrance.fr/public",
                "product": "COMEPHORE",
                "resolution_km": 1.0,
            },
            "seas5": {
                "product": "seasonal-monthly-single-levels",
                "system": 51,
                "variables": ["2m_temperature", "total_precipitation"],
                "ensemble_members": 51,
                "leadtime_months": [1, 2, 3, 4, 5, 6],
            },
            "arome": {
                "api_base_url": "https://public-api.meteofrance.fr/public",
                "model": "AROME",
                "resolution_km": 1.3,
                "forecast_horizon_h": 48,
            },
            "arpege": {
                "api_base_url": "https://public-api.meteofrance.fr/public",
                "model": "ARPEGE",
                "resolution_km": 10.0,
                "forecast_horizon_h": 96,
            },
            "eu_soilhydrogrids": {
                "variables": ["FC", "WP", "KS", "AWC"],
                "depths_cm": [0, 5, 15, 30, 60, 100, 200],
            },
            "ign": {
                "dem_product": "RGE ALTI 5m",
                "admin_product": "ADMIN EXPRESS",
            },
            "theia": {
                "stac_url": "https://theia.cnes.fr/atdistrib/rocket/api/stac",
                "collection": "SENTINEL2",
                "processing_level": "L2A",
                "max_cloud_cover": 20,
            },
        },
        "downscaling": {
            "temperature": {
                "method": "lapse_rate",
                "lapse_rate_c_per_km": -6.5,
            },
            "precipitation": {
                "method": "cdf_t",
                "calibration_period": {
                    "start": "2010-01-01",
                    "end": "2020-12-31",
                },
                "validation_period": {
                    "start": "2021-01-01",
                    "end": "2023-12-31",
                },
            },
            "radiation": {
                "terrain_correction": True,
            },
        },
        "validation_stations": stations,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write with a header comment
        header = (
            f"# IrriGator — Region Configuration: {dept_name} ({dept_code})\n"
            f"# Auto-generated by irrigator.region_builder\n"
            f"# Adjust validation_stations and review bounding boxes before use.\n"
            f"#\n"
            f"# All coordinates in Lambert-93 (EPSG:2154) unless noted otherwise.\n"
            f"# WGS84 bounds provided for CDS API queries.\n\n"
        )

        with open(output_path, "w") as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info("Config written: %s", output_path)

    return config


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command("init-region")
@click.option("--dept", required=True, help="Département code, e.g. '24' for Dordogne")
@click.option(
    "--output", "-o", default=None, help="Output YAML path (default: configs/<name>.yaml)"
)
@click.option("--admin-shp", default=None, help="Path to ADMIN EXPRESS DEPARTEMENT.shp")
@click.option("--resolution", default=250, type=int, help="Grid resolution in meters")
def init_region_cli(dept: str, output: str | None, admin_shp: str | None, resolution: int):
    """Generate a region config from a French département code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = build_region_config(
        dept_code=dept,
        output_path=output,
        admin_shp=admin_shp,
        resolution_m=resolution,
    )

    dept_name = config["region"]["name"]

    if not output:
        # Default output path
        slug = dept_name.lower().replace(" ", "_").replace("-", "_")
        output = f"configs/{slug}.yaml"
        build_region_config(
            dept_code=dept,
            output_path=output,
            admin_shp=admin_shp,
            resolution_m=resolution,
        )

    click.echo(f"Region config for {dept_name} ({dept}) written to {output}")
    bbox = config["region"]["bbox_wgs84"]
    click.echo(
        f"WGS84 bbox: N={bbox['north']}, S={bbox['south']}, W={bbox['west']}, E={bbox['east']}"
    )
    click.echo(f"Grid resolution: {resolution} m")

    stations = config.get("validation_stations", [])
    if stations:
        click.echo(f"Validation stations: {', '.join(s['name'] for s in stations)}")
    else:
        click.echo("⚠ No validation stations configured — add them manually.")


if __name__ == "__main__":
    init_region_cli()
