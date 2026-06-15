"""IrriGator CLI — entry point for pipeline execution."""

from __future__ import annotations

import logging
from datetime import date, datetime

import click

from irrigator.config import load_region_config

logger = logging.getLogger("irrigator")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# -----------------------------------------------------------------------
# Root group
# -----------------------------------------------------------------------


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
@click.version_option()
def main(verbose: bool) -> None:
    """IrriGator — Irrigation decision support for maize."""
    _setup_logging(verbose)


# -----------------------------------------------------------------------
# Block 0 — Data ingestion commands
# -----------------------------------------------------------------------


@main.group()
def fetch():
    """Download external datasets (Block 0)."""
    pass


@fetch.command("era5")
@click.option("--region", default="configs/dordogne.yaml", type=click.Path(exists=True))
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--validation/--no-validation", default=False, help="Include soil moisture layers")
@click.option("--overwrite", is_flag=True)
def fetch_era5(region: str, start: str, end: str, validation: bool, overwrite: bool) -> None:
    """Download ERA5-Land reanalysis data from CDS (monthly bulk files).

    This is the preferred method for historical data: one API request per month,
    each file containing all 24 hourly timesteps for every day in that month.
    """
    from irrigator.ingestion.cds_client import fetch_era5_land_range

    cfg = load_region_config(region)
    paths = fetch_era5_land_range(
        cfg,
        _parse_date(start),
        _parse_date(end),
        include_validation=validation,
        overwrite=overwrite,
    )
    click.echo(f"Downloaded {len(paths)} ERA5-Land monthly files.")


@fetch.command("era5-daily")
@click.option("--region", default="configs/dordogne.yaml", type=click.Path(exists=True))
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--validation/--no-validation", default=False, help="Include soil moisture layers")
@click.option("--overwrite", is_flag=True)
def fetch_era5_daily(region: str, start: str, end: str, validation: bool, overwrite: bool) -> None:
    """Download ERA5-Land day by day (for operational updates).

    Slower than 'era5' (one API call per day vs per month) but useful for
    fetching a few recent days to update the water balance operationally.
    """
    from irrigator.ingestion.cds_client import fetch_era5_land_days

    cfg = load_region_config(region)
    paths = fetch_era5_land_days(
        cfg,
        _parse_date(start),
        _parse_date(end),
        include_validation=validation,
        overwrite=overwrite,
    )
    click.echo(f"Downloaded {len(paths)} ERA5-Land daily files.")


@fetch.command("seas5")
@click.option("--region", default="configs/dordogne.yaml", type=click.Path(exists=True))
@click.option("--year", required=True, type=int)
@click.option("--month", required=True, type=int)
@click.option("--overwrite", is_flag=True)
def fetch_seas5(region: str, year: int, month: int, overwrite: bool) -> None:
    """Download SEAS5 seasonal forecast from CDS."""
    from irrigator.ingestion.cds_client import fetch_seas5

    cfg = load_region_config(region)
    path = fetch_seas5(cfg, year, month, overwrite=overwrite)
    click.echo(f"SEAS5 saved: {path}")


@fetch.command("forecasts")
@click.option("--region", default="configs/dordogne.yaml", type=click.Path(exists=True))
def fetch_forecasts(region: str) -> None:
    """Download latest AROME + ARPEGE forecasts from Météo-France."""
    from irrigator.ingestion.meteofrance_client import fetch_latest_forecasts

    cfg = load_region_config(region)
    results = fetch_latest_forecasts(cfg)
    for model, path in results.items():
        click.echo(f"{model.upper()}: {path}")
    if not results:
        click.echo("No forecasts retrieved. Check METEOFRANCE_API_KEY.")


@fetch.command("comephore")
@click.option("--region", default="configs/dordogne.yaml", type=click.Path(exists=True))
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--overwrite", is_flag=True)
def fetch_comephore(region: str, start: str, end: str, overwrite: bool) -> None:
    """Download COMEPHORE radar-gauge precipitation from Météo-France."""
    from irrigator.ingestion.meteofrance_client import fetch_comephore_range

    cfg = load_region_config(region)
    paths = fetch_comephore_range(cfg, _parse_date(start), _parse_date(end), overwrite=overwrite)
    click.echo(f"Downloaded {len(paths)} COMEPHORE files.")


@fetch.command("ndvi")
@click.option("--region", default="configs/dordogne.yaml", type=click.Path(exists=True))
@click.option("--date", "target_date", required=True, help="Target date YYYY-MM-DD")
@click.option("--window", default=10, help="Search ±N days around target date")
def fetch_ndvi(region: str, target_date: str, window: int) -> None:
    """Fetch and process Sentinel-2 NDVI from THEIA."""
    from irrigator.ingestion.grid import build_grid
    from irrigator.ingestion.theia_client import fetch_and_process_ndvi

    cfg = load_region_config(region)
    grid = build_grid(cfg)
    result = fetch_and_process_ndvi(cfg, grid, _parse_date(target_date), search_window_days=window)
    if result is not None:
        click.echo(f"NDVI retrieved: median={float(result.median()):.2f}")
    else:
        click.echo("No usable Sentinel-2 scene found.")


@fetch.command("ifs-ens")
@click.option("--region", default="configs/dordogne.yaml", type=click.Path(exists=True))
@click.option("--date", "run_date", default=None, help="Run date YYYY-MM-DD (default: today)")
@click.option("--hour", default=0, type=int, help="Run hour (0 or 12)")
@click.option("--overwrite", is_flag=True)
def fetch_ifs_ens_cmd(region: str, run_date: str | None, hour: int, overwrite: bool) -> None:
    """Download ECMWF IFS ENS ensemble forecast (51 members, 15 days).

    No API key needed — uses ECMWF open data (CC-BY-4.0).
    """
    from irrigator.ingestion.ifs_ens_client import fetch_ifs_ens, fetch_latest_ifs_ens

    cfg = load_region_config(region)
    if run_date:
        path = fetch_ifs_ens(cfg, _parse_date(run_date), hour, overwrite=overwrite)
    else:
        path = fetch_latest_ifs_ens(cfg)

    if path:
        click.echo(f"IFS ENS saved: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    else:
        click.echo("Could not fetch IFS ENS — check network connectivity.")


# -----------------------------------------------------------------------
# Static layer processing
# -----------------------------------------------------------------------


@main.command("build-static")
@click.option("--region", default="configs/dordogne.yaml", type=click.Path(exists=True))
def build_static(region: str) -> None:
    """Process static layers (soil + DEM) onto the target grid."""
    from irrigator.ingestion.esdac_loader import load_soil_hydro, save_soil_hydro
    from irrigator.ingestion.grid import build_grid
    from irrigator.ingestion.ign_loader import load_terrain, save_terrain

    cfg = load_region_config(region)
    grid = build_grid(cfg)

    click.echo("Processing soil hydraulic properties...")
    soil = load_soil_hydro(cfg, grid)
    save_soil_hydro(soil, cfg.processed_dir / "static")
    click.echo(f"Soil: median AWC = {float(soil.total_awc_mm.median()):.0f} mm")

    click.echo("Processing terrain (DEM, slope, aspect)...")
    terrain = load_terrain(cfg, grid)
    save_terrain(terrain, cfg.processed_dir / "static")
    click.echo(
        f"Terrain: elevation range [{float(terrain.elevation.min()):.0f}, "
        f"{float(terrain.elevation.max()):.0f}] m"
    )

    click.echo("Static layers built successfully.")


# -----------------------------------------------------------------------
# Region config generation
# -----------------------------------------------------------------------


@main.command("init-region")
@click.option("--dept", required=True, help="Département code, e.g. '24' for Dordogne")
@click.option(
    "--output", "-o", default=None, help="Output YAML path (default: configs/<name>.yaml)"
)
@click.option("--admin-shp", default=None, help="Path to ADMIN EXPRESS DEPARTEMENT.shp")
@click.option("--resolution", default=250, type=int, help="Grid resolution in meters")
def init_region(dept: str, output: str | None, admin_shp: str | None, resolution: int) -> None:
    """Generate a region config from a French département code.

    Requires ADMIN EXPRESS DEPARTEMENT.shp from IGN (free download).
    """
    from irrigator.region_builder import build_region_config

    config = build_region_config(dept_code=dept, admin_shp=admin_shp, resolution_m=resolution)
    dept_name = config["region"]["name"]

    if not output:
        slug = dept_name.lower().replace(" ", "_").replace("-", "_")
        output = f"configs/{slug}.yaml"

    build_region_config(
        dept_code=dept, output_path=output, admin_shp=admin_shp, resolution_m=resolution
    )

    click.echo(f"Region config for {dept_name} ({dept}) → {output}")
    bbox = config["region"]["bbox_wgs84"]
    click.echo(
        f"WGS84 bbox: N={bbox['north']}, S={bbox['south']}, W={bbox['west']}, E={bbox['east']}"
    )

    stations = config.get("validation_stations", [])
    if stations:
        click.echo(f"Validation stations: {', '.join(s['name'] for s in stations)}")
    else:
        click.echo("No validation stations configured — add them manually to the YAML.")


# -----------------------------------------------------------------------
# Full pipeline (placeholder for later blocks)
# -----------------------------------------------------------------------


@main.command("run")
@click.option("--region", default="configs/dordogne.yaml", type=click.Path(exists=True))
@click.option("--parcel", required=True, type=click.Path(exists=True), help="Parcel config file")
@click.option(
    "--date", "target_date", default=None, help="Target date (YYYY-MM-DD). Default: today."
)
def run(region: str, parcel: str, target_date: str | None) -> None:
    """Run the full pipeline for a parcel and produce an irrigation recommendation."""
    cfg = load_region_config(region)
    click.echo(f"Region: {cfg.name}")
    click.echo(f"Parcel: {parcel}")
    click.echo(f"Date:   {target_date or 'today'}")
    click.echo("Full pipeline not yet implemented — build blocks 1-6 progressively.")


if __name__ == "__main__":
    main()
