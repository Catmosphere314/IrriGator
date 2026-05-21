"""IrriGator CLI — entry point for pipeline execution."""

import click


@click.group()
@click.version_option()
def main():
    """IrriGator — Irrigation decision support for maize."""
    pass


@main.command()
@click.option("--region", default="configs/dordogne.yaml", help="Region config file")
@click.option("--parcel", required=True, help="Parcel config file")
@click.option("--date", default=None, help="Target date (YYYY-MM-DD). Default: today.")
def run(region, parcel, date):
    """Run the full pipeline for a parcel and produce an irrigation recommendation."""
    click.echo(f"Region config: {region}")
    click.echo(f"Parcel config: {parcel}")
    click.echo(f"Target date:   {date or 'today'}")
    click.echo("Pipeline not yet implemented — start with Block 0.")


@main.command()
@click.option("--region", default="configs/dordogne.yaml", help="Region config file")
def fetch_data(region):
    """Download/update all external datasets for the configured region."""
    click.echo(f"Fetching data for region: {region}")
    click.echo("Not yet implemented — start with src/irrigator/ingestion/")


@main.command()
@click.option("--region", default="configs/dordogne.yaml", help="Region config file")
@click.option("--period", nargs=2, type=str, help="Start and end date (YYYY-MM-DD)")
def validate(region, period):
    """Run validation suite against station data and ERA5-Land soil moisture."""
    click.echo(f"Validation for region: {region}")
    if period:
        click.echo(f"Period: {period[0]} to {period[1]}")
    click.echo("Not yet implemented — see configs/model_parameters.yaml validation section.")


if __name__ == "__main__":
    main()
