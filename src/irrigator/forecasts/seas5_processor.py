"""SEAS5 monthly bias correction via delta method.

Adapted from the Caterina hurricane forecast pipeline. The logic is:

    X_corrected(member, lead) = ERA5_climatology(valid_month) + SEAS5_anomaly(member, lead)

The SEAS5 anomaly dataset (from CDS `seasonal-postprocessed-*`) provides
anomalies relative to the SEAS5 1993-2016 hindcast climatology.  Adding
the ERA5 1993-2016 climatology transfers the forecast into the ERA5
domain, which is what IrriGator's water balance was calibrated on.

For irrigation use we need: 2m temperature, total precipitation,
surface solar radiation, 10m wind, dewpoint, surface pressure.
SEAS5 provides these at ~130 km monthly resolution. (1 degree)
"""

from __future__ import annotations

import logging

import xarray as xr

from irrigator.config import RegionConfig

logger = logging.getLogger(__name__)

# SEAS5 CDS leadtime convention: lead 1 = init month itself
LEAD_OFFSET = 0

# Variables we need for the irrigation pipeline
# CDS anomaly name -> short name
SEAS5_VARIABLES = {
    "2m_temperature_anomaly": "t2m",
    "total_precipitation_anomaly": "tp",
    "surface_solar_radiation_downwards_anomaly": "ssrd",
}


def compute_valid_month(init_month: int, leadtime_month: int) -> int:
    """Compute the target calendar month for a given init + lead."""
    return ((init_month - 1) + (leadtime_month - 1) + LEAD_OFFSET) % 12 + 1


def build_era5_monthly_climatology(
    cfg: RegionConfig,
    start_year: int = 1993,
    end_year: int = 2016,
) -> xr.Dataset:
    """Build ERA5 monthly climatology from downloaded monthly means.

    If you already have ERA5-Land daily data (from Block 0), aggregate
    to monthly means first, then compute the multi-year monthly average.

    Returns dataset with 'month' dimension (1-12).
    """
    from irrigator.atmospheric.era5_processor import load_daily

    daily = load_daily(cfg)

    # Filter to climatology period
    daily = daily.sel(valid_time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # Monthly means
    monthly = daily.resample(valid_time="1ME").mean()

    # Climatology: average by calendar month
    clim = monthly.groupby("valid_time.month").mean("valid_time")

    clim.attrs["climatology_period"] = f"{start_year}-{end_year}"
    clim.attrs["source"] = "ERA5-Land monthly climatology"

    logger.info(
        "ERA5 climatology built: %d-%d, %d variables",
        start_year,
        end_year,
        len(clim.data_vars),
    )
    return clim


def correct_seas5_monthly(
    seas5_anomaly: xr.Dataset,
    era5_climatology: xr.Dataset,
    init_month: int,
) -> xr.Dataset:
    """Apply delta correction to SEAS5 anomalies.

    X_corrected = ERA5_clim(valid_month) + SEAS5_anomaly

    Parameters
    ----------
    seas5_anomaly : SEAS5 anomaly dataset with leadtime_month dimension
    era5_climatology : ERA5 monthly climatology with month dimension (1-12)
    init_month : initialization month (1-12)

    Returns
    -------
    Bias-corrected monthly forecast on ERA5 absolute scale.
    """
    corrected_vars = {}

    for var in seas5_anomaly.data_vars:
        anom = seas5_anomaly[var]

        # For each lead time, find the valid month and add climatology
        leads = anom.coords.get("forecastMonth", anom.coords.get("leadtime_month"))
        if leads is None:
            logger.warning("No leadtime dimension found for %s", var)
            continue

        # Map anomaly variable name to ERA5 variable name
        era5_var = var.replace("_anomaly", "")
        # Try to find matching variable in climatology
        clim_var = None
        for cv in era5_climatology.data_vars:
            if era5_var in cv or cv in era5_var:
                clim_var = cv
                break

        if clim_var is None:
            logger.warning("No matching ERA5 climatology for %s", var)
            continue

        # Build corrected field: for each lead, add the appropriate month's climatology
        corrected_leads = []
        for lead_val in leads.values:
            valid_month = compute_valid_month(init_month, int(lead_val))
            clim_month = era5_climatology[clim_var].sel(month=valid_month)
            anom_lead = anom.sel({leads.name: lead_val})

            # Regrid climatology to SEAS5 grid if needed (bilinear)
            if clim_month.sizes != anom_lead.sizes:
                clim_month = clim_month.interp_like(anom_lead, method="linear")

            corrected_lead = clim_month + anom_lead
            corrected_leads.append(corrected_lead)

        corrected = xr.concat(corrected_leads, dim=leads.name)
        corrected.name = era5_var
        corrected_vars[era5_var] = corrected

    result = xr.Dataset(corrected_vars)
    result.attrs["bias_correction"] = "delta (ERA5_clim + SEAS5_anomaly)"
    result.attrs["init_month"] = init_month

    logger.info(
        "SEAS5 corrected: init month %d, %d variables, %d lead months",
        init_month,
        len(corrected_vars),
        len(leads),
    )
    return result
