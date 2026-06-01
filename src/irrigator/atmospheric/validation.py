"""Validation of atmospheric forcing against Météo-France stations.

Compares ERA5-Land (raw and bias-corrected) against synoptic station
observations for temperature and precipitation.  This is an independent
check — stations are NOT used in the bias correction (that uses COMEPHORE).

Stations in/near Dordogne are defined in the region config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Validation statistics for one variable at one station."""

    station_name: str
    variable: str
    n_days: int
    bias: float  # mean(model - obs)
    rmse: float  # root mean squared error
    correlation: float  # Pearson r
    nash_sutcliffe: float  # NSE (1 = perfect, 0 = mean, <0 = worse than mean)

    def __repr__(self) -> str:
        return (
            f"Validation({self.station_name}, {self.variable}): "
            f"bias={self.bias:+.2f}, RMSE={self.rmse:.2f}, "
            f"r={self.correlation:.3f}, NSE={self.nash_sutcliffe:.3f} "
            f"(n={self.n_days})"
        )


def compute_metrics(
    model: np.ndarray,
    observed: np.ndarray,
    station_name: str = "",
    variable: str = "",
) -> ValidationMetrics:
    """Compute validation metrics between model and observed time series.

    Parameters
    ----------
    model : model/reanalysis values
    observed : station observations
    station_name, variable : labels for the output

    Returns
    -------
    ValidationMetrics
    """
    mask = ~(np.isnan(model) | np.isnan(observed))
    m = model[mask]
    o = observed[mask]
    n = len(m)

    if n < 10:
        logger.warning("Too few valid days (%d) for validation", n)
        return ValidationMetrics(station_name, variable, n, np.nan, np.nan, np.nan, np.nan)

    bias = float(np.mean(m - o))
    rmse = float(np.sqrt(np.mean((m - o) ** 2)))

    if np.std(o) > 0 and np.std(m) > 0:
        correlation = float(np.corrcoef(m, o)[0, 1])
    else:
        correlation = np.nan

    ss_res = np.sum((m - o) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    metrics = ValidationMetrics(
        station_name=station_name,
        variable=variable,
        n_days=n,
        bias=bias,
        rmse=rmse,
        correlation=correlation,
        nash_sutcliffe=nse,
    )

    logger.info("%s", metrics)
    return metrics


def validate_against_stations(
    model_df: pd.DataFrame,
    station_df: pd.DataFrame,
    station_name: str,
    variables: list[str] | None = None,
) -> list[ValidationMetrics]:
    """Validate model forcing against station observations.

    Parameters
    ----------
    model_df : DataFrame with model forcing, indexed by date
    station_df : DataFrame with station observations, indexed by date
               Expected columns: t_min, t_max, precip_mm (at minimum)
    station_name : label
    variables : which variables to compare (default: all common columns)

    Returns
    -------
    List of ValidationMetrics, one per variable.
    """
    # Align on common dates
    common_dates = model_df.index.intersection(station_df.index)
    if len(common_dates) == 0:
        logger.warning("No overlapping dates between model and station %s", station_name)
        return []

    model = model_df.loc[common_dates]
    station = station_df.loc[common_dates]

    if variables is None:
        variables = [c for c in model.columns if c in station.columns]

    results = []
    for var in variables:
        if var not in model.columns or var not in station.columns:
            continue
        metrics = compute_metrics(
            model[var].values,
            station[var].values,
            station_name=station_name,
            variable=var,
        )
        results.append(metrics)

    return results
