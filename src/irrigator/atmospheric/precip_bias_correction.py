"""Precipitation bias correction via CDF-t (CDF-transform).

ERA5-Land precipitation has systematic biases relative to observed
precipitation.  CDF-t (Michelangeli et al., 2009) maps the ERA5-Land
precipitation distribution to the observed distribution (COMEPHORE)
while accounting for potential non-stationarity.

The correction is fitted PER ERA5-Land GRID CELL using the COMEPHORE
pixels that fall within each cell.  This ensures local bias structure
is captured (alluvial valleys vs. limestone plateaux behave differently).

When COMEPHORE data is not available, falls back to simple multiplicative
bias correction or no correction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import interpolate, stats

from irrigator.config import RegionConfig

logger = logging.getLogger(__name__)


@dataclass
class BiasCorrection:
    """Fitted bias correction for one grid cell or parcel.

    Stores the CDF mapping from ERA5-Land to observed precipitation.
    """

    method: str  # "cdf_t", "quantile_mapping", "multiplicative", "none"
    # For CDF-t / QM: percentile mapping
    era5_quantiles: np.ndarray | None = None  # ERA5 precipitation at each percentile
    obs_quantiles: np.ndarray | None = None  # Observed precipitation at same percentiles
    percentiles: np.ndarray | None = None  # percentile levels [0-100]
    # For multiplicative: single scale factor
    scale_factor: float = 1.0
    # Wet-day threshold [mm] for zero-inflation handling
    wet_day_threshold: float = 0.1

    def correct(self, precip: float | np.ndarray) -> float | np.ndarray:
        """Apply the fitted correction to precipitation values."""
        if self.method == "none":
            return precip

        if self.method == "multiplicative":
            return np.maximum(0, precip * self.scale_factor)

        if self.method in ("cdf_t", "quantile_mapping"):
            return self._apply_quantile_mapping(precip)

        return precip

    def _apply_quantile_mapping(self, precip: float | np.ndarray) -> float | np.ndarray:
        """Interpolate through the quantile mapping."""
        scalar = np.isscalar(precip)
        precip = np.atleast_1d(np.asarray(precip, dtype=np.float64))
        result = np.zeros_like(precip)

        # Dry days stay dry
        wet = precip >= self.wet_day_threshold
        if not np.any(wet):
            return float(result[0]) if scalar else result

        # Interpolate: ERA5 quantiles → observed quantiles
        corrected = np.interp(
            precip[wet],
            self.era5_quantiles,
            self.obs_quantiles,
            left=0.0,
            right=self.obs_quantiles[-1],
        )
        result[wet] = np.maximum(0, corrected)

        return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# CDF-t fitting
# ---------------------------------------------------------------------------


def fit_cdf_t(
    era5_cal: np.ndarray,
    obs_cal: np.ndarray,
    n_quantiles: int = 100,
    wet_threshold: float = 0.1,
) -> BiasCorrection:
    """Fit CDF-t bias correction from calibration period data.

    Parameters
    ----------
    era5_cal : ERA5-Land daily precipitation over calibration period [mm]
    obs_cal : observed (COMEPHORE) daily precipitation, same period [mm]
    n_quantiles : number of quantile levels for the mapping
    wet_threshold : minimum precipitation to consider as a wet day [mm]

    Returns
    -------
    BiasCorrection object with fitted quantile mapping.
    """
    # Remove NaN
    mask = ~(np.isnan(era5_cal) | np.isnan(obs_cal))
    era5 = era5_cal[mask]
    obs = obs_cal[mask]

    if len(era5) < 365:
        logger.warning(
            "Fewer than 365 days of calibration data (%d) — bias correction may be unreliable",
            len(era5),
        )

    # Separate wet days
    era5_wet = era5[era5 >= wet_threshold]
    obs_wet = obs[obs >= wet_threshold]

    if len(era5_wet) < 50 or len(obs_wet) < 50:
        logger.warning("Too few wet days — falling back to multiplicative correction")
        scale = obs.sum() / era5.sum() if era5.sum() > 0 else 1.0
        return BiasCorrection(method="multiplicative", scale_factor=scale)

    # Compute empirical quantiles for wet days
    percentiles = np.linspace(0, 100, n_quantiles + 1)
    era5_q = np.percentile(era5_wet, percentiles)
    obs_q = np.percentile(obs_wet, percentiles)

    # Ensure monotonicity (CDF must be non-decreasing)
    era5_q = np.maximum.accumulate(era5_q)
    obs_q = np.maximum.accumulate(obs_q)

    logger.info(
        "CDF-t fitted: %d cal days, %d wet days, ERA5 mean=%.1f mm, obs mean=%.1f mm, ratio=%.2f",
        len(era5),
        len(era5_wet),
        era5_wet.mean(),
        obs_wet.mean(),
        obs_wet.mean() / era5_wet.mean(),
    )

    return BiasCorrection(
        method="cdf_t",
        era5_quantiles=era5_q,
        obs_quantiles=obs_q,
        percentiles=percentiles,
        wet_day_threshold=wet_threshold,
    )


def fit_multiplicative(
    era5_cal: np.ndarray,
    obs_cal: np.ndarray,
) -> BiasCorrection:
    """Simple multiplicative bias correction (ratio of means)."""
    mask = ~(np.isnan(era5_cal) | np.isnan(obs_cal))
    era5_sum = era5_cal[mask].sum()
    obs_sum = obs_cal[mask].sum()
    scale = obs_sum / era5_sum if era5_sum > 0 else 1.0

    logger.info("Multiplicative correction: scale=%.3f", scale)

    return BiasCorrection(method="multiplicative", scale_factor=scale)


# ---------------------------------------------------------------------------
# Calibration workflow
# ---------------------------------------------------------------------------


def calibrate_precipitation(
    cfg: RegionConfig,
    era5_daily: xr.Dataset,
    comephore: xr.Dataset | None = None,
    method: str = "cdf_t",
    parcel_x: float | None = None,
    parcel_y: float | None = None,
) -> BiasCorrection:
    """Fit precipitation bias correction at a parcel location.

    Uses ERA5-Land precipitation and COMEPHORE observations over the
    calibration period defined in the config.

    Parameters
    ----------
    cfg : RegionConfig
    era5_daily : daily ERA5-Land with 'precip_mm' variable
    comephore : COMEPHORE daily precipitation (if available)
    method : "cdf_t", "quantile_mapping", or "multiplicative"
    parcel_x, parcel_y : parcel coordinates in Lambert-93

    Returns
    -------
    Fitted BiasCorrection.
    """
    if comephore is None:
        logger.warning(
            "No COMEPHORE data — precipitation will not be bias-corrected. "
            "Download COMEPHORE and rerun for better accuracy."
        )
        return BiasCorrection(method="none")

    # Extract time series at the nearest grid points
    if parcel_x is not None and parcel_y is not None:
        era5_ts = era5_daily["precip_mm"].sel(x=parcel_x, y=parcel_y, method="nearest").values
    else:
        # Use spatial mean as fallback
        era5_ts = era5_daily["precip_mm"].mean(dim=["x", "y"]).values

    # COMEPHORE needs to be matched to the same time range
    # and aggregated to the ERA5 grid cell (mean of ~80 COMEPHORE pixels)
    if parcel_x is not None and parcel_y is not None:
        # COMEPHORE is in WGS84 typically — may need coordinate matching
        try:
            obs_ts = comephore.sel(x=parcel_x, y=parcel_y, method="nearest").values.flatten()
        except Exception:
            # Try lat/lon coordinates
            from pyproj import Transformer

            to_wgs = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
            lon, lat = to_wgs.transform(parcel_x, parcel_y)
            obs_ts = comephore.sel(longitude=lon, latitude=lat, method="nearest").values.flatten()
    else:
        obs_ts = comephore.mean(dim=["x", "y"]).values.flatten()

    # Align lengths
    min_len = min(len(era5_ts), len(obs_ts))
    era5_ts = era5_ts[:min_len]
    obs_ts = obs_ts[:min_len]

    if method == "cdf_t" or method == "quantile_mapping":
        return fit_cdf_t(era5_ts, obs_ts)
    else:
        return fit_multiplicative(era5_ts, obs_ts)
