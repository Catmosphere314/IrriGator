"""Precipitation bias correction via CDF-t (CDF-transform).

ERA5-Land precipitation has systematic biases relative to observed
precipitation.  CDF-t (Michelangeli et al., 2009) maps the ERA5-Land
precipitation distribution to the observed distribution (COMEPHORE)
while accounting for potential non-stationarity.

The correction is fitted at the COMEPHORE pixel level (1 km), using
the ERA5-Land cell value that covers that pixel.  This captures
sub-grid bias structure — a valley parcel gets a different correction
than a hilltop parcel even within the same ~9 km ERA5 cell.

When COMEPHORE data is not available, falls back to simple multiplicative
bias correction or no correction.

Calibration period
------------------
COMEPHORE is available 1997–present (~2 month lag).  The recommended
calibration period is 2010–2025 (post-upgrade, better radar coverage).
Use ``quality_frac`` from the daily COMEPHORE to filter out low-quality
days before fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

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
# Calibration at parcel level
# ---------------------------------------------------------------------------


def calibrate_precipitation(
    era5_daily: xr.Dataset,
    comephore_daily: xr.Dataset | None = None,
    *,
    parcel_lon: float | None = None,
    parcel_lat: float | None = None,
    min_quality: float = 0.5,
    method: str = "cdf_t",
) -> BiasCorrection:
    """Fit precipitation bias correction at a parcel location.

    Pairs the ERA5-Land cell covering the parcel with the nearest
    COMEPHORE pixel (1 km).  The correction captures local bias
    at sub-ERA5-cell resolution.

    Parameters
    ----------
    era5_daily : daily ERA5-Land Dataset with 'precip_mm' variable
        (from ``era5_processor.load_daily``)
    comephore_daily : daily COMEPHORE Dataset with 'precip_mm' variable
        (from ``comephore_client.load_comephore_daily``).
        May also contain 'quality_frac' for filtering.
    parcel_lon, parcel_lat : parcel WGS84 coordinates
    min_quality : minimum quality_frac to include a day in calibration
        (default: 0.5 = at least 50% of hours had good radar coverage)
    method : "cdf_t", "quantile_mapping", or "multiplicative"

    Returns
    -------
    Fitted BiasCorrection.
    """
    if comephore_daily is None:
        logger.warning(
            "No COMEPHORE data — precipitation will not be bias-corrected. "
            "Download COMEPHORE and rerun for better accuracy."
        )
        return BiasCorrection(method="none")

    # --- Extract ERA5-Land time series at parcel ---
    if parcel_lon is not None and parcel_lat is not None:
        era5_cell = era5_daily["precip_mm"].sel(
            longitude=parcel_lon,
            latitude=parcel_lat,
            method="nearest",
        )
    else:
        era5_cell = era5_daily["precip_mm"].mean(dim=["longitude", "latitude"])

    # --- Extract COMEPHORE time series at nearest 1 km pixel ---
    comephore_precip = comephore_daily["precip_mm"]

    # COMEPHORE uses Lambert-93 (x, y) — try that first, then WGS84
    if parcel_lon is not None and parcel_lat is not None:
        try:
            # Try projected coordinates (Lambert-93)
            from pyproj import Transformer

            to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
            px, py = to_l93.transform(parcel_lon, parcel_lat)
            obs_cell = comephore_precip.sel(x=px, y=py, method="nearest")
        except (KeyError, ValueError):
            # Try WGS84 coordinates
            try:
                obs_cell = comephore_precip.sel(
                    longitude=parcel_lon,
                    latitude=parcel_lat,
                    method="nearest",
                )
            except (KeyError, ValueError):
                obs_cell = comephore_precip.sel(
                    x=parcel_lon,
                    y=parcel_lat,
                    method="nearest",
                )
    else:
        obs_cell = comephore_precip.mean(dim=[d for d in comephore_precip.dims if d != "time"])

    # --- Quality filtering ---
    if "quality_frac" in comephore_daily:
        if parcel_lon is not None and parcel_lat is not None:
            try:
                from pyproj import Transformer

                to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
                px, py = to_l93.transform(parcel_lon, parcel_lat)
                quality = comephore_daily["quality_frac"].sel(x=px, y=py, method="nearest")
            except (KeyError, ValueError):
                quality = comephore_daily["quality_frac"].sel(
                    longitude=parcel_lon,
                    latitude=parcel_lat,
                    method="nearest",
                )
        else:
            quality = comephore_daily["quality_frac"].mean(
                dim=[d for d in comephore_daily["quality_frac"].dims if d != "time"]
            )

        good_quality = quality >= min_quality
        obs_cell = obs_cell.where(good_quality)
        n_filtered = int((~good_quality).sum())
        if n_filtered > 0:
            logger.info(
                "Filtered %d low-quality COMEPHORE days (quality_frac < %.2f)",
                n_filtered,
                min_quality,
            )

    # --- Align on overlapping time ---
    era5_ts = era5_cell.to_series().dropna()
    obs_ts = obs_cell.to_series().dropna()

    common_dates = era5_ts.index.intersection(obs_ts.index)
    if len(common_dates) == 0:
        logger.warning(
            "No overlapping dates between ERA5-Land and COMEPHORE — cannot fit bias correction"
        )
        return BiasCorrection(method="none")

    era5_arr = era5_ts.loc[common_dates].values
    obs_arr = obs_ts.loc[common_dates].values

    logger.info(
        "Calibrating CDF-t: %d overlapping days (ERA5 %.1f mm/d, COMEPHORE %.1f mm/d)",
        len(common_dates),
        era5_arr[era5_arr > 0.1].mean() if (era5_arr > 0.1).any() else 0,
        obs_arr[obs_arr > 0.1].mean() if (obs_arr > 0.1).any() else 0,
    )

    if method in ("cdf_t", "quantile_mapping"):
        return fit_cdf_t(era5_arr, obs_arr)
    else:
        return fit_multiplicative(era5_arr, obs_arr)
