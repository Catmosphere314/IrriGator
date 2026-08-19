"""PCA frameworks for weather-analog matching.

The preferred seasonal implementation is ``RegionalPCA``: one PCA fitted to
France-wide calendar-month anomalies pooled over the complete historical
archive.  The seasonal cycle is removed before fitting and analog candidates
are restricted to the same calendar month only after projection.

The older ``GriddedPCA`` implementation (independent PCA per grid cell/month
plus spatial voting) is retained for compatibility and experiments but is not
used by the SEAS5 pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass to hold fitted PCA model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GriddedPCA:
    pc_scores: xr.DataArray  # (time, lat, lon, component)
    pca_components: xr.DataArray  # (month, lat, lon, component, feature)
    explained_variance: xr.DataArray  # (month, lat, lon, component) — EIGENVALUES
    explained_variance_ratio: xr.DataArray  # for diagnostics only
    feature_mean: xr.DataArray  # (month, lat, lon, feature)
    feature_scale: xr.DataArray  # (month, lat, lon, feature)
    pca_mean: xr.DataArray  # (month, lat, lon, feature)
    attrs: dict


# ---------------------------------------------------------------------------
# Fit PCA per month × cell
# ---------------------------------------------------------------------------


def fit_monthly_gridcell_pca(
    ds: xr.Dataset,
    variables: list[str],
    n_components: int | None = None,
    standardize: bool = True,
) -> GriddedPCA:
    """Fit PCA independently for each calendar month and each grid cell.

    For each (month, lat, lon):
        samples  = all years for that month
        features = variables

    Parameters
    ----------
    ds : xr.Dataset with dims (time, latitude, longitude)
    variables : list of variable names to include
    n_components : number of PCs to keep (default: min(n_vars, 4))
    standardize : if True, standardize features before PCA

    Returns
    -------
    GriddedPCA with everything needed for transform + Mahalanobis.
    """
    n_features = len(variables)
    if n_components is None:
        n_components = min(n_features, 4)
    n_components = min(n_components, n_features)

    components_coord = [f"PC{i + 1}" for i in range(n_components)]
    months = np.arange(1, 13)
    latitudes = ds.latitude.values
    longitudes = ds.longitude.values

    # Stack features into (time, lat, lon, feature) array
    X_all = (
        ds[variables].to_array(dim="feature").transpose("time", "latitude", "longitude", "feature")
    )

    # Pre-allocate output arrays
    shape_time = (ds.sizes["time"], len(latitudes), len(longitudes), n_components)
    shape_month = (12, len(latitudes), len(longitudes), n_components)
    shape_month_feat = (12, len(latitudes), len(longitudes), n_features)
    shape_comp = (12, len(latitudes), len(longitudes), n_components, n_features)

    pc_scores = np.full(shape_time, np.nan, dtype=np.float32)
    pca_components = np.full(shape_comp, np.nan, dtype=np.float32)
    explained_var = np.full(shape_month, np.nan, dtype=np.float32)
    explained_var_ratio = np.full(shape_month, np.nan, dtype=np.float32)
    feat_mean = np.full(shape_month_feat, np.nan, dtype=np.float32)
    feat_scale = np.full(shape_month_feat, np.nan, dtype=np.float32)
    pca_mean_arr = np.full(shape_month_feat, np.nan, dtype=np.float32)

    # Month masks (pre-compute)
    time_months = ds.time.dt.month.values

    for m_idx, month in enumerate(months):
        month_mask = time_months == month
        time_indices = np.where(month_mask)[0]

        if len(time_indices) < 2:
            continue

        logger.debug("Fitting month %d (%d samples)", month, len(time_indices))

        for j, lat_val in enumerate(latitudes):
            for k, lon_val in enumerate(longitudes):
                X = X_all.values[month_mask, j, k, :]  # (n_years, n_features)

                valid = np.isfinite(X).all(axis=1)
                X_valid = X[valid]
                if X_valid.shape[0] < 2:
                    continue

                ncomp = min(n_components, X_valid.shape[0], n_features)

                # Standardize
                if standardize:
                    scaler = StandardScaler()
                    X_prepared = scaler.fit_transform(X_valid)
                    mu, sigma = scaler.mean_, scaler.scale_
                else:
                    mu = X_valid.mean(axis=0)
                    sigma = np.ones(n_features)
                    X_prepared = X_valid - mu

                # Fit PCA
                pca = PCA(n_components=ncomp)
                scores = pca.fit_transform(X_prepared)

                # Store results
                valid_time_idx = time_indices[valid]
                pc_scores[valid_time_idx, j, k, :ncomp] = scores
                pca_components[m_idx, j, k, :ncomp, :] = pca.components_

                # CRITICAL: store explained_variance (eigenvalues), not just ratio
                explained_var[m_idx, j, k, :ncomp] = pca.explained_variance_
                explained_var_ratio[m_idx, j, k, :ncomp] = pca.explained_variance_ratio_

                feat_mean[m_idx, j, k, :] = mu
                feat_scale[m_idx, j, k, :] = sigma
                pca_mean_arr[m_idx, j, k, :] = pca.mean_

    # Wrap in xarray
    common_coords = {
        "latitude": latitudes,
        "longitude": longitudes,
    }

    return GriddedPCA(
        pc_scores=xr.DataArray(
            pc_scores,
            dims=("time", "latitude", "longitude", "component"),
            coords={"time": ds.time, "component": components_coord, **common_coords},
        ),
        pca_components=xr.DataArray(
            pca_components,
            dims=("month", "latitude", "longitude", "component", "feature"),
            coords={
                "month": months,
                "component": components_coord,
                "feature": variables,
                **common_coords,
            },
        ),
        explained_variance=xr.DataArray(
            explained_var,
            dims=("month", "latitude", "longitude", "component"),
            coords={"month": months, "component": components_coord, **common_coords},
        ),
        explained_variance_ratio=xr.DataArray(
            explained_var_ratio,
            dims=("month", "latitude", "longitude", "component"),
            coords={"month": months, "component": components_coord, **common_coords},
        ),
        feature_mean=xr.DataArray(
            feat_mean,
            dims=("month", "latitude", "longitude", "feature"),
            coords={"month": months, "feature": variables, **common_coords},
        ),
        feature_scale=xr.DataArray(
            feat_scale,
            dims=("month", "latitude", "longitude", "feature"),
            coords={"month": months, "feature": variables, **common_coords},
        ),
        pca_mean=xr.DataArray(
            pca_mean_arr,
            dims=("month", "latitude", "longitude", "feature"),
            coords={"month": months, "feature": variables, **common_coords},
        ),
        attrs={"variables": ",".join(variables), "standardize": standardize},
    )


# ---------------------------------------------------------------------------
# Transform new data into PCA space
# ---------------------------------------------------------------------------


def transform_to_pca(
    new_ds: xr.Dataset,
    pca_model: GriddedPCA,
    variables: list[str],
) -> xr.DataArray:
    """Transform new gridded data into the pre-fitted PCA domain.

    Parameters
    ----------
    new_ds : xr.Dataset with dims (time, latitude, longitude)
    pca_model : fitted GriddedPCA
    variables : same variables used in fitting

    Returns
    -------
    xr.DataArray of PC scores (time, latitude, longitude, component)
    """
    X_new = (
        new_ds[variables]
        .to_array(dim="feature")
        .transpose("time", "latitude", "longitude", "feature")
    )

    components_coord = pca_model.pca_components.component.values
    n_components = len(components_coord)

    result = np.full(
        (new_ds.sizes["time"], new_ds.sizes["latitude"], new_ds.sizes["longitude"], n_components),
        np.nan,
        dtype=np.float32,
    )

    for t_idx, t in enumerate(new_ds.time.values):
        month = pd.Timestamp(t).month
        m_idx = month - 1

        for j, lat in enumerate(new_ds.latitude.values):
            for k, lon in enumerate(new_ds.longitude.values):
                x = X_new.values[t_idx, j, k, :]
                if not np.isfinite(x).all():
                    continue

                mu = pca_model.feature_mean.values[m_idx, j, k, :]
                sigma = pca_model.feature_scale.values[m_idx, j, k, :]
                pca_mu = pca_model.pca_mean.values[m_idx, j, k, :]
                W = pca_model.pca_components.values[m_idx, j, k, :, :]  # (ncomp, nfeat)

                valid = np.isfinite(W).all(axis=1)
                if not valid.any():
                    continue

                x_std = (x - mu) / sigma
                x_centered = x_std - pca_mu
                scores = x_centered @ W[valid].T
                result[t_idx, j, k, valid] = scores

    return xr.DataArray(
        result,
        dims=("time", "latitude", "longitude", "component"),
        coords={
            "time": new_ds.time,
            "latitude": new_ds.latitude,
            "longitude": new_ds.longitude,
            "component": components_coord,
        },
    )


# ---------------------------------------------------------------------------
# Mahalanobis distance — VECTORIZED, with correct eigenvalues
# ---------------------------------------------------------------------------


def mahalanobis_candidates(
    pca_model: GriddedPCA,
    new_pc_scores: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute Mahalanobis distance for each candidate at each grid cell.

    Uses explained_variance (eigenvalues), NOT ratio, for the
    precision matrix: VI = diag(1/λ₁, 1/λ₂, ...).

    Vectorized: computes all candidates at once per cell.

    Parameters
    ----------
    pca_model : fitted GriddedPCA
    new_pc_scores : PC scores of the SEAS5 target (time=1, lat, lon, component)

    Returns
    -------
    (distances, scores) — both (latitude, longitude, candidate)
    """
    if new_pc_scores.sizes["time"] != 1:
        raise ValueError("Expected single time step in new_pc_scores")

    month = int(new_pc_scores.time.dt.month.values[0])

    # Historical candidates for this month
    month_mask = pca_model.pc_scores.time.dt.month == month
    candidates = pca_model.pc_scores.sel(time=month_mask)
    candidate_times = candidates.time.values
    n_candidates = len(candidate_times)

    n_lat = new_pc_scores.sizes["latitude"]
    n_lon = new_pc_scores.sizes["longitude"]

    distances = np.full((n_lat, n_lon, n_candidates), np.nan, dtype=np.float64)
    target = new_pc_scores.isel(time=0).values  # (lat, lon, component)

    # Eigenvalues for this month: (lat, lon, component)
    eigenvalues = pca_model.explained_variance.sel(month=month).values

    # Historical PC scores for this month: (n_candidates, lat, lon, component)
    hist = candidates.values

    for j in range(n_lat):
        for k in range(n_lon):
            t_vec = target[j, k, :]  # (n_components,)
            lam = eigenvalues[j, k, :]  # (n_components,)

            # Skip cells with invalid PCA
            valid_pc = np.isfinite(lam) & (lam > 1e-12)
            if not valid_pc.any():
                continue

            # Precision weights: 1/λ for each component
            precision = np.zeros_like(lam)
            precision[valid_pc] = 1.0 / lam[valid_pc]

            # Vectorized: all candidates at once
            # diff: (n_candidates, n_components)
            diff = hist[:, j, k, :] - t_vec[np.newaxis, :]

            # Mahalanobis: sqrt(Σ (diff²_i / λ_i))
            # = sqrt(diff @ diag(1/λ) @ diff.T) per candidate
            weighted_sq = diff**2 * precision[np.newaxis, :]
            distances[j, k, :] = np.sqrt(np.nansum(weighted_sq, axis=1))

    # Score: inverse distance with guard against zero
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(distances > 1e-12, 1.0 / distances, 1e12)

    coords = {
        "latitude": new_pc_scores.latitude,
        "longitude": new_pc_scores.longitude,
        "candidate": candidate_times,
    }

    return (
        xr.DataArray(
            distances,
            dims=("latitude", "longitude", "candidate"),
            coords=coords,
            name="mahalanobis_distance",
        ),
        xr.DataArray(
            scores,
            dims=("latitude", "longitude", "candidate"),
            coords=coords,
            name="mahalanobis_score",
        ),
    )


# ---------------------------------------------------------------------------
# Spatial voting to select best analog
# ---------------------------------------------------------------------------


def select_candidate(
    mahalanobis_score: xr.DataArray,
    method: str = "soft",
) -> tuple[pd.Timestamp, dict]:
    """Select best analog month by spatial voting.

    Parameters
    ----------
    mahalanobis_score : (latitude, longitude, candidate) scores
    method : "soft" (mean inverse distance) or "hard" (majority vote)

    Returns
    -------
    (best_candidate_time, diagnostics)
    """
    if method == "soft":
        # Mean score across all cells per candidate
        candidate_scores = mahalanobis_score.mean(dim=["latitude", "longitude"])
    elif method == "hard":
        # Count how many cells vote for each candidate (argmax per cell)
        best_per_cell = mahalanobis_score.argmax(dim="candidate")
        n_candidates = mahalanobis_score.sizes["candidate"]
        votes = np.zeros(n_candidates)
        for idx in best_per_cell.values.flat:
            if np.isfinite(idx):
                votes[int(idx)] += 1
        candidate_scores = xr.DataArray(
            votes,
            dims="candidate",
            coords={"candidate": mahalanobis_score.candidate},
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Rank candidates
    ranked = candidate_scores.sortby(candidate_scores, ascending=False)
    best = ranked.candidate.values[0]
    runner_up = ranked.candidate.values[1] if len(ranked) > 1 else None

    best_score = float(ranked.values[0])
    runner_score = float(ranked.values[1]) if len(ranked) > 1 else 0

    diagnostics = {
        "best_year": pd.Timestamp(best).year,
        "best_month": pd.Timestamp(best).month,
        "best_score": best_score,
        "runner_up_year": pd.Timestamp(runner_up).year if runner_up else None,
        "vote_margin": best_score / runner_score if runner_score > 0 else float("inf"),
        "method": method,
    }

    logger.info(
        "Analog selected: %d-%02d (score=%.2f, margin=%.2f vs runner-up %s)",
        diagnostics["best_year"],
        diagnostics["best_month"],
        best_score,
        diagnostics["vote_margin"],
        diagnostics.get("runner_up_year"),
    )

    return pd.Timestamp(best), diagnostics


# ---------------------------------------------------------------------------
# Regional / France-level PCA
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Regional / France-level pooled anomaly PCA
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionalPCA:
    """One pooled PCA fitted to France-wide monthly weather anomalies.

    One historical month is one sample and the features are the requested
    meteorological variables at every retained grid cell, flattened in stable
    ``variable × latitude × longitude`` order.

    The seasonal cycle is removed *before* fitting PCA:

    1. compute a climatology independently for each calendar month/feature;
    2. subtract that climatology from every historical month;
    3. divide every feature by one residual standard deviation estimated from
       the complete anomaly archive; and
    4. fit a single PCA to all months together.

    The PCA basis is therefore estimated from the full historical archive
    rather than from only ~40--60 realizations of each calendar month.  Analog
    selection is still restricted to the target calendar month later on.
    """

    pc_scores: xr.DataArray  # (time, component)
    pca_components: xr.DataArray  # (component, feature)
    explained_variance: xr.DataArray  # (component)
    explained_variance_ratio: xr.DataArray  # (component)
    monthly_climatology: xr.DataArray  # (month, feature)
    anomaly_scale: xr.DataArray  # (feature)
    feature_weight: xr.DataArray  # (feature), e.g. optional latitude weighting
    pca_mean: xr.DataArray  # (feature)
    valid_feature: xr.DataArray  # (feature)
    spatial_mask: xr.DataArray # mask to reuse
    feature_variable: xr.DataArray  # (feature,)
    feature_latitude: xr.DataArray  # (feature,)
    feature_longitude: xr.DataArray  # (feature,)
    attrs: dict

    # Compatibility aliases for older notebook/code references.  ``feature_mean``
    # used to mean a month-specific StandardScaler mean, which is now exactly the
    # calendar-month climatology.  ``feature_scale`` is now one pooled residual
    # scale per feature.
    @property
    def feature_mean(self) -> xr.DataArray:
        return self.monthly_climatology

    @property
    def feature_scale(self) -> xr.DataArray:
        return self.anomaly_scale

    @property
    def era5_scores(self) -> pd.DataFrame:
        """Historical PCA scores as the DataFrame used in the exploratory notebook."""
        return pd.DataFrame(
            self.pc_scores.values,
            index=pd.DatetimeIndex(self.pc_scores.time.values),
            columns=[str(v) for v in self.pc_scores.component.values],
        )

    @property
    def feature_index(self) -> pd.MultiIndex:
        """Variable/latitude/longitude feature index for notebook-style inspection."""
        return pd.MultiIndex.from_arrays(
            [
                self.feature_variable.values,
                self.feature_latitude.values,
                self.feature_longitude.values,
            ],
            names=["variable", "latitude", "longitude"],
        )

# def _regional_feature_matrix(
#     ds: xr.Dataset,
#     variables: list[str],
#     *,
#     spatial_mask: xr.DataArray | None = None,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Flatten a gridded dataset to (time, feature) in stable order."""

#     missing = [v for v in variables if v not in ds.data_vars]
#     if missing:
#         raise ValueError(f"Dataset is missing PCA variable(s): {missing}")

#     if "time" not in ds.dims:
#         raise ValueError("Regional PCA expects a 'time' dimension.")

#     if "latitude" not in ds.dims or "longitude" not in ds.dims:
#         raise ValueError("Regional PCA expects 'latitude' and 'longitude' dimensions.")

#     arr = (
#         ds[variables]
#         .to_array(dim="variable")
#         .transpose("time", "variable", "latitude", "longitude")
#     )

#     if spatial_mask is not None:
#         if set(spatial_mask.dims) != {"latitude", "longitude"}:
#             raise ValueError("spatial_mask must have latitude/longitude dimensions.")

#         arr = arr.where(spatial_mask)

#     # Stack all spatial/variable dimensions
#     X = arr.stack(feature=("variable", "latitude", "longitude"))

#     # Remove features that are completely missing
#     X = X.dropna("feature", how="all")

#     # Remove features with any missing time step
#     valid = np.isfinite(X).all("time")
#     X = X.isel(feature=valid.values)

#     # Matrix used by PCA
#     matrix = X.values

#     # Metadata corresponding EXACTLY to columns of matrix
#     feature_variable = X["variable"].values
#     feature_latitude = X["latitude"].values
#     feature_longitude = X["longitude"].values

#     return (
#         matrix,
#         feature_variable,
#         feature_latitude,
#         feature_longitude,
#     )
def _regional_feature_matrix(
    ds: xr.Dataset,
    variables: list[str],
    *,
    spatial_mask: xr.DataArray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a gridded dataset to (time, feature) in stable order."""

    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Dataset is missing PCA variable(s): {missing}")

    if "time" not in ds.dims:
        raise ValueError("Regional PCA expects a 'time' dimension.")

    if "latitude" not in ds.dims or "longitude" not in ds.dims:
        raise ValueError("Regional PCA expects 'latitude' and 'longitude' dimensions.")

    arr = (
        ds[variables]
        .to_array(dim="variable")
        .transpose("time", "variable", "latitude", "longitude")
    )

    X = arr.stack(feature=("variable", "latitude", "longitude"))

    if spatial_mask is not None:
        if set(spatial_mask.dims) != {"latitude", "longitude"}:
            raise ValueError("spatial_mask must have latitude/longitude dimensions.")

        # Align mask explicitly to this grid
        mask = spatial_mask.sel(
            latitude=arr.latitude,
            longitude=arr.longitude,
        )

        # Repeat the 2-D mask once per variable because feature ordering is
        # variable × latitude × longitude.
        keep = np.broadcast_to(
            mask.values[None, :, :],
            (
                len(variables),
                mask.sizes["latitude"],
                mask.sizes["longitude"],
            ),
        ).reshape(-1)

        X = X.isel(feature=keep)

    matrix = X.values

    return (
        matrix,
        X["variable"].values,
        X["latitude"].values,
        X["longitude"].values,
    )



def fit_regional_anomaly_pca(
    ds: xr.Dataset,
    variables: list[str],
    n_components: int = 10,
    *,
    spatial_mask: xr.DataArray | None = None,
    min_valid_fraction: float = 0.99,
    latitude_weighting: bool = False,
) -> RegionalPCA:
    """Fit one pooled PCA to calendar-month anomalies over a regional field.

    Parameters
    ----------
    ds
        Monthly gridded dataset with dimensions ``time, latitude, longitude``.
        Each time step represents one historical month.
    variables
        Variables concatenated into the France-wide feature vector.
    n_components
        Number of retained PCs.  Ten is a useful default for analog matching;
        PC1/PC2 can still be used only for visualization.
    spatial_mask
        Optional boolean France mask on the PCA grid.  Cells outside the mask
        are removed from the fitted feature space.  Cross-border cells can be
        retained simply by marking every grid cell that intersects France.
    min_valid_fraction
        Fraction of historical samples that must be finite for a feature to be
        retained.  ``1.0`` reproduces the conservative notebook behavior.
    latitude_weighting
        If True, multiply standardized anomalies by ``sqrt(cos(latitude))``.
        This is optional because the France domain is relatively compact and
        the main notebook experiments did not use area weighting.
    """
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1].")
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    matrix, f_var, f_lat, f_lon = _regional_feature_matrix(
        ds,
        variables,
        spatial_mask=spatial_mask,
    )
    n_samples, n_features = matrix.shape
    finite_fraction = np.mean(np.isfinite(matrix), axis=0)
    valid = finite_fraction >= min_valid_fraction
    if not valid.any():
        raise ValueError("No valid features remain for regional PCA.")

    X = matrix[:, valid].astype(np.float64, copy=True)
    if not np.isfinite(X).all():
        # Only used when min_valid_fraction < 1.  With the default 1.0, the
        # notebook behavior is preserved and no imputation occurs.
        col_means = np.nanmean(X, axis=0)
        bad = np.where(~np.isfinite(X))
        X[bad] = col_means[bad[1]]

    months = np.arange(1, 13, dtype=int)
    time_months = ds.time.dt.month.values.astype(int)
    climatology_valid = np.full((12, X.shape[1]), np.nan, dtype=np.float64)
    for month in months:
        month_mask = time_months == month
        if month_mask.any():
            climatology_valid[month - 1] = np.mean(X[month_mask], axis=0)

    if not np.isfinite(climatology_valid).all():
        missing_months = months[~np.isfinite(climatology_valid).all(axis=1)]
        raise ValueError(
            "Regional anomaly PCA needs at least one historical sample for every "
            f"calendar month; missing={missing_months.tolist()}."
        )

    anomalies = X - climatology_valid[time_months - 1]
    anomaly_scale_valid = np.std(anomalies, axis=0)
    anomaly_scale_valid = np.where(
        np.isfinite(anomaly_scale_valid) & (anomaly_scale_valid > 1e-12),
        anomaly_scale_valid,
        1.0,
    )
    Z = anomalies / anomaly_scale_valid

    weight_valid = np.ones(X.shape[1], dtype=np.float64)
    if latitude_weighting:
        lat_valid = f_lat[valid].astype(float)
        weight_valid = np.sqrt(np.clip(np.cos(np.deg2rad(lat_valid)), 0.0, None))
        Z = Z * weight_valid[None, :]

    ncomp = min(int(n_components), n_samples - 1, Z.shape[1])
    if ncomp < 1:
        raise ValueError("Not enough samples/features to fit PCA.")

    pca = PCA(n_components=ncomp)
    scores = pca.fit_transform(Z)
    component_names = [f"PC{i + 1}" for i in range(ncomp)]
    feature_coord = np.arange(n_features, dtype=int)

    components = np.full((ncomp, n_features), np.nan, dtype=np.float64)
    components[:, valid] = pca.components_

    climatology = np.full((12, n_features), np.nan, dtype=np.float64)
    climatology[:, valid] = climatology_valid
    anomaly_scale = np.full(n_features, np.nan, dtype=np.float64)
    anomaly_scale[valid] = anomaly_scale_valid
    feature_weight = np.full(n_features, np.nan, dtype=np.float64)
    feature_weight[valid] = weight_valid
    pca_mean = np.full(n_features, np.nan, dtype=np.float64)
    pca_mean[valid] = pca.mean_

    logger.info(
        "Pooled regional anomaly PCA: %d monthly samples, %d/%d features, %d PCs, %.1f%% variance",
        n_samples,
        int(valid.sum()),
        n_features,
        ncomp,
        100 * float(np.sum(pca.explained_variance_ratio_)),
    )

    return RegionalPCA(
        pc_scores=xr.DataArray(
            scores,
            dims=("time", "component"),
            coords={"time": ds.time, "component": component_names},
            name="regional_pc_scores",
        ),
        pca_components=xr.DataArray(
            components,
            dims=("component", "feature"),
            coords={"component": component_names, "feature": feature_coord},
        ),
        explained_variance=xr.DataArray(
            pca.explained_variance_,
            dims="component",
            coords={"component": component_names},
        ),
        explained_variance_ratio=xr.DataArray(
            pca.explained_variance_ratio_,
            dims="component",
            coords={"component": component_names},
        ),
        monthly_climatology=xr.DataArray(
            climatology,
            dims=("month", "feature"),
            coords={"month": months, "feature": feature_coord},
        ),
        anomaly_scale=xr.DataArray(
            anomaly_scale,
            dims="feature",
            coords={"feature": feature_coord},
        ),
        feature_weight=xr.DataArray(
            feature_weight,
            dims="feature",
            coords={"feature": feature_coord},
        ),
        pca_mean=xr.DataArray(
            pca_mean,
            dims="feature",
            coords={"feature": feature_coord},
        ),
        valid_feature=xr.DataArray(
            valid,
            dims="feature",
            coords={"feature": feature_coord},
        ),
        spatial_mask=spatial_mask,
        feature_variable=xr.DataArray(f_var, dims="feature", coords={"feature": feature_coord}),
        feature_latitude=xr.DataArray(f_lat, dims="feature", coords={"feature": feature_coord}),
        feature_longitude=xr.DataArray(f_lon, dims="feature", coords={"feature": feature_coord}),
        attrs={
            "variables": ",".join(variables),
            "standardization": "calendar_month_climatology_then_pooled_residual_sd",
            "scope": "regional_field_pooled_monthly_anomalies",
            "latitude_weighting": bool(latitude_weighting),
        },
    )


def fit_monthly_regional_pca(
    ds: xr.Dataset,
    variables: list[str],
    n_components: int | None = 10,
    standardize: bool = True,
    min_valid_fraction: float = 1.0,
    *,
    spatial_mask: xr.DataArray | None = None,
    latitude_weighting: bool = False,
) -> RegionalPCA:
    """Backward-compatible wrapper for :func:`fit_regional_anomaly_pca`.

    Despite the legacy name, this now fits **one pooled PCA**, not one PCA per
    calendar month.  ``standardize=False`` is no longer supported because the
    pooled anomaly formulation relies on residual scaling to balance mixed
    meteorological variables.
    """
    if not standardize:
        raise ValueError(
            "Pooled regional PCA requires anomaly standardization. "
            "Use fit_regional_anomaly_pca() for the explicit API."
        )
    return fit_regional_anomaly_pca(
        ds,
        variables,
        n_components=10 if n_components is None else int(n_components),
        spatial_mask=spatial_mask,
        min_valid_fraction=min_valid_fraction,
        latitude_weighting=latitude_weighting,
    )


def transform_to_regional_pca(
    new_ds: xr.Dataset,
    pca_model: RegionalPCA,
    variables: list[str],
) -> xr.DataArray:
    """Project monthly regional fields into the fitted pooled anomaly PCA."""
    expected_vars = list(map(str, pca_model.attrs["variables"].split(",")))
    if list(variables) != expected_vars:
        raise ValueError(f"PCA variables/order mismatch: expected {expected_vars}, got {variables}")
    matrix, f_var, f_lat, f_lon = _regional_feature_matrix(
        new_ds,
        variables,
        spatial_mask=pca_model.spatial_mask
    )
    if not np.array_equal(f_var, pca_model.feature_variable.values):
        raise ValueError("Variable feature order differs from fitted RegionalPCA.")
    if not np.allclose(f_lat.astype(float), pca_model.feature_latitude.values.astype(float)):
        raise ValueError("Latitude grid differs from fitted RegionalPCA.")
    if not np.allclose(f_lon.astype(float), pca_model.feature_longitude.values.astype(float)):
        raise ValueError("Longitude grid differs from fitted RegionalPCA.")

    valid = pca_model.valid_feature.values.astype(bool)
    n_components = pca_model.pca_components.sizes["component"]
    out = np.full((new_ds.sizes["time"], n_components), np.nan, dtype=np.float64)

    W = pca_model.pca_components.values[:, valid]
    valid_pc = np.isfinite(W).all(axis=1)
    pca_mu = pca_model.pca_mean.values[valid]
    scale = pca_model.anomaly_scale.values[valid]
    weights = pca_model.feature_weight.values[valid]


    for t_idx, t in enumerate(new_ds.time.values):
        month = pd.Timestamp(t).month
        x = matrix[t_idx, valid].astype(float)
        if not np.isfinite(x).all():
            print("ERROR")
            np.savetxt("/home/mbaldacchino/data/matrix.txt",x)
            raise ValueError(
                f"New regional field contains NaNs in PCA features for {pd.Timestamp(t):%Y-%m}. "
                "Ensure the SEAS5 field covers every retained France cell."
            )

        climatology = pca_model.monthly_climatology.sel(month=month).values[valid]
        z = ((x - climatology) / scale) * weights
        out[t_idx, valid_pc] = (z - pca_mu) @ W[valid_pc].T

    return xr.DataArray(
        out,
        dims=("time", "component"),
        coords={"time": new_ds.time, "component": pca_model.pc_scores.component},
        name="regional_pc_scores",
    )


def rank_regional_analogs(
    pca_model: RegionalPCA,
    new_pc_scores: xr.DataArray,
    *,
    top_k: int = 5,
    temperature: float = 1.0,
    metric: Literal["euclidean", "mahalanobis"] = "euclidean",
) -> pd.DataFrame:
    """Rank historical analogs of the same calendar month in PC space.

    ``euclidean`` is the default because the PCA is used as a truncated
    similarity representation: distance then measures separation in the
    retained standardized-anomaly subspace.  ``mahalanobis`` remains available
    for experiments and weights differences by inverse PC variance.

    No forecast-time cutoff is applied here.  This deliberately supports the
    notebook diagnostic where a forecast is visualized against the complete
    reanalysis cloud.  A past-only candidate filter can be added later for
    strict hindcast validation without changing the fitted PCA.
    """
    if new_pc_scores.sizes.get("time", 0) != 1:
        raise ValueError("rank_regional_analogs expects exactly one target month.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    if metric not in {"euclidean", "mahalanobis"}:
        raise ValueError("metric must be 'euclidean' or 'mahalanobis'.")

    target_time = pd.Timestamp(new_pc_scores.time.values[0])
    month = target_time.month
    hist_mask = pca_model.pc_scores.time.dt.month == month
    hist = pca_model.pc_scores.sel(time=hist_mask)
    target = new_pc_scores.isel(time=0).values.astype(float)

    valid_pc = np.isfinite(target)
    H = hist.values.astype(float)
    valid_pc &= np.isfinite(H).all(axis=0)
    if metric == "mahalanobis":
        eigen = pca_model.explained_variance.values.astype(float)
        valid_pc &= np.isfinite(eigen) & (eigen > 1e-12)
    if not valid_pc.any():
        raise ValueError("No valid PCA components are available for analog ranking.")

    diff = H[:, valid_pc] - target[valid_pc][None, :]
    if metric == "euclidean":
        distances = np.sqrt(np.sum(diff**2, axis=1))
    else:
        eigen = pca_model.explained_variance.values.astype(float)[valid_pc]
        distances = np.sqrt(np.sum((diff**2) / eigen[None, :], axis=1))

    order = np.argsort(distances)[: min(top_k, len(distances))]
    d = distances[order]

    # Retain the existing Gaussian sampling-temperature semantics.  Subtracting
    # the maximum log-weight avoids underflow without changing probabilities.
    logw = -0.5 * (d / temperature) ** 2
    logw -= np.nanmax(logw)
    weights = np.exp(logw)
    weights /= weights.sum()

    times = pd.to_datetime(hist.time.values[order])
    return pd.DataFrame(
        {
            "time": times,
            "year": times.year,
            "month": times.month,
            "distance": d,
            "weight": weights,
            "rank": np.arange(1, len(order) + 1),
            "metric": metric,
        }
    )


def plot_regional_pca_candidates(
    pca_model: RegionalPCA,
    forecast_pc: xr.DataArray | np.ndarray,
    *,
    month: int,
    year: int | None = None,
    top_n: int | None = None,
    annotate: bool = True,
    ax=None,
):
    """Plot a forecast in PC1/PC2 against same-month historical reanalysis.

    This is the cleaned production equivalent of the notebook diagnostic.  It
    intentionally allows the complete reanalysis archive (including years later
    than ``year``) because its purpose is visual comparison, not hindcast skill
    assessment.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - plotting is optional
        raise ImportError("matplotlib is required for PCA plotting.") from exc

    if pca_model.pc_scores.sizes["component"] < 2:
        raise ValueError("At least two PCA components are required for a 2-D plot.")
    if not 1 <= int(month) <= 12:
        raise ValueError("month must be in 1..12.")

    forecast = np.asarray(forecast_pc, dtype=float).reshape(-1)
    if forecast.size < 2:
        raise ValueError("forecast_pc must contain at least PC1 and PC2.")

    historical = pca_model.pc_scores.sel(time=pca_model.pc_scores.time.dt.month == int(month))
    if top_n is not None:
        h = historical.values
        d = np.linalg.norm(h - forecast[None, : h.shape[1]], axis=1)
        keep = np.argsort(d)[: min(int(top_n), len(d))]
        historical = historical.isel(time=keep)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    x = historical.sel(component="PC1").values
    y = historical.sel(component="PC2").values
    ax.scatter(x, y, alpha=0.65, s=30, label="ERA5 reanalysis")

    if annotate:
        for t, px, py in zip(pd.to_datetime(historical.time.values), x, y, strict=False):
            ax.annotate(
                t.strftime("%Y-%m"),
                xy=(px, py),
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=9,
                ha="center",
            )

    ax.scatter(forecast[0], forecast[1], s=55, marker="x", label="Forecast")
    if year is not None:
        ax.annotate(
            f"{int(year)}-{int(month):02d}",
            xy=(forecast[0], forecast[1]),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=11,
            ha="center",
        )

    ratio = pca_model.explained_variance_ratio.values
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%})")
    ax.legend()
    return ax
