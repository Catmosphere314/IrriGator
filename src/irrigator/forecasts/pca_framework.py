"""Gridded PCA framework for analog matching.

Fits PCA independently per calendar month × grid cell, then computes
Mahalanobis distances in PC space for analog selection with spatial voting.

Bug fixes vs draft:
- Uses explained_variance (eigenvalues), NOT explained_variance_ratio
  for the Mahalanobis precision matrix. The ratio is λ/Σλ which has no
  statistical meaning as a precision; the actual λ is needed so that
  VI = diag(1/λ₁, 1/λ₂, ...) = Σ⁻¹ in PC space.
- Vectorized inner loop: Mahalanobis computed for all candidates at once
  per cell, ~10-50× faster than the per-candidate loop.
- Division-by-zero guard on 1/distance scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RegionalPCA:
    """Monthly PCA fitted to an entire multivariate regional field.

    One historical month is one sample.  The features are the requested
    variables at every retained grid cell, flattened in a stable
    ``variable × latitude × longitude`` order.  A separate PCA is fitted for
    each calendar month so that, for example, August forecasts are compared
    only with historical Augusts.

    This is the preferred analog model for IrriGator seasonal matching.  It
    captures spatial covariance over the full France domain instead of fitting
    an unrelated PCA at each grid cell and voting afterwards.
    """

    pc_scores: xr.DataArray  # (time, component)
    pca_components: xr.DataArray  # (month, component, feature)
    explained_variance: xr.DataArray  # (month, component)
    explained_variance_ratio: xr.DataArray  # (month, component)
    feature_mean: xr.DataArray  # (month, feature)
    feature_scale: xr.DataArray  # (month, feature)
    pca_mean: xr.DataArray  # (month, feature)
    valid_feature: xr.DataArray  # (month, feature)
    feature_variable: xr.DataArray  # (feature,)
    feature_latitude: xr.DataArray  # (feature,)
    feature_longitude: xr.DataArray  # (feature,)
    attrs: dict


def _regional_feature_matrix(
    ds: xr.Dataset,
    variables: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a gridded dataset to ``(time, feature)`` in a stable order."""
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
    matrix = arr.values.reshape(ds.sizes["time"], -1)

    n_lat = ds.sizes["latitude"]
    n_lon = ds.sizes["longitude"]
    feature_variable = np.repeat(np.asarray(variables, dtype=object), n_lat * n_lon)
    feature_latitude = np.tile(np.repeat(ds.latitude.values, n_lon), len(variables))
    feature_longitude = np.tile(np.tile(ds.longitude.values, n_lat), len(variables))
    return matrix, feature_variable, feature_latitude, feature_longitude


def fit_monthly_regional_pca(
    ds: xr.Dataset,
    variables: list[str],
    n_components: int | None = 4,
    standardize: bool = True,
    min_valid_fraction: float = 1.0,
) -> RegionalPCA:
    """Fit one PCA per calendar month to the complete regional weather field.

    Parameters
    ----------
    ds
        Monthly gridded dataset with dimensions ``time, latitude, longitude``.
        Each time step should represent one historical month.
    variables
        Variables to concatenate into the regional feature vector.
    n_components
        Maximum number of retained PCs.  It is also limited by the number of
        historical years available for each calendar month.
    standardize
        Standardize every variable/grid-cell feature across historical years
        before PCA.  This is strongly recommended for mixed meteorological
        units.
    min_valid_fraction
        Fraction of historical samples that must be finite for a feature to be
        retained.  The default (1.0) is conservative and naturally removes sea
        cells when ERA5-Land contains missing values there.
    """
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1].")

    matrix, f_var, f_lat, f_lon = _regional_feature_matrix(ds, variables)
    n_features = matrix.shape[1]
    requested_components = min(len(variables), 4) if n_components is None else int(n_components)
    requested_components = max(1, requested_components)
    component_names = [f"PC{i + 1}" for i in range(requested_components)]
    months = np.arange(1, 13)

    scores = np.full((ds.sizes["time"], requested_components), np.nan, dtype=np.float64)
    components = np.full((12, requested_components, n_features), np.nan, dtype=np.float64)
    eigenvalues = np.full((12, requested_components), np.nan, dtype=np.float64)
    variance_ratio = np.full((12, requested_components), np.nan, dtype=np.float64)
    means = np.full((12, n_features), np.nan, dtype=np.float64)
    scales = np.full((12, n_features), np.nan, dtype=np.float64)
    pca_means = np.full((12, n_features), np.nan, dtype=np.float64)
    valid_features = np.zeros((12, n_features), dtype=bool)

    time_months = ds.time.dt.month.values
    for month in months:
        m_idx = month - 1
        time_idx = np.flatnonzero(time_months == month)
        if len(time_idx) < 2:
            continue

        X = matrix[time_idx]
        finite_fraction = np.mean(np.isfinite(X), axis=0)
        valid = finite_fraction >= min_valid_fraction
        if not valid.any():
            logger.warning("No valid regional PCA features for month %d", month)
            continue

        # When min_valid_fraction < 1, fill the few remaining holes with the
        # historical feature mean.  At 1.0 no imputation is performed.
        Xv = X[:, valid].astype(np.float64, copy=True)
        if not np.isfinite(Xv).all():
            col_means = np.nanmean(Xv, axis=0)
            bad = np.where(~np.isfinite(Xv))
            Xv[bad] = col_means[bad[1]]

        if standardize:
            scaler = StandardScaler()
            X_prepared = scaler.fit_transform(Xv)
            mu = scaler.mean_.astype(float)
            sigma = scaler.scale_.astype(float)
        else:
            mu = Xv.mean(axis=0)
            sigma = np.ones(Xv.shape[1], dtype=float)
            X_prepared = Xv - mu

        ncomp = min(requested_components, X_prepared.shape[0], X_prepared.shape[1])
        pca = PCA(n_components=ncomp)
        month_scores = pca.fit_transform(X_prepared)

        scores[time_idx, :ncomp] = month_scores
        valid_idx = np.flatnonzero(valid)
        components[m_idx][np.ix_(np.arange(ncomp), valid_idx)] = pca.components_
        eigenvalues[m_idx, :ncomp] = pca.explained_variance_
        variance_ratio[m_idx, :ncomp] = pca.explained_variance_ratio_
        means[m_idx, valid] = mu
        scales[m_idx, valid] = sigma
        pca_means[m_idx, valid] = pca.mean_
        valid_features[m_idx, valid] = True

        logger.debug(
            "Regional PCA month %02d: %d years, %d features, %d PCs, %.1f%% variance",
            month,
            len(time_idx),
            int(valid.sum()),
            ncomp,
            100 * float(np.nansum(pca.explained_variance_ratio_)),
        )

    feature_coord = np.arange(n_features, dtype=int)
    return RegionalPCA(
        pc_scores=xr.DataArray(
            scores,
            dims=("time", "component"),
            coords={"time": ds.time, "component": component_names},
            name="regional_pc_scores",
        ),
        pca_components=xr.DataArray(
            components,
            dims=("month", "component", "feature"),
            coords={"month": months, "component": component_names, "feature": feature_coord},
        ),
        explained_variance=xr.DataArray(
            eigenvalues,
            dims=("month", "component"),
            coords={"month": months, "component": component_names},
        ),
        explained_variance_ratio=xr.DataArray(
            variance_ratio,
            dims=("month", "component"),
            coords={"month": months, "component": component_names},
        ),
        feature_mean=xr.DataArray(
            means, dims=("month", "feature"), coords={"month": months, "feature": feature_coord}
        ),
        feature_scale=xr.DataArray(
            scales, dims=("month", "feature"), coords={"month": months, "feature": feature_coord}
        ),
        pca_mean=xr.DataArray(
            pca_means,
            dims=("month", "feature"),
            coords={"month": months, "feature": feature_coord},
        ),
        valid_feature=xr.DataArray(
            valid_features,
            dims=("month", "feature"),
            coords={"month": months, "feature": feature_coord},
        ),
        feature_variable=xr.DataArray(f_var, dims="feature", coords={"feature": feature_coord}),
        feature_latitude=xr.DataArray(f_lat, dims="feature", coords={"feature": feature_coord}),
        feature_longitude=xr.DataArray(f_lon, dims="feature", coords={"feature": feature_coord}),
        attrs={
            "variables": ",".join(variables),
            "standardize": bool(standardize),
            "scope": "regional_field",
        },
    )


def transform_to_regional_pca(
    new_ds: xr.Dataset,
    pca_model: RegionalPCA,
    variables: list[str],
) -> xr.DataArray:
    """Project one or more monthly regional fields into a fitted RegionalPCA."""
    expected_vars = list(map(str, pca_model.attrs["variables"].split(",")))
    if list(variables) != expected_vars:
        raise ValueError(f"PCA variables/order mismatch: expected {expected_vars}, got {variables}")

    matrix, f_var, f_lat, f_lon = _regional_feature_matrix(new_ds, variables)
    # Exact coordinate/order agreement matters because components are spatial.
    if not np.array_equal(f_var, pca_model.feature_variable.values):
        raise ValueError("Variable feature order differs from fitted RegionalPCA.")
    if not np.allclose(f_lat.astype(float), pca_model.feature_latitude.values.astype(float)):
        raise ValueError("Latitude grid differs from fitted RegionalPCA.")
    if not np.allclose(f_lon.astype(float), pca_model.feature_longitude.values.astype(float)):
        raise ValueError("Longitude grid differs from fitted RegionalPCA.")

    n_components = pca_model.pca_components.sizes["component"]
    out = np.full((new_ds.sizes["time"], n_components), np.nan, dtype=np.float64)

    for t_idx, t in enumerate(new_ds.time.values):
        month = pd.Timestamp(t).month
        valid = pca_model.valid_feature.sel(month=month).values.astype(bool)
        if not valid.any():
            continue

        x = matrix[t_idx, valid].astype(float)
        if not np.isfinite(x).all():
            raise ValueError(
                f"New regional field contains NaNs in PCA features for {pd.Timestamp(t):%Y-%m}. "
                "Regrid/fill the SEAS5 field before projection."
            )

        mu = pca_model.feature_mean.sel(month=month).values[valid]
        sigma = pca_model.feature_scale.sel(month=month).values[valid]
        pca_mu = pca_model.pca_mean.sel(month=month).values[valid]
        W = pca_model.pca_components.sel(month=month).values[:, valid]
        valid_pc = np.isfinite(W).all(axis=1)
        if not valid_pc.any():
            continue

        x_prepared = (x - mu) / sigma
        out[t_idx, valid_pc] = (x_prepared - pca_mu) @ W[valid_pc].T

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
) -> pd.DataFrame:
    """Rank historical months by Mahalanobis distance in regional PC space.

    The returned ``weight`` column is a normalized sampling probability over
    the retained top-k analogs, using ``exp(-d² / (2 * temperature²))``.
    ``temperature=1`` corresponds to the natural Gaussian distance scale;
    larger values deliberately diversify the sampled analogs.
    """
    if new_pc_scores.sizes.get("time", 0) != 1:
        raise ValueError("rank_regional_analogs expects exactly one target month.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    target_time = pd.Timestamp(new_pc_scores.time.values[0])
    month = target_time.month
    hist_mask = pca_model.pc_scores.time.dt.month == month
    hist = pca_model.pc_scores.sel(time=hist_mask)
    target = new_pc_scores.isel(time=0).values.astype(float)
    eigen = pca_model.explained_variance.sel(month=month).values.astype(float)

    valid_pc = np.isfinite(target) & np.isfinite(eigen) & (eigen > 1e-12)
    if not valid_pc.any():
        raise ValueError(f"No valid PCA components for calendar month {month}.")

    H = hist.values[:, valid_pc].astype(float)
    diff = H - target[valid_pc][None, :]
    distances = np.sqrt(np.sum((diff**2) / eigen[valid_pc][None, :], axis=1))
    order = np.argsort(distances)[: min(top_k, len(distances))]
    d = distances[order]

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
        }
    )
