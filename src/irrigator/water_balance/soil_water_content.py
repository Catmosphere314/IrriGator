import logging
import pandas as pd


from irrigator.static_layers.soil import SoilProfile
from irrigator.config import ParcelConfig
from irrigator.ingestion.cds_client import (
    DEFAULT_STATIC_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_PROCESSED_DIR,
    ERA5_LAND_SOIL_VARIABLES,
    _init_cds_client,
    _era5_output_path,
    monthrange,
    FRANCE_BBOX,
    BBoxWGS84,
)
from datetime import date
from pathlib import Path
import xarray as xr
import numpy as np

logger = logging.getLogger(__name__)

ERA5_SOIL_DEPTHS = pd.DataFrame(
    {"depth_min": [0.0, 0.07, 0.28, 1.0], "depth_max": [0.07, 0.28, 1.0, 2.89]}
)  # in meters

DEFAULT_SOIL_DEPTHS = pd.DataFrame(
    {"depth_min": [i/10 for i in range(20)], "depth_max": [i/10 for i in range(1, 21)]}
)  # in meters

SOIL_LAYERS = {
    "swvl1": {
        "layer": 1,
        "depth_top_cm": 0,
        "depth_bottom_cm": 7,
    },
    "swvl2": {
        "layer": 2,
        "depth_top_cm": 7,
        "depth_bottom_cm": 28,
    },
    "swvl3": {
        "layer": 3,
        "depth_top_cm": 28,
        "depth_bottom_cm": 100,
    },
    "swvl4": {
        "layer": 4,
        "depth_top_cm": 100,
        "depth_bottom_cm": 289,
    },
}

ERA5_SOIL_HYDRAULICS = {
    1: {"name": "coarse", "theta_s": 0.403, "theta_fc": 0.244, "theta_wp": 0.059},
    2: {"name": "medium", "theta_s": 0.439, "theta_fc": 0.347, "theta_wp": 0.151},
    3: {"name": "medium_fine", "theta_s": 0.430, "theta_fc": 0.383, "theta_wp": 0.133},
    4: {"name": "fine", "theta_s": 0.520, "theta_fc": 0.448, "theta_wp": 0.279},
    5: {"name": "very_fine", "theta_s": 0.614, "theta_fc": 0.541, "theta_wp": 0.335},
    6: {"name": "organic", "theta_s": 0.766, "theta_fc": 0.663, "theta_wp": 0.267},
}

from pathlib import Path
import zipfile


def extract_and_remove_zip(zip_path):
    """Extracts a ZIP file and removes the original ZIP file.
    
    Notes:
        Useful for the ERA5-Land soil moisture data, which is downloaded as a ZIP file.
    """

    zip_path = Path(zip_path)

    # Extract into a folder with the same name as the ZIP
    output_dir = zip_path.with_suffix("")

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    # Remove original ZIP
    zip_path.unlink()

    return output_dir




def _era5_soil_output_path(
    raw_dir: Path, year: int, month: int, day: int,
) -> Path:
    """Consistent file naming: era5land_YYYY_MM_DD.nc"""
    out_dir = raw_dir / "era5_soil"
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / f"era5land_{year:04d}_{month:02d}_{day:02d}.zip"

def _era5_soil_static_output_path(
    raw_dir: Path,
) -> Path:
    """Consistent file naming: era5land_static_soil.nc"""
    out_dir = raw_dir / "era5_soil"
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / f"era5land_static_soil.nc"


def fetch_era5_land_soil(
    year: int,
    month: int,
    day: int,
    *,
    bounding_box: BBoxWGS84 = FRANCE_BBOX,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
    force_cds_refresh: bool = False,
) -> Path:
    """Download one month of hourly ERA5-Land data.

    Parameters
    ----------
    year, month, day : target period
    bounding_box : spatial extent (default: France metropolitan)
    raw_dir : output directory for ERA5 downloads
    overwrite : re-download even if file exists
    force_cds_refresh : to avoid cache issues when overwriting

    Returns
    -------
    Path to the downloaded NetCDF file.
    """
    raw_dir = Path(raw_dir)
    out_path = _era5_soil_output_path(raw_dir, year, month, day)

    if out_path.exists() and not overwrite:
        logger.info("ERA5-Land %04d-%02d already exists: %s", year, month, out_path)
        return out_path
        
    request = {
        "variable": list(ERA5_LAND_SOIL_VARIABLES),
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{day:02d}"],
        "time_zone": "utc+00:00",
        "daily_statistic": "daily_mean",
        "frequency": "1_hourly",
        "data_format": "netcdf",
        "download_format": "zip",
        "area": bounding_box.as_cds_area(),
    }

    if force_cds_refresh:
        request["nocache"] = "123"

    logger.info("Requesting ERA5-Land %04d-%02d from CDS...", year, month)
    client = _init_cds_client()
    client.retrieve("derived-era5-land-daily-statistics", request, str(out_path))
    logger.info("Downloaded: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    return out_path


def fetch_era5_static_soil(
    *,
    bounding_box: BBoxWGS84 = FRANCE_BBOX,
    static_dir: str | Path = DEFAULT_STATIC_DIR,
    overwrite: bool = False,
    force_cds_refresh: bool = False,
) -> Path:
    """Download one month of hourly ERA5-Land data.

    Parameters
    ----------
    year, month, day : target period
    bounding_box : spatial extent (default: France metropolitan)
    static_dir : output directory for ERA5 downloads
    overwrite : re-download even if file exists
    force_cds_refresh : to avoid cache issues when overwriting

    Returns
    -------
    Path to the downloaded NetCDF file.
    """
    static_dir = Path(static_dir)
    out_path = _era5_soil_static_output_path(static_dir)

    if out_path.exists() and not overwrite:
        logger.info("ERA5-Land static soil already exists:")
        return out_path

    request = {
        "variable": ["soil_type"],
        "download_format": "unarchived",
        "data_format" : "netcdf",
        "area": bounding_box.as_cds_area(),
    }

    if force_cds_refresh:
        request["nocache"] = "123"

    logger.info("Requesting ERA5-Land static soil from CDS...")
    client = _init_cds_client()
    client.retrieve("reanalysis-era5-land", request, str(out_path))
    logger.info("Downloaded: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    return out_path

def open_era5_soil_water(folder):
    """Open ERA5-Land soil water content data from a folder of NetCDF files.
    Requires pre-extraction of the ZIP file downloaded from CDS."""

    folder = Path(folder)

    layers = []

    for path in sorted(folder.glob("*.nc")):
        print(path)
        ds = xr.open_dataset(path)

        # Find which soil-water variable is present
        found = [v for v in SOIL_LAYERS if v in ds.data_vars]

        if len(found) != 1:
            raise ValueError(f"{path.name}: expected exactly one soil-water layer, found {found}")

        var = found[0]
        info = SOIL_LAYERS[var]

        da = ds[var].rename("soil_water").expand_dims(soil_layer=[info["layer"]])

        da = da.assign_coords(
            depth_top_cm=("soil_layer", [info["depth_top_cm"]]),
            depth_bottom_cm=("soil_layer", [info["depth_bottom_cm"]]),
            depth_mid_cm=(
                "soil_layer",
                [(info["depth_top_cm"] + info["depth_bottom_cm"]) / 2],
            ),
        )

        layers.append(da)

    if not layers:
        raise FileNotFoundError(f"No .nc files found in {folder}")

    soil = xr.concat(layers, dim="soil_layer").sortby("soil_layer")

    soil.attrs.update(
        {
            "long_name": "Volumetric soil water",
            "units": "m3 m-3",
        }
    )

    return soil



def remap_era5_layers_to_profile(
    data: xr.DataArray,
    soil_profile: SoilProfile,
    *,
    name : str = "SMI",
) -> xr.DataArray:
    """Remap ERA5-Land processed (or not) to SoilProfile layers.

    ERA5-Land SWVL values represent averages over four fixed soil layers:
        0–7 cm
        7–28 cm
        28–100 cm
        100–289 cm

    Therefore this uses depth-overlap weighting rather than treating the
    values as point measurements at the layer centres.

    If a target layer extends below the deepest ERA5-Land layer, the
    deepest ERA5 value is held constant.

    Parameters
    ----------
    data
        Either an xarray DataArray containing a ``soil_water``, or computed SMI

        Remaining dimensions such as valid_time, latitude and longitude
        are preserved.

    soil_profile
        IrriGator SoilProfile. Its ``z_layers_cm`` defines the target
        layer intervals.

    name
        Name for the output variable in the returned Dataset.

    Returns
    -------
    xr.Dataset
        Dataset containing ``soil_water`` remapped onto ``profile_layer``.
    """
    
    da = data

    if "soil_layer" not in da.dims:
        raise ValueError("ERA5 soil water must have a 'soil_layer' dimension.")

    for coord in ("depth_top_cm", "depth_bottom_cm"):
        if coord not in da.coords:
            raise ValueError(f"ERA5 soil water is missing coordinate {coord!r}.")

    src_top = np.asarray(da["depth_top_cm"].values, dtype=float)
    src_bottom = np.asarray(da["depth_bottom_cm"].values, dtype=float)

    # Ensure source layers are ordered by depth.
    order = np.argsort(src_top)
    src_top = src_top[order]
    src_bottom = src_bottom[order]
    da = da.isel(soil_layer=order)

    target_layers = np.asarray(soil_profile.z_layers_cm, dtype=float)

    if target_layers.ndim != 2 or target_layers.shape[1] != 2:
        raise ValueError("soil_profile.z_layers_cm must contain (top_cm, bottom_cm) pairs.")

    source_min = float(src_top.min())
    source_max = float(src_bottom.max())

    remapped = []

    for layer_idx, (target_top, target_bottom) in enumerate(target_layers):
        if target_bottom <= target_top:
            raise ValueError(f"Invalid target soil layer: {target_top}–{target_bottom} cm.")

        thickness = target_bottom - target_top

        # Depth shared by each ERA5 layer and this target layer.
        overlap = np.maximum(
            0.0,
            np.minimum(target_bottom, src_bottom) - np.maximum(target_top, src_top),
        )

        weights = xr.DataArray(
            overlap,
            dims=("soil_layer",),
            coords={"soil_layer": da["soil_layer"]},
        )

        # Numerator in units theta * cm.
        value = (da * weights).sum("soil_layer")

        covered = float(overlap.sum())

        # Constant extrapolation above the first source layer, if ever needed.
        if target_top < source_min:
            extra = max(
                0.0,
                min(target_bottom, source_min) - target_top,
            )
            if extra > 0:
                value = value + da.isel(soil_layer=0) * extra
                covered += extra

        # Constant extrapolation below ERA5-Land layer 4.
        if target_bottom > source_max:
            extra = max(
                0.0,
                target_bottom - max(target_top, source_max),
            )
            if extra > 0:
                value = value + da.isel(soil_layer=-1) * extra
                covered += extra

        if not np.isclose(covered, thickness):
            raise RuntimeError(
                f"Target layer {target_top}–{target_bottom} cm is not fully "
                f"covered by ERA5 layers ({covered:.2f}/{thickness:.2f} cm)."
            )

        value = value / thickness

        value = value.expand_dims(profile_layer=[layer_idx])

        remapped.append(value)

    data_processed = xr.concat(remapped, dim="profile_layer")

    data_processed = data_processed.assign_coords(
        depth_top_cm=(
            "profile_layer",
            target_layers[:, 0],
        ),
        depth_bottom_cm=(
            "profile_layer",
            target_layers[:, 1],
        ),
        depth_mid_cm=(
            "profile_layer",
            target_layers.mean(axis=1),
        ),
    )

    data_processed.name = name

    return data_processed



def get_era5_soil_type(
    lat: float,
    lon: float,
    static_dir: Path = DEFAULT_STATIC_DIR,
) -> int:
    """Get the ERA5-Land soil type for a given latitude and longitude.

    Parameters
    ----------
    lat : Latitude in decimal degrees
    lon : Longitude in decimal degrees
    static_dir : Directory where the static ERA5-Land soil data is stored

    Returns
    -------
    int : ERA5-Land soil type (1-6)
    """

    static_soil_file = _era5_soil_static_output_path(static_dir)

    if not static_soil_file.exists():
        fetch_era5_static_soil(
            static_dir=static_dir,
        )

    era5_static_soil = xr.open_dataset(static_soil_file)

    soil_type = era5_static_soil["slt"].sel(
        latitude=lat,
        longitude=lon,
        method="nearest",
    ).item()

    return int(soil_type)




def process_era5_soil(
    date: date,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> Path:
    """
    Processes the raw ERA5-Land soil moisture data for the specified date and saves it to the processed directory.
    """
    # 0 - pre-test existence of processed file
    processed_file_path = Path(processed_dir) / "era5_soil" / f"era5_soil_{date.year:04d}_{date.month:02d}_{date.day:02d}.nc"
    processed_file_path.parent.mkdir(parents=True, exist_ok=True)
    if processed_file_path.exists():
        logger.info("Processed ERA5-Land soil moisture data already exists: %s", processed_file_path)
        return processed_file_path
    # Implementation to process raw data and save to processed_dir
    # 1 - Extract the raw data from the ZIP file
    raw_zip_path = _era5_soil_output_path(raw_dir, date.year, date.month, date.day)
    extracted_folder = extract_and_remove_zip(raw_zip_path)
    # 2 - Open the extracted data and concatenate the soil layers
    era5_soil = open_era5_soil_water(extracted_folder)

    #3 - Export the processed data to the processed directory
    
    era5_soil.to_netcdf(processed_file_path)
    logger.info("Processed ERA5-Land soil moisture data saved to: %s", processed_file_path)
    return processed_file_path


def build_initial_water_content_from_era5_land(
    soil_profile: SoilProfile,
    parcel_profile: ParcelConfig,
    date: date,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
    static_dir: Path = DEFAULT_STATIC_DIR,
):
    # Load daily ERA5-Land moisture
    soil_file = (
        Path(processed_dir)
        / "era5_soil"
        / f"era5_soil_{date.year:04d}_{date.month:02d}_{date.day:02d}.nc"
    )

    if not soil_file.exists():
        fetch_era5_land_soil(
            year=date.year,
            month=date.month,
            day=date.day,
            raw_dir=raw_dir,
        )
        process_era5_soil(
            date=date,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )

    era5_soil = xr.open_dataset(soil_file)

    lat = parcel_profile.lat
    lon = parcel_profile.lon

    parcel_soil = era5_soil.sel(
        latitude=lat,
        longitude=lon,
        method="nearest",
    )

    # Ideally average over the requested day if valid_time remains hourly.
    if "valid_time" in parcel_soil.dims:
        parcel_soil = parcel_soil.mean("valid_time")

    # Retrieve the static ERA5 soil type separately.
    soil_type = get_era5_soil_type(
        lat=lat,
        lon=lon,
        static_dir=static_dir,
    )

    hydraulic = ERA5_SOIL_HYDRAULICS[int(soil_type)]

    era5_smi = (parcel_soil["soil_water"] - hydraulic["theta_wp"]) / (
        hydraulic["theta_fc"] - hydraulic["theta_wp"]
    )

    # Conservative first version.
    era5_smi = era5_smi.clip(0.0, 1.0)

    smi_profile = remap_era5_layers_to_profile(
        era5_smi,
        soil_profile,
    )

    theta_wp = xr.DataArray(
        np.asarray(soil_profile.theta_wp, dtype=float),
        dims="profile_layer",
    )

    theta_fc = xr.DataArray(
        np.asarray(soil_profile.theta_fc, dtype=float),
        dims="profile_layer",
    )

    theta_initial = theta_wp + smi_profile * (theta_fc - theta_wp)

    return xr.Dataset(
        {
            "relative_wetness": smi_profile,
            "theta_initial": theta_initial,
        }
    )