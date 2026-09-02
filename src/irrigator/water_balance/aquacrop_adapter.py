"""AquaCrop adapter for IrriGator.

This is the only file that imports ``aquacrop`` directly. All other
modules interact with AquaCrop through the functions defined here.

Replaces the FAO-56 bucket model (Blocks 3-4) with AquaCrop's
mechanistic crop-water model, which couples water stress to canopy
expansion, root growth, biomass accumulation, and yield formation.

The adapter converts IrriGator's data structures (DailyForcing,
SoilProfile, ParcelConfig) to AquaCrop's format, runs the model,
and extracts results in a format compatible with the decision layer.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import multiprocessing as mp
import xarray as xr

from aquacrop import (
    AquaCropModel,
    Crop,
    InitialWaterContent,
    IrrigationManagement,
    Soil,
)

from irrigator.atmospheric.forcing import DailyForcing
from irrigator.config import ParcelConfig
from irrigator.static_layers.soil import SoilProfile
from irrigator.static_layers.terrain import TerrainParams
from irrigator.water_balance.et0 import compute_et0

from irrigator.water_balance.aquacrop_stress import (
    dynamic_water_stress_thresholds,
)

logger = logging.getLogger(__name__)

# AquaCrop's root-development routine searches for the numerical soil
# compartment containing the prospective expansion front.  A soil ending
# exactly at Crop.Zmax can fail that lookup at the lower boundary because
# of floating-point/timestep arithmetic.  Keep one standard compartment
# below the nominal maximum root depth for every crop simulation.
AQUACROP_ROOT_DEPTH_BUFFER_M = 0.10

CROP_ALLOWED_OVERRIDES = {
    "CalendarType",
    "GDDmethod",
    "Tbase",
    "Tupp",
    "Emergence",
    "MaxRooting",
    "Senescence",
    "Maturity",
    "HIstart",
    "Flowering",
    "YldForm",
    "PlantPop",
    "SeedSize",
    "CCx",
    "CGC",
    "CDC",
    "Kcb",
    "WP",
    "WPy",
    "HI0",
    "Zmin",
    "Zmax",
    "fshape_r",
    "SxTopQ",
    "SxBotQ",
    "p_up1",
    "p_up2",
    "p_up3",
    "p_up4",
    "p_lo1",
    "p_lo2",
    "p_lo3",
    "p_lo4",
    "fshape_w1",
    "fshape_w2",
    "fshape_w3",
    "fshape_w4",
}

VARIETY_AQUACROP_PRESETS: dict[str, dict[str, Any]] = {
    "DKC4728": {
        # DKC documentation: GDD thresholds are base 6.
        "CalendarType": 2,  # GDD mode
        "GDDmethod": 2,
        "Tbase": 6,
        "Tupp": 30,
        # Sowing -> flowering
        "HIstart": 970,
        # Sowing -> grain 32% H2O.
        # Practical maturity proxy; not necessarily black-layer maturity.
        "Maturity": 1900,
        # Derived grain/yield formation duration.
        "YldForm": 1900 - 970,
        # Conservative scaling from built-in maize default:
        # default Senescence/Maturity = 1420/1670 ≈ 0.85.
        "Senescence": round(0.85 * 1900),
        "MaxRooting": round(0.85 * 1900),
    }
}

AQUACROPMODEL_HELPER = {
    "crop_growth": pd.DataFrame(
        {
            "variable": [
                "dap",
                "gdd",
                "gdd_cum",
                "z_root",
                "canopy_cover",
                "canopy_cover_ns",
                "biomass",
                "harvest_index",
            ],
            "unit": [
                "days",
                "°C-day",
                "°C-day",
                "m",
                "fraction 0-1",
                "fraction 0-1",
                "kg/ha",
                "fraction 0-1",
            ],
            "meaning": [
                "days after planting",
                "daily gdd increment",
                "cumulative gdd since planting",
                "current root depth",
                "actual green canopy cover",
                "canopy cover under no-stress conditions",
                "above-ground dry biomass",
                "ratio of grain to total biomass",
            ],
        }
    ),
    "water_flux": pd.DataFrame(
        {
            "variable": [
                "IrrDay",
                "Infl",
                "Runoff",
                "DeepPerc",
                "CR",
                "GwIn",
                "Es",
                "EsPot",
                "Tr",
                "TrPot",
                "Wr",
                "z_gw",
                "surface_storage",
            ],
            "unit": ["mm"] * 11 + ["m", "mm"],
            "meaning": [
                "Irrigation Applied today",
                "Infiltration into soil",
                "Surface runoff",
                "Deep percolation below root zone",
                "Capillary rise from groundwater",
                "Groundwater inflow",
                "Soil evaporation",
                "Potential soil evaporation",
                "Crop transpiration",
                "Potential transpiration",
                "Root zone water content",
                "Groundwater table depth",
                "Ponded water on surface",
            ],
        }
    ),
}

# ---------------------------------------------------------------------------
# Data conversion: IrriGator → AquaCrop
# ---------------------------------------------------------------------------


def forcing_to_weather(
    forcing: DailyForcing,
    terrain: TerrainParams,
    lat_deg: float,
) -> pd.DataFrame:
    """Convert DailyForcing + ET0 to AquaCrop weather DataFrame.

    AquaCrop expects columns: MinTemp, MaxTemp, Precipitation,
    ReferenceET, Date — in that order, with Date as datetime64.
    """
    et0 = compute_et0(forcing, terrain, lat_deg)
    if hasattr(et0, "values"):
        et0 = et0.values
    et0 = np.asarray(et0, dtype=np.float64)

    # Ensure MaxTemp > MinTemp (AquaCrop requires this)
    t_min = forcing.t_min.astype(np.float64)
    t_max = forcing.t_max.astype(np.float64)
    t_max = np.maximum(t_max, t_min + 0.1)

    weather = pd.DataFrame(
        {
            "MinTemp": t_min,
            "MaxTemp": t_max,
            "Precipitation": forcing.precip_mm.astype(np.float64),
            "ReferenceET": np.clip(et0, 0.0, None),
            "Date": pd.to_datetime(forcing.dates),
        }
    )

    # Replace NaN with 0 for precipitation and ET0
    weather["Precipitation"] = weather["Precipitation"].fillna(0.0).clip(lower=0.0)
    weather["ReferenceET"] = weather["ReferenceET"].fillna(0.1).clip(lower=0.01)
    weather["MinTemp"] = weather["MinTemp"].ffill().fillna(10.0)
    weather["MaxTemp"] = weather["MaxTemp"].ffill().fillna(20.0)

    return weather


def _profile_theta_s(profile: SoilProfile, i: int) -> float:
    """Return saturation water content for layer i, with a conservative fallback."""
    if hasattr(profile, "theta_s") and profile.theta_s is not None:
        if isinstance(profile.theta_s, list):
            return float(profile.theta_s[i])
        return float(profile.theta_s)
    return min(0.60, float(profile.theta_fc[i]) + 0.12)


def _weighted_profile_interval(
    profile: SoilProfile,
    top_m: float,
    bottom_m: float,
) -> tuple[float, float, float, float]:
    """Depth-weight soil properties over [top_m, bottom_m].

    Returns th_wp, th_fc, th_s, ksat_mm_day. Ksat is averaged arithmetically;
    if drainage becomes too optimistic in layered soils, replace this by a
    harmonic mean for Ksat only.
    """
    top_cm = top_m * 100.0
    bottom_cm = bottom_m * 100.0
    w_sum = 0.0
    th_wp = th_fc = th_s = ksat = 0.0
    for i, (layer_top_cm, layer_bottom_cm) in enumerate(profile.z_layers_cm):
        overlap_cm = max(
            0.0,
            min(bottom_cm, layer_bottom_cm) - max(top_cm, layer_top_cm),
        )
        if overlap_cm <= 0:
            continue

        w_sum += overlap_cm
        th_wp += float(profile.theta_wp[i]) * overlap_cm
        th_fc += float(profile.theta_fc[i]) * overlap_cm
        th_s += _profile_theta_s(profile, i) * overlap_cm
        # SoilProfile stores cm/day; AquaCrop expects mm/day.
        ksat += float(profile.k_sat[i]) * 10.0 * overlap_cm

    if w_sum == 0.0:
        # This can only happen when the requested AquaCrop compartment extends
        # below the available soil data. Do not silently invent a new horizon;
        # reuse the deepest observed layer and let the caller log that choice.
        i = len(profile.z_layers_cm) - 1
        th_wp = float(profile.theta_wp[i])
        th_fc = float(profile.theta_fc[i])
        th_s = _profile_theta_s(profile, i)
        ksat = float(profile.k_sat[i]) * 10.0
    else:
        th_wp /= w_sum
        th_fc /= w_sum
        th_s /= w_sum
        ksat /= w_sum
    # Physical sanity clamps. Keep these minimal; bad soil data should remain visible.
    th_fc = max(th_fc, th_wp + 0.02)
    th_s = max(th_s, th_fc + 0.01)
    ksat = max(ksat, 1.0)
    return th_wp, th_fc, th_s, ksat


def soil_to_aquacrop(
    profile: SoilProfile,
    min_depth_m: float | None = None,
    *,
    dz_comp_m: float = 0.10,
    extrapolate_below_profile: bool = False,
) -> Soil:
    """Convert IrriGator SoilProfile to an AquaCrop Soil object.

    This avoids the fragile ``Soil.add_layer`` path for multi-layer custom soils.
    AquaCrop's ``dz`` argument defines numerical compartments, not agronomic
    horizons. We therefore build uniform 10 cm compartments and write the
    per-compartment hydraulic properties directly into ``soil.profile``.

    Compared with collapsing the profile into one depth-weighted layer, this
    keeps the vertical FC/WP/THS/Ksat signal used for soil water storage,
    drainage, and root-zone extraction, while avoiding artificial layer-boundary
    root barriers.
    """
    data_depth_m = profile.z_layers_cm[-1][1] / 100.0
    requested_depth_m = data_depth_m if min_depth_m is None else max(data_depth_m, min_depth_m)

    if requested_depth_m > data_depth_m and not extrapolate_below_profile:
        logger.warning(
            "AquaCrop soil depth limited to available profile depth %.2fm; "
            "requested %.2fm. Set extrapolate_below_profile=True only if "
            "copying the deepest observed layer below the measured profile is acceptable.",
            data_depth_m,
            requested_depth_m,
        )
        total_depth_m = data_depth_m
    else:
        total_depth_m = requested_depth_m
    n_comp = int(np.ceil(total_depth_m / dz_comp_m))
    dz = [dz_comp_m] * n_comp
    dz[-1] = round(total_depth_m - dz_comp_m * (n_comp - 1), 10)
    if dz[-1] <= 0:
        dz[-1] = dz_comp_m

    soil = Soil("custom", dz=dz)
    prof = soil.profile.copy()

    # Bypass add_layer: fill each numerical compartment explicitly.
    # Keep Layer=1 to avoid treating EU-SoilHydroGrids horizons as mechanical
    # root-penetrability barriers. The hydraulic columns still vary by depth.
    for idx, row in prof.iterrows():
        th_wp, th_fc, th_s, ksat = _weighted_profile_interval(
            profile,
            float(row["z_top"]),
            float(row["zBot"]),
        )
        tau = round(0.0866 * (ksat**0.35), 2)
        tau = min(1.0, max(0.0, tau))

        prof.loc[idx, "Layer"] = 1
        prof.loc[idx, "th_dry"] = th_wp / 2.0
        prof.loc[idx, "th_wp"] = th_wp
        prof.loc[idx, "th_fc"] = th_fc
        prof.loc[idx, "th_s"] = th_s
        prof.loc[idx, "Ksat"] = ksat
        prof.loc[idx, "penetrability"] = 100.0
        prof.loc[idx, "tau"] = tau
        # Required by AquaCrop's SoilProfile jit object; only used for capillary rise.
        prof.loc[idx, "aCR"] = 0.0
        prof.loc[idx, "bCR"] = 0.0

    required = ["Layer", "th_dry", "th_wp", "th_fc", "th_s", "Ksat", "penetrability", "tau"]
    if prof[required].isna().any().any():
        raise ValueError(f"AquaCrop soil profile contains NaNs:\n{prof}")

    soil.profile = prof
    soil.nLayer = 1
    soil.zSoil = round(float(sum(dz)), 2)
    soil.nComp = len(dz)

    logger.info(
        "AquaCrop soil: %.2fm depth, %d x %.2fm compartments, "
        "FC range %.3f-%.3f, WP range %.3f-%.3f, Ksat range %.0f-%.0f mm/day",
        soil.zSoil,
        soil.nComp,
        dz_comp_m,
        float(prof["th_fc"].min()),
        float(prof["th_fc"].max()),
        float(prof["th_wp"].min()),
        float(prof["th_wp"].max()),
        float(prof["Ksat"].min()),
        float(prof["Ksat"].max()),
    )

    return soil


def _soil_for_crop(
    profile: SoilProfile,
    crop: Crop,
    *,
    root_buffer_m: float = AQUACROP_ROOT_DEPTH_BUFFER_M,
) -> Soil:
    """Build an AquaCrop soil deep enough for root-development lookups.

    AquaCrop may evaluate the prospective root-expansion front very close to
    ``Crop.Zmax`` and then locate the containing compartment using
    ``prof.dzsum >= ZiTmp``.  Ending the numerical soil exactly at ``Zmax`` is
    therefore fragile.  We consistently provide one extra 10-cm compartment
    (by default) below the nominal maximum rooting depth.

    The deepest observed IrriGator hydraulic layer is extrapolated only when
    the source soil profile is shallower than this numerical depth, matching
    the behaviour already used by :func:`run_aquacrop`.
    """
    root_buffer_m = max(0.0, float(root_buffer_m))
    requested_depth_m = float(crop.Zmax) + root_buffer_m
    observed_depth_m = float(profile.z_layers_cm[-1][1]) / 100.0
    if requested_depth_m > observed_depth_m + 1e-9:
        logger.warning(
            "Extending AquaCrop numerical soil from observed %.2fm to %.2fm "
            "so roots with Zmax=%.2fm have a safe terminal compartment; "
            "the deepest observed hydraulic layer is extrapolated below %.2fm.",
            observed_depth_m,
            requested_depth_m,
            float(crop.Zmax),
            observed_depth_m,
        )

    soil = soil_to_aquacrop(
        profile,
        min_depth_m=requested_depth_m,
        extrapolate_below_profile=True,
    )

    available_depth_m = float(soil.profile["dz"].sum())
    if available_depth_m + 1e-9 < requested_depth_m:
        raise ValueError(
            "AquaCrop soil is shallower than the required crop root domain: "
            f"available={available_depth_m:.3f} m, "
            f"required={requested_depth_m:.3f} m "
            f"(Zmax={float(crop.Zmax):.3f} m)."
        )

    logger.debug(
        "AquaCrop crop/soil depth guard: crop Zmax=%.2fm, numerical soil=%.2fm (buffer=%.2fm)",
        float(crop.Zmax),
        available_depth_m,
        root_buffer_m,
    )
    return soil


def _mmdd(value: str | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%m/%d")


def _plants_per_ha(crop_cfg: dict[str, Any]) -> int:
    """Return AquaCrop PlantPop, i.e. established plants/ha."""

    if crop_cfg.get("plant_population") is not None:
        return int(crop_cfg["plant_population"])

    if crop_cfg.get("sowing_density") is not None:
        emergence_rate = float(crop_cfg.get("emergence_rate", 1.0))
        if not 0 < emergence_rate <= 1:
            raise ValueError(f"Invalid emergence_rate: {emergence_rate}")
        return round(int(crop_cfg["sowing_density"]) * emergence_rate)

    # Backward compatibility with your current YAML.
    # Your comment says seed/ha, so use emergence_rate if provided.
    if crop_cfg.get("density") is not None:
        emergence_rate = float(crop_cfg.get("emergence_rate", 1.0))
        if not 0 < emergence_rate <= 1:
            raise ValueError(f"Invalid emergence_rate: {emergence_rate}")
        return round(int(crop_cfg["density"]) * emergence_rate)

    return 75_000


def _validate_aquacrop_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    unknown = set(overrides) - CROP_ALLOWED_OVERRIDES
    if unknown:
        raise ValueError(
            "Unknown AquaCrop crop override(s): "
            f"{sorted(unknown)}. Use AquaCrop Crop attribute names."
        )
    return overrides


def _crop_overrides_from_config(crop_cfg: dict[str, Any]) -> dict[str, Any]:
    variety = str(crop_cfg.get("variety", "")).upper()

    overrides: dict[str, Any] = {}

    # 1. Variety preset from documentation.
    if variety in VARIETY_AQUACROP_PRESETS:
        overrides.update(VARIETY_AQUACROP_PRESETS[variety])

    # 2. Generic YAML GDD thresholds override the preset if present.
    gdd = crop_cfg.get("gdd_thresholds") or {}

    if gdd.get("base") is not None:
        overrides["Tbase"] = float(gdd["base"])

    if gdd.get("flowering") is not None:
        overrides["HIstart"] = int(gdd["flowering"])

    maturity = gdd.get("maturity") or gdd.get("harvest_32pct_h2o") or gdd.get("harvest")
    if maturity is not None:
        overrides["Maturity"] = int(maturity)

    if "HIstart" in overrides and "Maturity" in overrides:
        yld_form = int(overrides["Maturity"] - overrides["HIstart"])
        if yld_form <= 0:
            raise ValueError(
                f"Invalid phenology: Maturity={overrides['Maturity']} "
                f"must be greater than HIstart={overrides['HIstart']}."
            )
        overrides["YldForm"] = yld_form
        overrides.setdefault("Senescence", round(0.85 * overrides["Maturity"]))
        overrides.setdefault("MaxRooting", overrides["Senescence"])

    rooting = crop_cfg.get("rooting_depth") or {}

    if rooting.get("at_emergence_m") is not None:
        overrides["Zmin"] = float(rooting["at_emergence_m"])

    if rooting.get("max_m") is not None:
        overrides["Zmax"] = float(rooting["max_m"])
    # 3. Plant population.
    overrides["PlantPop"] = _plants_per_ha(crop_cfg)

    # 4. Manual expert overrides have final priority.
    overrides.update(crop_cfg.get("aquacrop_overrides") or {})

    return _validate_aquacrop_overrides(overrides)


def parcel_to_crop(parcel: ParcelConfig) -> Crop:
    """Convert parcel crop config to AquaCrop Crop object."""

    crop_cfg = parcel.crop
    crop_type = crop_cfg.get("type", "grain_maize")

    if crop_type not in {"grain_maize", "maize"}:
        raise ValueError(
            f"Unsupported AquaCrop crop: {crop_type}. Currently only grain_maize is supported."
        )

    planting_mmdd = _mmdd(crop_cfg["planting_date"])
    harvest_mmdd = _mmdd(crop_cfg.get("expected_harvest"))

    overrides = _crop_overrides_from_config(crop_cfg)

    name = "MaizeGDD" if overrides["Maturity"] else "Maize"
    print(name)

    return Crop(
        name,
        planting_date=planting_mmdd,
        harvest_date=harvest_mmdd,
        **overrides,
    )


def parcel_to_irrigation(parcel: ParcelConfig) -> IrrigationManagement:
    """Convert parcel irrigation log to AquaCrop IrrigationManagement.

    Uses method 3 (predefined schedule) if the farmer has logged
    irrigation events, otherwise rainfed (method 0).
    """
    log = parcel.irrigation_log
    if not log:
        return IrrigationManagement(irrigation_method=0)

    rows = []
    for item in log:
        rows.append(
            {
                "Date": pd.Timestamp(item["date"]),
                "Depth": float(item["amount_mm"]),
            }
        )

    schedule = pd.DataFrame(rows, columns=["Date", "Depth"])

    max_dose = parcel.irrigation.get("max_dose_mm", 40.0)

    return IrrigationManagement(
        irrigation_method=3,
        Schedule=schedule,
        MaxIrr=max_dose,
    )



def soil_water_profile_to_aquacrop_iwc(
    soil_water_profile: xr.Dataset,
    aquacrop_soil: Soil,
) -> InitialWaterContent:
    """Convert an IrriGator initial soil-water profile to AquaCrop IWC.

    ``soil_water_profile`` is the output of
    build_initial_soil_water_profile_from_era5_land().

    Its theta_initial values are layer means over the original SoilProfile
    horizons. They are remapped by depth overlap onto AquaCrop's numerical
    compartments, then supplied at the exact compartment midpoints.
    """
    if "theta_initial" not in soil_water_profile:
        raise ValueError("soil_water_profile must contain 'theta_initial'.")

    for coord in ("depth_top_cm", "depth_bottom_cm"):
        if coord not in soil_water_profile.coords:
            raise ValueError(f"soil_water_profile is missing {coord!r}.")

    theta = np.asarray(
        soil_water_profile["theta_initial"].values,
        dtype=float,
    )

    src_top = (
        np.asarray(
            soil_water_profile["depth_top_cm"].values,
            dtype=float,
        )
        / 100.0
    )

    src_bottom = (
        np.asarray(
            soil_water_profile["depth_bottom_cm"].values,
            dtype=float,
        )
        / 100.0
    )

    if theta.ndim != 1:
        raise ValueError("theta_initial must be one-dimensional over profile_layer.")

    if not (len(theta) == len(src_top) == len(src_bottom)):
        raise ValueError("theta_initial and depth coordinates have incompatible lengths.")

    if not np.all(np.isfinite(theta)):
        raise ValueError("theta_initial contains non-finite values.")

    # AquaCrop numerical compartments.
    prof = aquacrop_soil.profile

    comp_top = prof["z_top"].to_numpy(dtype=float)
    comp_bottom = prof["zBot"].to_numpy(dtype=float)
    comp_mid = (comp_top + comp_bottom) / 2.0

    src_min = float(src_top.min())
    src_max = float(src_bottom.max())

    theta_comp = np.empty(len(prof), dtype=float)

    for j, (top, bottom) in enumerate(zip(comp_top, comp_bottom, strict=True)):
        thickness = bottom - top

        overlap = np.maximum(
            0.0,
            np.minimum(bottom, src_bottom) - np.maximum(top, src_top),
        )

        numerator = float(np.sum(theta * overlap))
        covered = float(overlap.sum())

        # Constant extrapolation if the numerical AquaCrop soil extends
        # outside the available initial-water profile.
        if top < src_min:
            extra = max(
                0.0,
                min(bottom, src_min) - top,
            )
            numerator += float(theta[0]) * extra
            covered += extra

        if bottom > src_max:
            extra = max(
                0.0,
                bottom - max(top, src_max),
            )
            numerator += float(theta[-1]) * extra
            covered += extra

        if not np.isclose(covered, thickness):
            raise RuntimeError(
                f"Could not map initial water content onto "
                f"AquaCrop compartment {top:.2f}-{bottom:.2f} m."
            )

        theta_comp[j] = numerator / thickness

    # Current ERA5-SMI transformation is constrained to WP <= theta <= FC.
    th_wp = prof["th_wp"].to_numpy(dtype=float)
    th_fc = prof["th_fc"].to_numpy(dtype=float)

    tol = 1e-6
    if np.any(theta_comp < th_wp - tol):
        raise ValueError("Mapped initial water content is below AquaCrop wilting point.")

    if np.any(theta_comp > th_fc + tol):
        raise ValueError(
            "Mapped initial water content is above AquaCrop field capacity. "
            "This is unexpected while ERA5 SMI is clipped to [0, 1]."
        )

    return InitialWaterContent(
        wc_type="Num",
        method="Depth",
        depth_layer=comp_mid.tolist(),
        value=theta_comp.tolist(),
    )



# ---------------------------------------------------------------------------
# AquaCrop runner
# ---------------------------------------------------------------------------


@dataclass
class AquaCropResult:
    """Results from a single AquaCrop simulation."""

    # Raw AquaCrop outputs
    final_results: pd.DataFrame  # season summary (yield, irrigation)
    crop_growth: pd.DataFrame  # daily: canopy, biomass, root depth, GDD
    water_flux: pd.DataFrame  # daily: Tr, TrPot, Es, IrrDay, Wr, DeepPerc
    water_storage: pd.DataFrame  # daily: soil moisture per compartment
    weather_daily: pd.DataFrame | None = None
    stress_thresholds: pd.DataFrame | None = None

    # Derived daily stress metric (Tr/TrPot, equivalent to Ks)
    @property
    def daily_stress(self) -> pd.DataFrame:
        """Daily crop-water diagnostics derived from AquaCrop outputs."""

        cg = self.crop_growth
        wf = self.water_flux

        n = min(len(cg), len(wf))

        ks = np.where(
            wf["TrPot"].values[:n] > 0.01,
            wf["Tr"].values[:n] / wf["TrPot"].values[:n],
            1.0,
        )

        # Actual forcing used by AquaCrop
        if self.weather_daily is not None:
            precip = self.weather_daily["Precipitation"].to_numpy(dtype=float)[:n]

            et0 = self.weather_daily["ReferenceET"].to_numpy(dtype=float)[:n]

        else:
            # Fallback for older AquaCropResult objects.
            precip = np.maximum(
                wf["Infl"].to_numpy(dtype=float)[:n] - wf["IrrDay"].to_numpy(dtype=float)[:n],
                0.0,
            )
            et0 = np.full(n, np.nan)

        result = pd.DataFrame(
            {
                "dap": cg["dap"].values[:n],
                "gdd_cum": cg["gdd_cum"].values[:n],
                "ks": ks,
                "canopy_cover": cg["canopy_cover"].values[:n],
                "canopy_cover_ns": cg["canopy_cover_ns"].values[:n],
                "biomass_kg_ha": (cg["biomass"].values[:n] * 10.0),
                "z_root_m": cg["z_root"].values[:n],
                "harvest_index": cg["harvest_index"].values[:n],
                "precip_mm": precip,
                "et0_mm": et0,
                "irrigation_mm": wf["IrrDay"].values[:n],
                "tr_mm": wf["Tr"].values[:n],
                "tr_pot_mm": wf["TrPot"].values[:n],
                "es_mm": wf["Es"].values[:n],
                "deep_perc_mm": wf["DeepPerc"].values[:n],
                "wr_mm": wf["Wr"].values[:n],
            }
        )

        # Add AquaCrop dynamic thresholds
        if self.stress_thresholds is not None:
            if len(self.stress_thresholds) < n:
                raise ValueError("stress_thresholds is shorter than AquaCrop outputs.")

            thresholds = self.stress_thresholds.iloc[:n].reset_index(drop=True)

            for column in thresholds.columns:
                result[column] = thresholds[column].to_numpy(dtype=float)

        return result


def _prepare_aquacrop_init(
    weather: pd.DataFrame,
    crop: Crop,
    sim_start: date,
    sim_end: date,
) -> tuple[pd.DataFrame, str, int]:
    """Extend weather and compute init_end for AquaCrop calendar init.

    AquaCrop's ``compute_variables`` calls ``compute_crop_calendar``
    unconditionally.  In GDD mode (``CalendarType=2``), this scans the
    weather for the calendar day when GDD reaches ``Maturity``.  If the
    simulation period is shorter than a full growing season, the
    assertion fails.

    Fix: set ``sim_end`` to the harvest date for AquaCrop's constructor
    (so the calendar computation succeeds), extend the weather with
    synthetic climatological fill, and run only ``actual_sim_days``
    timesteps.  The synthetic rows are consumed only by the calendar
    init — never by the daily crop simulation.

    Returns
    -------
    weather_ext : extended weather DataFrame
    init_end_str : sim_end string for AquaCropModel constructor
    actual_sim_days : number of real forcing days to simulate
    """
    actual_sim_days = (sim_end - sim_start).days

    harvest_date_str = crop.harvest_date
    if harvest_date_str is not None:
        init_end = pd.Timestamp(f"{sim_start.year}/{harvest_date_str}")
        if init_end <= pd.Timestamp(sim_start):
            init_end = pd.Timestamp(f"{sim_start.year + 1}/{harvest_date_str}")
        extend_to = init_end + pd.Timedelta(days=30)
        last_weather_date = weather["Date"].iloc[-1]
        if extend_to > last_weather_date:
            extra_dates = pd.date_range(
                last_weather_date + pd.Timedelta(days=1), extend_to, freq="D"
            )
            fill = pd.DataFrame(
                {
                    "MinTemp": weather["MinTemp"].tail(30).mean(),
                    "MaxTemp": weather["MaxTemp"].tail(30).mean(),
                    "Precipitation": 0.0,
                    "ReferenceET": weather["ReferenceET"].tail(30).mean(),
                    "Date": extra_dates,
                }
            )
            weather = pd.concat([weather, fill], ignore_index=True)
            logger.debug(
                "Extended weather to %s (+%d synthetic days) for crop calendar init",
                extend_to.date(),
                len(extra_dates),
            )
        init_end_str = init_end.strftime("%Y/%m/%d")
    else:
        init_end_str = sim_end.strftime("%Y/%m/%d")

    return weather, init_end_str, actual_sim_days


def _trim_aquacrop_outputs(
    model: AquaCropModel,
    actual_sim_days: int,
) -> AquaCropResult:
    """Extract AquaCrop results trimmed to actual simulation days.

    Discards rows produced from synthetic weather beyond the real
    forcing boundary.  Uses ``time_step_counter`` for data-driven
    trimming rather than fixed offsets.
    """
    cg = model.get_crop_growth()
    wf = model.get_water_flux()
    ws = model.get_water_storage()

    last_nonzero = max(i for i, v in enumerate(cg["time_step_counter"]) if v != 0)

    valid_mask = [
        (i <= last_nonzero and i <= actual_sim_days) for i in range(len(cg["time_step_counter"]))
    ]

    return AquaCropResult(
        final_results=model.get_simulation_results(),
        crop_growth=cg[valid_mask].reset_index(drop=True),
        water_flux=wf[valid_mask].reset_index(drop=True),
        water_storage=ws[valid_mask].reset_index(drop=True),
    )


def run_aquacrop(
    forcing: DailyForcing,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    sim_end: date,
    *,
    irrigation_management: IrrigationManagement | None = None,
    initial_water_content: InitialWaterContent | None = None,
    initial_soil_water_profile: xr.Dataset | None = None,
) -> AquaCropResult:
    """Run AquaCrop simulation.

    Parameters
    ----------
    forcing : daily atmospheric forcing (from extract_parcel_forcing)
    parcel : parcel configuration (crop, irrigation log)
    terrain : terrain parameters (for ET0 computation)
    soil_profile : soil hydraulic properties per layer
    sim_start, sim_end : simulation period. Can be a partial year
        (e.g., Jan 1 → Jul 15 for operational use). AquaCrop handles
        this correctly — crop growth starts at planting_date regardless.
    irrigation_management : override irrigation (default: from parcel log)
    initial_water_content : override initial soil moisture
        (default: field capacity)
    initial_soil_water_profile : compute InitialWaterContent from this ERA5-SMI-based profile
        (default: None)
    """
    weather = forcing_to_weather(forcing, terrain, parcel.lat)

    crop = parcel_to_crop(parcel)
    soil = _soil_for_crop(soil_profile, crop)

    weather, init_end_str, actual_sim_days = _prepare_aquacrop_init(
        weather=weather,
        crop=crop,
        sim_start=sim_start,
        sim_end=sim_end,
    )

    if (
        initial_water_content is not None
        and initial_soil_water_profile is not None
    ):
        raise ValueError(
            "Provide either initial_water_content or "
            "initial_soil_water_profile, not both."
        )

    if initial_water_content is not None:
        iwc = initial_water_content
    elif initial_soil_water_profile is not None:
        iwc = soil_water_profile_to_aquacrop_iwc(
            initial_soil_water_profile,
            soil,
        )
    else:
        iwc = InitialWaterContent(value=["FC"])



    irr = irrigation_management or parcel_to_irrigation(parcel)

    model = AquaCropModel(
        sim_start_time=sim_start.strftime("%Y/%m/%d"),
        sim_end_time=init_end_str,
        weather_df=weather,
        soil=soil,
        crop=crop,
        initial_water_content=iwc,

        irrigation_management=irr,
    )

    # Let AquaCrop run — it will stop at crop maturity or sim_end (harvest).
    # We then trim outputs to the actual days with real forcing.
    # model.run_model(till_termination=True)
    model.run_model(till_termination=True)

    result = _trim_aquacrop_outputs(model, actual_sim_days=actual_sim_days)

    output_days = result.crop_growth["time_step_counter"].to_numpy(dtype=int)

    if len(output_days):
        if output_days.min() < 0 or output_days.max() >= len(weather):
            raise RuntimeError("Cannot align AquaCrop output rows with weather DataFrame.")

        result.weather_daily = weather.iloc[output_days][
            [
                "Date",
                "Precipitation",
                "ReferenceET",
            ]
        ].reset_index(drop=True)

        result.stress_thresholds = dynamic_water_stress_thresholds(
            crop,
            et0=result.weather_daily["ReferenceET"].to_numpy(dtype=float),
        )

    n_days = len(result.crop_growth)
    logger.info(
        "AquaCrop run: %s → %s (%d days), GDD=%.0f, CC=%.2f, biomass=%.0f kg/ha",
        sim_start,
        sim_end,
        n_days,
        result.crop_growth["gdd_cum"].iloc[-2] if n_days > 1 else 0,
        result.crop_growth["canopy_cover"].iloc[-2] if n_days > 1 else 0,
        result.crop_growth["biomass"].iloc[-2] if n_days > 1 else 0,
    )

    return result


# ---------------------------------------------------------------------------
# Ensemble forecast with AquaCrop
# ---------------------------------------------------------------------------


@dataclass
class AquaCropEnsembleStats:
    """Per-day ensemble statistics from AquaCrop forecast runs."""

    date: date
    day_offset: int
    source: str  # "historical" or "ifs_ens"
    # Stress (Ks equivalent = Tr/TrPot)
    ks_mean: float
    ks_median: float
    ks_p25: float
    ks_p75: float
    ks_min: float
    ks_max: float
    # Canopy cover
    cc_mean: float
    # Water content
    wr_mean: float
    # Precipitation
    precip_mean: float
    # Members
    n_members_stressed: int
    n_members_total: int


def run_ensemble_aquacrop(
    historical_forcing: DailyForcing,
    arome_forcing: DailyForcing,
    member_forcings: dict[int, DailyForcing],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    today: date,
    *,
    stress_threshold: float = 0.9,
    initial_soil_water_profile: xr.Dataset | None = None,
) -> list[AquaCropEnsembleStats]:
    """Run AquaCrop for each IFS ENS member and compute ensemble statistics.

    Optimized: weather DataFrames (including ET₀) are pre-computed once
    for the historical+AROME portion and once per member for the forecast
    portion, rather than redundantly inside each ``run_aquacrop`` call.

    Parameters
    ----------
    historical_forcing : DailyForcing from Jan 1 → today
    arome_forcing : DailyForcing from today+1 -> today+2
    member_forcings : {member_id: DailyForcing} for each IFS ENS member
    parcel : parcel configuration
    terrain : terrain parameters
    soil_profile : soil hydraulic properties
    sim_start : simulation start (typically Jan 1)
    today : current date (forecast branches from here)
    stress_threshold : Ks below this counts as "stressed"
    initial_soil_water_profile : ERA5-SMI-based soil water profile for AquaCrop IWC
    """
    daily_stats: list[AquaCropEnsembleStats] = []

    # Pre-compute shared objects

    crop = parcel_to_crop(parcel)
    soil = _soil_for_crop(soil_profile, crop)

    iwc = InitialWaterContent(value=["FC"]) if initial_soil_water_profile is None else soil_water_profile_to_aquacrop_iwc(
        initial_soil_water_profile,
        soil,
    )

    # Base weather: historical + AROME (ET₀ computed once)
    deter_forcing = historical_forcing.concat(arome_forcing)
    base_weather = forcing_to_weather(deter_forcing, terrain, parcel.lat)

    # Determine forecast length from first member
    first_forcing = next(iter(member_forcings.values()))
    n_forecast = first_forcing.n_days

    logger.info(
        "Running AquaCrop ensemble: %d members, %s → %s "
        "(weather pre-computed, %d hist + %d forecast days)",
        len(member_forcings),
        sim_start,
        pd.Timestamp(first_forcing.dates[-1]).date(),
        deter_forcing.n_days,
        n_forecast,
    )

    # Pre-compute per-member weather
    member_weathers: dict[int, pd.DataFrame] = {}
    member_end_dates: dict[int, date] = {}
    for m_id, m_forcing in member_forcings.items():
        m_weather = forcing_to_weather(m_forcing, terrain, parcel.lat)
        full = (
            pd.concat([base_weather, m_weather])
            .drop_duplicates("Date", keep="first")
            .sort_values("Date")
            .reset_index(drop=True)
        )
        member_weathers[m_id] = full
        member_end_dates[m_id] = pd.Timestamp(m_forcing.dates[-1]).date()

    # Run each member
    all_member_ks: dict[int, np.ndarray] = {}
    all_member_cc: dict[int, np.ndarray] = {}
    all_member_wr: dict[int, np.ndarray] = {}
    all_member_precip: dict[int, np.ndarray] = {}
    member_dates = None

    for member_id in member_weathers:
        weather = member_weathers[member_id]
        full_end = member_end_dates[member_id]

        logger.debug("Member %d: %s → %s", member_id, sim_start, full_end)

        # Extend weather for crop calendar init and get init_end
        weather_ext, init_end_str, actual_sim_days = _prepare_aquacrop_init(
            weather,
            crop,
            sim_start,
            full_end,
        )

        model = AquaCropModel(
            sim_start_time=sim_start.strftime("%Y/%m/%d"),
            sim_end_time=init_end_str,
            weather_df=weather_ext,
            soil=soil,
            crop=crop,
            initial_water_content=iwc,
            irrigation_management=parcel_to_irrigation(parcel),
        )

        model.run_model(till_termination=True)

        # Trim to real-forcing days
        result = _trim_aquacrop_outputs(model, actual_sim_days)
        stress = result.daily_stress

        # Extract forecast portion (last n_forecast days)
        # Drop terminal zero rows
        n_valid = len(stress)

        forecast_start_idx = max(0, n_valid - n_forecast)
        forecast_slice = stress.iloc[forecast_start_idx:n_valid]

        all_member_ks[member_id] = forecast_slice["ks"].values
        all_member_cc[member_id] = forecast_slice["canopy_cover"].values
        all_member_wr[member_id] = forecast_slice["wr_mm"].values
        all_member_precip[member_id] = forecast_slice["precip_mm"].values

        if member_dates is None:
            member_dates = pd.to_datetime(first_forcing.dates[: len(forecast_slice)])

    # Aggregate ensemble statistics per day
    n_days = min(len(v) for v in all_member_ks.values())

    for d in range(n_days):
        ks_arr = np.array([all_member_ks[m][d] for m in all_member_ks])
        cc_arr = np.array([all_member_cc[m][d] for m in all_member_cc])
        wr_arr = np.array([all_member_wr[m][d] for m in all_member_wr])
        pr_arr = np.array([all_member_precip[m][d] for m in all_member_precip])

        day_date = member_dates[d] if d < len(member_dates) else None

        daily_stats.append(
            AquaCropEnsembleStats(
                date=day_date,
                day_offset=d,
                source="ifs_ens",
                ks_mean=float(np.nanmean(ks_arr)),
                ks_median=float(np.nanmedian(ks_arr)),
                ks_p25=float(np.nanpercentile(ks_arr, 25)),
                ks_p75=float(np.nanpercentile(ks_arr, 75)),
                ks_min=float(np.nanmin(ks_arr)),
                ks_max=float(np.nanmax(ks_arr)),
                cc_mean=float(np.nanmean(cc_arr)),
                wr_mean=float(np.nanmean(wr_arr)),
                precip_mean=float(np.nanmean(pr_arr)),
                n_members_stressed=int((ks_arr < stress_threshold).sum()),
                n_members_total=len(ks_arr),
            )
        )

    logger.info(
        "Ensemble complete: %d days, %d members, stress_days=%d",
        len(daily_stats),
        len(member_forcings),
        sum(1 for s in daily_stats if s.n_members_stressed > len(member_forcings) // 2),
    )

    return daily_stats


# ---------------------------------------------------------------------------
# Bridge: AquaCrop outputs → decision layer format
# ---------------------------------------------------------------------------
from irrigator.water_balance.state import WaterBalanceState


def aquacrop_to_current_state(
    hist_run: AquaCropResult,
    soil_profile: SoilProfile,
    today: date,
) -> WaterBalanceState:
    """Extract current state from AquaCrop historical run.

    Converts AquaCrop's last-day outputs into a WaterBalanceState
    compatible with the existing decision layer (compute_recommendation).
    """

    stress = hist_run.daily_stress
    wf = hist_run.water_flux
    cg = hist_run.crop_growth

    # Use second-to-last row (last row is often a zero/terminal row)
    idx = -2 if len(stress) > 1 else -1

    z_root = float(stress["z_root_m"].iloc[idx])
    if z_root <= 0:
        z_root = 0.1  # minimum

    # Compute TAW and depletion from root zone water content
    # Weighted average FC/WP over root depth
    total_depth_cm = soil_profile.z_layers_cm[-1][1]
    total_thick = sum(b - t for t, b in soil_profile.z_layers_cm)
    avg_fc, avg_wp = 0.0, 0.0
    for i, (top, bot) in enumerate(soil_profile.z_layers_cm):
        w = (bot - top) / total_thick
        avg_fc += soil_profile.theta_fc[i] * w
        avg_wp += soil_profile.theta_wp[i] * w

    taw = (avg_fc - avg_wp) * z_root * 1000  # mm
    wr = float(stress["wr_mm"].iloc[idx])
    fc_content = avg_fc * z_root * 1000  # water at FC in mm
    depletion = max(0.0, fc_content - wr)

    ks = float(stress["ks"].iloc[idx])
    gdd = float(stress["gdd_cum"].iloc[idx])
    cc = float(stress["canopy_cover"].iloc[idx])
    dap = float(stress["dap"].iloc[idx])

    # Estimate Kc from canopy cover (Beer's law)
    kc = 1.2 * (1.0 - np.exp(-0.65 * cc * 10)) if cc > 0.01 else 0.0
    kc = min(1.2, max(0.0, kc))

    # Crop stage from GDD
    if dap <= 0:
        stage = "not_planted"
    elif gdd < 80:
        stage = "pre_emergence"
    elif gdd < 300:
        stage = "initial"
    elif gdd < 700:
        stage = "development"
    elif gdd < 1300:
        stage = "midseason"
    elif gdd < 1700:
        stage = "mature"
    else:
        stage = "harvest"

    # p_adj for RAW
    etc_act = float(stress["tr_mm"].iloc[idx]) + float(stress["es_mm"].iloc[idx])
    p = 0.55  # maize default
    p_adj = p + 0.04 * (5.0 - etc_act)
    p_adj = max(0.1, min(0.8, p_adj))
    raw = p_adj * taw

    et0 = float(stress["tr_pot_mm"].iloc[idx]) / max(kc, 0.01) if kc > 0.01 else 0.0

    return WaterBalanceState(
        date=today,
        et0=et0,
        kc=kc,
        etc_pot=float(stress["tr_pot_mm"].iloc[idx]),
        etc_act=etc_act,
        precip=float(stress["precip_mm"].iloc[idx]),
        irrigation=float(stress["irrigation_mm"].iloc[idx]),
        runoff=0.0,
        drainage=float(stress["deep_perc_mm"].iloc[idx]),
        depletion=depletion,
        taw=taw,
        raw=raw,
        stress_coeff=ks,
        z_root=z_root,
        gdd=gdd,
        crop_stage=stage,
    )


from irrigator.forecasts.short_term import (
    DailyEnsembleStats,
    EnsembleStressReport,
)


def build_blended_stress_report(
    arome_run: AquaCropResult,
    ensemble_stats: list[AquaCropEnsembleStats],
    today: date,
    arome_days: int = 2,
) -> EnsembleStressReport:
    """Build an EnsembleStressReport blending AROME + IFS ENS.

    Days 0 to arome_days-1: from AROME deterministic AquaCrop run.
    Days arome_days onwards: from IFS ENS ensemble AquaCrop runs.

    Returns an EnsembleStressReport compatible with compute_recommendation().
    """

    daily_stats = []

    # AROME deterministic days (days 0 to arome_days-1)
    arome_stress = arome_run.daily_stress
    # Find the rows corresponding to today onwards
    arome_wf = arome_run.water_flux
    n_total = len(arome_stress)
    # Skip the terminal zero row
    if n_total > 1 and arome_stress["dap"].iloc[-1] == 0:
        n_total -= 1
    # The last arome_days rows before the terminal row are the forecast
    arome_forecast_start = max(0, n_total - arome_days)

    for d in range(min(arome_days, n_total - arome_forecast_start)):
        row_idx = arome_forecast_start + d
        if row_idx >= len(arome_stress):
            break
        ks = float(arome_stress["ks"].iloc[row_idx])
        day_date = today + pd.Timedelta(days=d)

        daily_stats.append(
            DailyEnsembleStats(
                date=day_date,
                day_offset=d,
                source="arome",
                ks_mean=ks,
                ks_median=ks,
                ks_p25=ks,
                ks_p75=ks,
                ks_min=ks,
                ks_max=ks,
                depletion_mean=0.0,
                depletion_p25=0.0,
                depletion_p75=0.0,
                precip_mean=float(arome_stress["precip_mm"].iloc[row_idx]),
                precip_p25=float(arome_stress["precip_mm"].iloc[row_idx]),
                precip_p75=float(arome_stress["precip_mm"].iloc[row_idx]),
                etc_mean=float(arome_stress["tr_mm"].iloc[row_idx]),
                n_members_stressed=1 if ks < 0.9 else 0,
                n_members_total=1,
            )
        )

    # IFS ENS ensemble days (offset day_offset by arome_days for continuity)
    for stat in ensemble_stats:
        daily_stats.append(
            DailyEnsembleStats(
                date=stat.date,
                day_offset=arome_days + stat.day_offset,
                source="ifs_ens",
                ks_mean=stat.ks_mean,
                ks_median=stat.ks_median,
                ks_p25=stat.ks_p25,
                ks_p75=stat.ks_p75,
                ks_min=stat.ks_min,
                ks_max=stat.ks_max,
                depletion_mean=0.0,
                depletion_p25=0.0,
                depletion_p75=0.0,
                precip_mean=stat.precip_mean,
                precip_p25=stat.precip_mean,
                precip_p75=stat.precip_mean,
                etc_mean=0.0,
                n_members_stressed=stat.n_members_stressed,
                n_members_total=stat.n_members_total,
            )
        )

    n_ens = ensemble_stats[0].n_members_total if ensemble_stats else 0

    return EnsembleStressReport(
        daily_stats=daily_stats,
        n_members=n_ens,
        arome_days=arome_days,
        ens_days=len(daily_stats) - arome_days,
    )


# ---------------------------------------------------------------------------
# Ensemble irrigation optimizer (Block 6)
# ---------------------------------------------------------------------------


@dataclass
class IrrigationCandidate:
    """One candidate irrigation schedule to evaluate.

    Supports 0–N irrigation events.  ``events`` is a list of
    ``(day_offset, dose_mm)`` tuples, where ``day_offset`` is the
    number of days from today (0 = today).
    """

    events: list[tuple[int, float]]  # [(day_offset, dose_mm), ...]
    label: str  # human-readable label

    @property
    def total_mm(self) -> float:
        return sum(dose for _, dose in self.events)

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def is_rainfed(self) -> bool:
        return len(self.events) == 0

    # Convenience for single-event backward compat
    @property
    def day_offset(self) -> int:
        return self.events[0][0] if self.events else -1

    @property
    def dose_mm(self) -> float:
        return self.total_mm


def _build_candidate_irrigation(
    parcel: ParcelConfig,
    candidate: IrrigationCandidate,
    today: date,
) -> IrrigationManagement:
    """Build AquaCrop IrrigationManagement from parcel log + candidate events."""
    base = parcel_to_irrigation(parcel)
    max_dose = parcel.irrigation.get("max_dose_mm", 40.0)

    if candidate.is_rainfed:
        return base

    extra_rows = []
    for day_offset, dose_mm in candidate.events:
        if dose_mm > 0 and day_offset >= 0:
            event_date = today + pd.Timedelta(days=day_offset)
            extra_rows.append({"Date": pd.Timestamp(event_date), "Depth": dose_mm})

    if not extra_rows:
        return base

    extra = pd.DataFrame(extra_rows)

    if base.irrigation_method == 3:
        combined = (
            pd.concat([base.Schedule, extra], ignore_index=True)
            .sort_values("Date")
            .reset_index(drop=True)
        )
        return IrrigationManagement(irrigation_method=3, Schedule=combined, MaxIrr=max_dose)
    else:
        return IrrigationManagement(irrigation_method=3, Schedule=extra, MaxIrr=max_dose)


# ---------------------------------------------------------------------------
# Two-level private-internals candidate evaluator
# ---------------------------------------------------------------------------


@dataclass
class _TwoLevelBranchState:
    """A pre-run AquaCrop model at a branching date.

    This is intentionally private: it stores an AquaCropModel that has already
    been advanced to a particular timestep.  The optimization path below uses
    AquaCrop private attributes to avoid rerunning the full season for every
    candidate/member pair.
    """

    model: AquaCropModel
    branch_start_idx: int
    branch_steps: int
    metric_start_idx: int
    metric_steps: int
    branch_date: date
    crop_finished: bool = False


@dataclass
class _MemberFutureWeather:
    """Prepared member-specific weather array aligned to an AquaCrop time_span."""

    weather_array: np.ndarray
    time_span: Any
    full_end: date


def _model_dates(model: AquaCropModel) -> list[date]:
    """Return model time_span as Python dates."""
    return [ts.date() for ts in pd.to_datetime(model._clock_struct.time_span)]


def _index_for_model_date(model: AquaCropModel, target: date) -> int:
    """Return the AquaCrop timestep index corresponding to target."""
    dates = _model_dates(model)
    target_date = pd.Timestamp(target).date()
    try:
        return dates.index(target_date)
    except ValueError as exc:
        first = dates[0] if dates else None
        last = dates[-1] if dates else None
        raise ValueError(
            f"Date {target_date} is outside AquaCrop time_span ({first} -> {last})."
        ) from exc


def _advance_model_to_date(
    model: AquaCropModel,
    *,
    sim_start: date,
    branch_date: date,
) -> int:
    """Initialize model and advance it to the beginning of branch_date.

    If branch_date == today, candidate day+0 irrigation is still available
    because today's timestep has not been executed yet.
    """
    sim_start_ts = pd.Timestamp(sim_start).normalize()
    branch_ts = pd.Timestamp(branch_date).normalize()
    prefix_steps = int((branch_ts - sim_start_ts).days)

    if prefix_steps < 0:
        raise ValueError(f"branch_date={branch_date} is before sim_start={sim_start}.")
    if prefix_steps == 0:
        # Private, but needed because run_model(num_steps=0) is not allowed.
        model._initialize()
    else:
        model.run_model(
            num_steps=prefix_steps,
            till_termination=False,
            initialize_model=True,
            process_outputs=False,
        )

    branch_idx = _index_for_model_date(model, branch_date)
    current_idx = int(model._clock_struct.time_step_counter)

    if current_idx != branch_idx:
        dates = _model_dates(model)
        current_date = dates[current_idx] if 0 <= current_idx < len(dates) else None
        raise RuntimeError(
            "AquaCrop branch alignment failed: "
            f"expected index {branch_idx} for {branch_date}, "
            f"but model is at index {current_idx} ({current_date})."
        )

    return branch_idx


def _add_candidate_to_initialized_schedule(
    model: AquaCropModel,
    candidate: IrrigationCandidate,
    *,
    today: date,
    max_dose: float,
    earliest_idx: int,
    latest_idx_exclusive: int | None = None,
) -> None:
    """Inject candidate events into AquaCrop's initialized daily schedule array.

    Parameters
    ----------
    earliest_idx
        First AquaCrop timestep index where candidate events may be added.
    latest_idx_exclusive
        Optional exclusive upper bound.  This is useful for two-level branching:
        add day+0/day+1 events while running the deterministic AROME window,
        then add day+2+ events only after switching to the member-specific
        weather branch.

    AquaCrop method 3 converts the user schedule DataFrame to a dense daily
    array during initialization.  This function mutates that array directly.
    """
    if candidate.is_rainfed:
        return

    schedule = np.asarray(model._param_struct.IrrMngt.Schedule, dtype=float).copy()
    date_to_idx = {d: i for i, d in enumerate(_model_dates(model))}

    for day_offset, dose_mm in candidate.events:
        if dose_mm <= 0 or day_offset < 0:
            continue

        event_date = (pd.Timestamp(today) + pd.Timedelta(days=int(day_offset))).date()
        idx = date_to_idx.get(event_date)
        if idx is None or idx < earliest_idx:
            continue
        if latest_idx_exclusive is not None and idx >= latest_idx_exclusive:
            continue

        # Preserve the physical MaxIrr daily cap when a farmer event and a
        # candidate event fall on the same date.
        schedule[idx] = min(schedule[idx] + float(dose_mm), float(max_dose))

    model._param_struct.IrrMngt.irrigation_method = 3
    model._param_struct.IrrMngt.Schedule = schedule
    model._param_struct.IrrMngt.MaxIrr = float(max_dose)


def _prepare_member_future_weather(
    *,
    member_weathers: dict[int, pd.DataFrame],
    member_end_dates: dict[int, date],
    parcel: ParcelConfig,
    sim_start: date,
    soil: Soil,
    crop: Crop,
    iwc: InitialWaterContent,
) -> dict[int, _MemberFutureWeather]:
    """Initialize each member once to obtain AquaCrop's internal weather array.

    The two-level evaluator uses one reference member to produce the shared
    historical and candidate-specific AROME states.  From ensemble_start onward,
    each job swaps in the relevant member's private ``_weather`` array.
    """
    prepared: dict[int, _MemberFutureWeather] = {}

    for member_id, weather in member_weathers.items():
        full_end = member_end_dates[member_id]
        weather_next, init_end_str, actual_sim_days = _prepare_aquacrop_init(
            weather=weather, crop=crop, sim_start=sim_start, sim_end=full_end
        )
        model = AquaCropModel(
            sim_start_time=sim_start.strftime("%Y/%m/%d"),
            sim_end_time=init_end_str,
            weather_df=weather_next,
            soil=soil,
            crop=crop,
            initial_water_content=iwc,
            irrigation_management=parcel_to_irrigation(parcel),
        )
        model._initialize()
        prepared[member_id] = _MemberFutureWeather(
            weather_array=np.asarray(model._weather).copy(),
            time_span=copy.deepcopy(model._clock_struct.time_span),
            full_end=full_end,
        )

    return prepared


def _assert_compatible_member_time_spans(
    reference_model: AquaCropModel,
    member_future_weather: dict[int, _MemberFutureWeather],
) -> None:
    """Ensure all member weather arrays share the reference model time axis."""
    ref_dates = [ts.date() for ts in pd.to_datetime(reference_model._clock_struct.time_span)]

    for member_id, prepared in member_future_weather.items():
        member_dates = [ts.date() for ts in pd.to_datetime(prepared.time_span)]
        if member_dates != ref_dates:
            raise ValueError(
                "Two-level branching currently requires all members to share "
                "the same AquaCrop time_span. "
                f"Member {member_id} differs from the reference member."
            )



_TWO_LEVEL_WORKER_CTX: dict[str, Any] = {}


@dataclass(frozen=True)
class YieldBranchMemberResult:
    """Final dry yield and first water-stress date for one forecast member."""

    member_id: int
    dry_yield_t_ha: float
    first_stress_date: date | None


def _extract_yield_branch_metrics(
    model: AquaCropModel,
    *,
    stress_threshold_ks: float,
    stress_not_before: date | None,
) -> tuple[float, date | None]:
    """Read yield/stress directly from AquaCrop private output arrays.

    This deliberately avoids ``get_crop_growth``/``get_water_flux`` because the
    event-driven optimizer calls this path many times.  The column positions are
    the same ones already used by ``_extract_optimizer_metrics_from_private_outputs``.
    In AquaCrop 3.x ``crop_growth[:, 12]`` is ``DryYield`` and water-flux columns
    15/16 are actual/potential transpiration.
    """
    cg = model._outputs.crop_growth
    wf = model._outputs.water_flux

    if isinstance(cg, pd.DataFrame):
        dry = cg["DryYield"].to_numpy(dtype=float)
        counters = cg["time_step_counter"].to_numpy(dtype=float)
    else:
        cg_arr = np.asarray(cg)
        if cg_arr.size == 0:
            dry = np.asarray([], dtype=float)
            counters = np.asarray([], dtype=float)
        else:
            counters = cg_arr[:, 0].astype(float)
            dry = cg_arr[:, 12].astype(float)

    # AquaCrop preallocates trailing zero rows. Restrict to timesteps that have
    # actually been completed, then use the largest finite DryYield.  DryYield
    # is cumulative within the crop season, so this is robust to terminal rows.
    completed_stop = int(model._clock_struct.time_step_counter)
    valid_yield = (
        np.isfinite(counters) & np.isfinite(dry) & (counters >= 0) & (counters < completed_stop)
    )
    yield_t_ha = float(np.nanmax(dry[valid_yield])) if np.any(valid_yield) else 0.0

    if isinstance(wf, pd.DataFrame):
        wf_counter = wf["time_step_counter"].to_numpy(dtype=float)
        tr = wf["Tr"].to_numpy(dtype=float)
        tr_pot = wf["TrPot"].to_numpy(dtype=float)
    else:
        wf_arr = np.asarray(wf)
        if wf_arr.size == 0:
            return yield_t_ha, None
        wf_counter = wf_arr[:, 0].astype(float)
        tr = wf_arr[:, 15].astype(float)
        tr_pot = wf_arr[:, 16].astype(float)

    model_dates = _model_dates(model)
    valid = (
        np.isfinite(wf_counter)
        & np.isfinite(tr)
        & np.isfinite(tr_pot)
        & (wf_counter >= 0)
        & (wf_counter < completed_stop)
        & (wf_counter < len(model_dates))
    )
    if not np.any(valid):
        return yield_t_ha, None

    counters_i = wf_counter[valid].astype(int)
    tr_v = tr[valid]
    tr_pot_v = tr_pot[valid]
    ks = np.where(tr_pot_v > 0.01, tr_v / tr_pot_v, 1.0)

    for counter, value in zip(counters_i, ks):
        d = model_dates[int(counter)]
        if stress_not_before is not None and d < stress_not_before:
            continue
        if np.isfinite(value) and value < float(stress_threshold_ks):
            return yield_t_ha, d

    return yield_t_ha, None


_YIELD_BRANCH_WORKER_CTX: dict[str, Any] = {}


def _init_yield_branch_worker(
    candidate_state: _TwoLevelBranchState,
    member_future_weather: dict[int, _MemberFutureWeather],
    candidate: IrrigationCandidate,
    today: date,
    max_dose: float,
    stress_threshold_ks: float,
    stress_not_before: date | None,
) -> None:
    global _YIELD_BRANCH_WORKER_CTX
    _YIELD_BRANCH_WORKER_CTX = {
        "candidate_state": candidate_state,
        "member_future_weather": member_future_weather,
        "candidate": candidate,
        "today": today,
        "max_dose": float(max_dose),
        "stress_threshold_ks": float(stress_threshold_ks),
        "stress_not_before": stress_not_before,
    }


def _run_yield_branch_member_job(member_id: int) -> YieldBranchMemberResult:
    """Run one forecast member from an already-computed deterministic state."""
    ctx = _YIELD_BRANCH_WORKER_CTX
    base_state: _TwoLevelBranchState = ctx["candidate_state"]
    member_weather: _MemberFutureWeather = ctx["member_future_weather"][member_id]

    model = copy.deepcopy(base_state.model)
    model._weather = member_weather.weather_array

    _add_candidate_to_initialized_schedule(
        model,
        ctx["candidate"],
        today=ctx["today"],
        max_dose=ctx["max_dose"],
        earliest_idx=base_state.branch_start_idx,
    )

    if base_state.branch_steps > 0:
        model.run_model(
            num_steps=base_state.branch_steps,
            till_termination=False,
            initialize_model=False,
            process_outputs=False,
        )

    dry_yield, first_stress = _extract_yield_branch_metrics(
        model,
        stress_threshold_ks=ctx["stress_threshold_ks"],
        stress_not_before=ctx["stress_not_before"],
    )
    return YieldBranchMemberResult(
        member_id=int(member_id),
        dry_yield_t_ha=float(dry_yield),
        first_stress_date=first_stress,
    )


class AquaCropYieldBranchingEvaluator:
    """Reusable two-level AquaCrop evaluator for event-driven yield optimization.

    The expensive season history is advanced exactly once when this object is
    constructed.  Each schedule evaluation then performs only:

    1. one deterministic AROME branch for the candidate; and
    2. one continuation per ensemble member from the ensemble boundary.

    This is the same private-internals strategy as
    ``evaluate_candidates_ensemble_branching_2level`` but exposed as a reusable
    evaluator because the yield optimizer chooses its next schedule adaptively.
    """

    def __init__(
        self,
        *,
        historical_forcing: DailyForcing,
        arome_forcing: DailyForcing,
        member_forcings: dict[int, DailyForcing],
        parcel: ParcelConfig,
        terrain: TerrainParams,
        soil_profile: SoilProfile,
        sim_start: date,
        today: date,
        initial_soil_water_profile: xr.Dataset | None = None,
        workers: int = 1,
    ) -> None:
        if not member_forcings:
            raise ValueError("member_forcings is empty.")

        self.today = pd.Timestamp(today).date()
        self.parcel = parcel
        self.max_dose = float(parcel.irrigation.get("max_dose_mm", 40.0))
        self.member_ids = sorted(int(m) for m in member_forcings)
        self.workers = max(1, int(workers))

        crop = parcel_to_crop(parcel)
        soil = _soil_for_crop(soil_profile, crop)
        iwc = InitialWaterContent(value=["FC"]) if initial_soil_water_profile is None else soil_water_profile_to_aquacrop_iwc(
            initial_soil_water_profile,
            soil,
        )

        base_weather = forcing_to_weather(
            historical_forcing.concat(arome_forcing), terrain, parcel.lat
        )
        member_weathers: dict[int, pd.DataFrame] = {}
        member_end_dates: dict[int, date] = {}
        member_start_dates: dict[int, date] = {}

        for member_id in self.member_ids:
            member_forcing = member_forcings[member_id]
            member_start_dates[member_id] = pd.Timestamp(member_forcing.dates[0]).date()
            member_end_dates[member_id] = pd.Timestamp(member_forcing.dates[-1]).date()
            member_weather = forcing_to_weather(member_forcing, terrain, parcel.lat)
            member_weathers[member_id] = (
                pd.concat([base_weather, member_weather])
                .drop_duplicates("Date", keep="first")
                .sort_values("Date")
                .reset_index(drop=True)
            )

        unique_starts = set(member_start_dates.values())
        unique_ends = set(member_end_dates.values())
        if len(unique_starts) != 1 or len(unique_ends) != 1:
            raise ValueError(
                "Yield branching requires all forecast members to share the same "
                f"start/end dates. Starts={member_start_dates}, ends={member_end_dates}"
            )
        self.ensemble_start_date = next(iter(unique_starts))
        self.full_end = next(iter(unique_ends))

        self.member_future_weather = _prepare_member_future_weather(
            member_weathers=member_weathers,
            member_end_dates=member_end_dates,
            parcel=parcel,
            sim_start=sim_start,
            soil=soil,
            crop=crop,
            iwc=iwc,
        )

        reference_member = self.member_ids[0]
        weather_ext, init_end_str, _ = _prepare_aquacrop_init(
            weather=member_weathers[reference_member],
            crop=crop,
            sim_start=sim_start,
            sim_end=self.full_end,
        )
        model = AquaCropModel(
            sim_start_time=sim_start.strftime("%Y/%m/%d"),
            sim_end_time=init_end_str,
            weather_df=weather_ext,
            soil=soil,
            crop=crop,
            initial_water_content=iwc,
            irrigation_management=parcel_to_irrigation(parcel),
        )
        model._initialize()
        _assert_compatible_member_time_spans(model, self.member_future_weather)

        # Rebuild because the compatibility check initialized the model.  Then
        # advance the complete observed season exactly once and retain that state.
        model = AquaCropModel(
            sim_start_time=sim_start.strftime("%Y/%m/%d"),
            sim_end_time=init_end_str,
            weather_df=weather_ext,
            soil=soil,
            crop=crop,
            initial_water_content=iwc,
            irrigation_management=parcel_to_irrigation(parcel),
        )
        self.crop_finished_before_today = False
        self.finished_date: date | None = None
        try:
            self.today_idx = _advance_model_to_date(
                model, sim_start=sim_start, branch_date=self.today
            )
        except RuntimeError:
            # AquaCrop can terminate before the requested branch date when the
            # crop has already reached maturity. In that case there is no
            # future irrigation decision left to optimize: every forecast
            # member has the same already-realized final yield.
            if not model._clock_struct.model_is_finished:
                raise

            self.crop_finished_before_today = True
            self.today_idx = int(model._clock_struct.time_step_counter)
            dates = _model_dates(model)
            if 0 <= self.today_idx < len(dates):
                self.finished_date = dates[self.today_idx]
            elif dates:
                self.finished_date = dates[-1]

            self.ensemble_start_idx = self.today_idx
            self.deterministic_steps = 0
            self.branch_steps = 0
            self._today_model = model
            return

        self.ensemble_start_idx = _index_for_model_date(model, self.ensemble_start_date)
        self.deterministic_steps = self.ensemble_start_idx - self.today_idx
        if self.deterministic_steps < 0:
            raise ValueError(
                f"Forecast members start {self.ensemble_start_date}, before today={self.today}."
            )
        self.branch_steps = min(
            next(iter(member_forcings.values())).n_days,
            len(model._clock_struct.time_span) - self.ensemble_start_idx,
        )
        if self.branch_steps <= 0:
            raise ValueError("No forecast timesteps remain after the ensemble boundary.")
        self._today_model = model

    def _candidate_state(self, candidate: IrrigationCandidate) -> _TwoLevelBranchState:
        model = copy.deepcopy(self._today_model)
        _add_candidate_to_initialized_schedule(
            model,
            candidate,
            today=self.today,
            max_dose=self.max_dose,
            earliest_idx=self.today_idx,
            latest_idx_exclusive=self.ensemble_start_idx,
        )
        if self.deterministic_steps > 0:
            model.run_model(
                num_steps=self.deterministic_steps,
                till_termination=False,
                initialize_model=False,
                process_outputs=False,
            )

        if model._clock_struct.model_is_finished:
            return _TwoLevelBranchState(
                model=model,
                branch_start_idx=int(model._clock_struct.time_step_counter),
                branch_steps=0,
                metric_start_idx=self.today_idx,
                metric_steps=0,
                branch_date=self.today,
                crop_finished=True,
            )
        current_idx = int(model._clock_struct.time_step_counter)
        if current_idx != self.ensemble_start_idx:
            raise RuntimeError(
                "Yield branch alignment failed: expected ensemble index "
                f"{self.ensemble_start_idx}, got {current_idx}."
            )
        return _TwoLevelBranchState(
            model=model,
            branch_start_idx=self.ensemble_start_idx,
            branch_steps=self.branch_steps,
            metric_start_idx=max(0, self.ensemble_start_idx - 1),
            metric_steps=self.branch_steps,
            branch_date=self.ensemble_start_date,
            crop_finished=False
        )

    def evaluate(
        self,
        candidate: IrrigationCandidate,
        *,
        stress_threshold_ks: float = 0.98,
        stress_not_before: date | None = None,
    ) -> dict[int, YieldBranchMemberResult]:
        if self.crop_finished_before_today:
            dry_yield, _ = _extract_yield_branch_metrics(
                self._today_model,
                stress_threshold_ks=stress_threshold_ks,
                stress_not_before=stress_not_before,
            )
            return {
                member_id: YieldBranchMemberResult(
                    member_id=member_id,
                    dry_yield_t_ha=float(dry_yield),
                    first_stress_date=None,
                )
                for member_id in self.member_ids
            }

        candidate_state = self._candidate_state(candidate)
        if candidate_state.crop_finished:
            dry_yield, first_stress = _extract_yield_branch_metrics(
                candidate_state.model,
                stress_threshold_ks=stress_threshold_ks,
                stress_not_before=stress_not_before,
            )

            return {
                member_id: YieldBranchMemberResult(
                    member_id=member_id,
                    dry_yield_t_ha=float(dry_yield),
                    first_stress_date=first_stress,
                )
                for member_id in self.member_ids
            }


        n_workers = min(self.workers, mp.cpu_count(), len(self.member_ids))
        initargs = (
            candidate_state,
            self.member_future_weather,
            candidate,
            self.today,
            self.max_dose,
            float(stress_threshold_ks),
            stress_not_before,
        )

        if n_workers <= 1:
            _init_yield_branch_worker(*initargs)
            rows = [_run_yield_branch_member_job(m) for m in self.member_ids]
        else:
            # Fork is important here: candidate_state contains a live AquaCrop
            # model.  On Windows/spawn this still works by pickling, but WSL/Linux
            # avoids that extra cost.
            mp_ctx = (
                mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
            )
            with mp_ctx.Pool(
                processes=n_workers,
                initializer=_init_yield_branch_worker,
                initargs=initargs,
            ) as pool:
                rows = pool.map(_run_yield_branch_member_job, self.member_ids)

        return {row.member_id: row for row in rows}

