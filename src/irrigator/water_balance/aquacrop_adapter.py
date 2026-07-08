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

logger = logging.getLogger(__name__)

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

    # Derived daily stress metric (Tr/TrPot, equivalent to Ks)
    @property
    def daily_stress(self) -> pd.DataFrame:
        """Daily stress metrics derived from AquaCrop outputs.

        Returns DataFrame with columns: date, dap, ks, canopy_cover,
        canopy_cover_ns, biomass, z_root, precip, irrigation, Wr.
        """
        cg = self.crop_growth
        wf = self.water_flux

        # Compute Ks equivalent: actual transpiration / potential transpiration
        ks = np.where(
            wf["TrPot"] > 0.01,
            wf["Tr"] / wf["TrPot"],
            1.0,  # no demand → no stress
        )

        n = min(len(cg), len(wf))

        return pd.DataFrame(
            {
                "dap": cg["dap"].values[:n],
                "gdd_cum": cg["gdd_cum"].values[:n],
                "ks": ks[:n],
                "canopy_cover": cg["canopy_cover"].values[:n],
                "canopy_cover_ns": cg["canopy_cover_ns"].values[:n],
                "biomass_kg_ha": cg["biomass"].values[:n],
                "z_root_m": cg["z_root"].values[:n],
                "harvest_index": cg["harvest_index"].values[:n],
                "precip_mm": wf["Infl"].values[:n],  # infiltration ≈ effective precip
                "irrigation_mm": wf["IrrDay"].values[:n],
                "tr_mm": wf["Tr"].values[:n],
                "tr_pot_mm": wf["TrPot"].values[:n],
                "es_mm": wf["Es"].values[:n],
                "deep_perc_mm": wf["DeepPerc"].values[:n],
                "wr_mm": wf["Wr"].values[:n],
            }
        )


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
    """
    weather = forcing_to_weather(forcing, terrain, parcel.lat)

    crop = parcel_to_crop(parcel)
    soil = soil_to_aquacrop(soil_profile, min_depth_m=crop.Zmax, extrapolate_below_profile=True)

    weather, init_end_str, actual_sim_days = _prepare_aquacrop_init(
        weather=weather,
        crop=crop,
        sim_start=sim_start,
        sim_end=sim_end,
    )

    iwc = initial_water_content or InitialWaterContent(value=["FC"])
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
    #model.run_model(till_termination=True)
    model.run_model(till_termination=True)

    result = _trim_aquacrop_outputs(model, actual_sim_days=actual_sim_days)

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
    """
    daily_stats: list[AquaCropEnsembleStats] = []

    # Pre-compute shared objects

    crop = parcel_to_crop(parcel)
    soil = soil_to_aquacrop(soil_profile, min_depth_m=crop.Zmax, extrapolate_below_profile=True)
    iwc = InitialWaterContent(value=["FC"])

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
# Irrigation optimization
# ---------------------------------------------------------------------------


def evaluate_smt_strategy(
    smt: list[float],
    forcing: DailyForcing,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    sim_end: date,
    *,
    max_irr_season_mm: float = 300.0,
    max_daily_mm: float = 40.0,
    crop_price_eur_t: float = 180.0,
    water_cost_eur_mm_ha: float = 2.5,
) -> dict[str, float]:
    """Evaluate a soil-moisture-target irrigation strategy.

    AquaCrop's SMT method (method=1) irrigates when soil moisture
    in the root zone drops below specified thresholds (as % of TAW)
    for each growth stage.

    Parameters
    ----------
    smt : soil moisture targets [initial, development, mid, late] as % TAW
    forcing, parcel, terrain, soil_profile : the usual
    sim_start, sim_end : simulation period
    max_irr_season_mm : maximum seasonal irrigation
    max_daily_mm : maximum per-application dose
    crop_price_eur_t : maize price for profit calculation
    water_cost_eur_mm_ha : water cost per mm per hectare

    Returns
    -------
    Dict with yield_t_ha, seasonal_irrigation_mm, profit_eur_ha,
    stress_days, max_stress.
    """
    irr = IrrigationManagement(
        irrigation_method=1,
        SMT=smt,
        MaxIrr=max_daily_mm,
        MaxIrrSeason=max_irr_season_mm,
    )

    result = run_aquacrop(
        forcing=forcing,
        parcel=parcel,
        terrain=terrain,
        soil_profile=soil_profile,
        sim_start=sim_start,
        sim_end=sim_end,
        irrigation_management=irr,
    )

    # Extract yield
    final = result.final_results
    if len(final) > 0:
        yield_t_ha = float(final.iloc[-1]["Dry yield (tonne/ha)"])
        seasonal_irr = float(final.iloc[-1]["Seasonal irrigation (mm)"])
    else:
        # Crop didn't reach harvest — estimate from biomass
        cg = result.crop_growth
        biomass = cg["biomass"].iloc[-2] if len(cg) > 1 else 0
        hi = cg["harvest_index"].iloc[-2] if len(cg) > 1 else 0
        yield_t_ha = biomass * hi / 1000.0  # kg/ha → t/ha
        wf = result.water_flux
        seasonal_irr = float(wf["IrrDay"].sum())

    # Stress metrics
    stress = result.daily_stress
    ks = stress["ks"].values
    stress_days = int((ks < 0.9).sum())
    max_stress = float(1.0 - np.nanmin(ks)) if len(ks) > 0 else 0.0

    # Profit
    area_ha = parcel.area_ha if hasattr(parcel, "area_ha") else 1.0
    revenue = yield_t_ha * crop_price_eur_t
    cost = seasonal_irr * water_cost_eur_mm_ha
    profit = revenue - cost

    return {
        "yield_t_ha": yield_t_ha,
        "seasonal_irrigation_mm": seasonal_irr,
        "profit_eur_ha": profit,
        "stress_days": stress_days,
        "max_stress": max_stress,
        "smt": smt,
    }


def optimize_irrigation(
    forcing: DailyForcing,
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    sim_end: date,
    **kwargs,
) -> dict[str, Any]:
    """Find optimal soil-moisture-target irrigation strategy.

    Uses scipy differential_evolution to search the 4-dimensional
    SMT space (one threshold per growth stage: initial, development,
    mid-season, late-season).

    Returns the best strategy found and its performance metrics.
    """
    from scipy.optimize import differential_evolution

    def objective(x):
        smt = list(x)
        metrics = evaluate_smt_strategy(
            smt=smt,
            forcing=forcing,
            parcel=parcel,
            terrain=terrain,
            soil_profile=soil_profile,
            sim_start=sim_start,
            sim_end=sim_end,
            **kwargs,
        )
        return -metrics["profit_eur_ha"]  # minimize negative profit

    # Bounds: SMT thresholds as % TAW [0-100] per growth stage
    bounds = [(30, 100), (30, 100), (50, 100), (20, 80)]

    logger.info("Optimizing irrigation strategy (differential evolution)...")
    result = differential_evolution(
        objective,
        bounds,
        maxiter=20,
        seed=42,
        tol=0.01,
        polish=False,
    )

    optimal_smt = list(result.x)
    metrics = evaluate_smt_strategy(
        smt=optimal_smt,
        forcing=forcing,
        parcel=parcel,
        terrain=terrain,
        soil_profile=soil_profile,
        sim_start=sim_start,
        sim_end=sim_end,
        **kwargs,
    )

    logger.info(
        "Optimization complete: SMT=%s, yield=%.1f t/ha, irrigation=%d mm, profit=€%.0f/ha",
        [f"{s:.0f}" for s in optimal_smt],
        metrics["yield_t_ha"],
        metrics["seasonal_irrigation_mm"],
        metrics["profit_eur_ha"],
    )

    return metrics


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


@dataclass
class CandidateResult:
    """Ensemble evaluation of one irrigation candidate."""

    candidate: IrrigationCandidate
    # Per-member outcomes
    member_max_stress: list[float]  # max(1-Ks) per member over forecast
    member_stress_days: list[int]  # days with Ks < 0.9 per member
    member_yield_impact: list[float]  # CC reduction vs no-stress (proxy)
    # Ensemble summary
    mean_max_stress: float
    median_max_stress: float
    pct_members_avoid_stress: float  # fraction with max Ks > 0.9
    mean_stress_days: float


def build_candidates(
    parcel_config: ParcelConfig,
    max_days_ahead: int = 7,
    max_events: int = 3,
) -> list[IrrigationCandidate]:
    """Build candidate irrigation schedules to evaluate.

    Generates:
    - 0 events: rainfed baseline
    - 1 event:  every (day, dose) combination
    - 2 events: pairs separated by ≥ min_interval, using min/max doses
    - 3 events: triples separated by ≥ min_interval, max dose only

    Parameters
    ----------
    parcel_config : parcel with irrigation constraints
    max_days_ahead : furthest day to consider for irrigation
    max_events : maximum number of irrigations per candidate (1–3)
    """
    irr_cfg = parcel_config.irrigation
    min_dose = irr_cfg.get("min_dose_mm", 15)
    max_dose = irr_cfg.get("max_dose_mm", 35)
    min_interval = irr_cfg.get("min_interval_days", 3)

    # Dose levels for single events: min, mid, max
    doses_single = sorted(set([min_dose, (min_dose + max_dose) / 2, max_dose]))
    # Dose levels for multi-event: just min and max to limit combinatorics
    doses_multi = sorted(set([min_dose, max_dose]))

    days = list(range(0, max_days_ahead + 1))

    candidates: list[IrrigationCandidate] = [
        IrrigationCandidate(events=[], label="No irrigation"),
    ]

    # --- Single events ---
    for d in days:
        for dose in doses_single:
            candidates.append(
                IrrigationCandidate(
                    events=[(d, dose)],
                    label=f"{dose:.0f}mm on day+{d}",
                )
            )

    # --- Double events ---
    if max_events >= 2:
        for d1 in days:
            for d2 in days:
                if d2 < d1 + min_interval:
                    continue
                for dose in doses_multi:
                    candidates.append(
                        IrrigationCandidate(
                            events=[(d1, dose), (d2, dose)],
                            label=f"2×{dose:.0f}mm on day+{d1} & +{d2}",
                        )
                    )

    # --- Triple events (max dose only to keep count manageable) ---
    if max_events >= 3:
        for d1 in range(0, min(4, max_days_ahead + 1)):
            for d2 in range(d1 + min_interval, max_days_ahead + 1):
                for d3 in range(d2 + min_interval, max_days_ahead + 1):
                    candidates.append(
                        IrrigationCandidate(
                            events=[(d1, max_dose), (d2, max_dose), (d3, max_dose)],
                            label=f"3×{max_dose:.0f}mm on day+{d1},{d2},{d3}",
                        )
                    )

    logger.info(
        "Built %d candidates (1-event: %d, 2-event: %d, 3-event: %d)",
        len(candidates),
        sum(1 for c in candidates if c.n_events == 1),
        sum(1 for c in candidates if c.n_events == 2),
        sum(1 for c in candidates if c.n_events == 3),
    )
    return candidates


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


_CANDIDATE_WORKER_CTX: dict[str, Any] = {}


def _init_candidate_worker(
    candidates: list[IrrigationCandidate],
    member_weathers: dict[int, pd.DataFrame],
    member_end_dates: dict[int, date],
    parcel: ParcelConfig,
    today: date,
    sim_start: date,
    soil: Soil,
    crop: Crop,
    iwc: InitialWaterContent,
    stress_threshold: float,
    n_forecast: int,
) -> None:
    """Initialize process-local state for candidate evaluation workers.

    The pool calls this once per worker process.  Jobs can then stay tiny:
    only (candidate_index, member_id) needs to be sent for each AquaCrop run.
    This avoids repeatedly pickling the same weather DataFrames, soil, crop,
    parcel, and candidate list for every job.
    """
    global _CANDIDATE_WORKER_CTX
    _CANDIDATE_WORKER_CTX = {
        "candidates": candidates,
        "member_weathers": member_weathers,
        "member_end_dates": member_end_dates,
        "parcel": parcel,
        "today": today,
        "sim_start": sim_start,
        "soil": soil,
        "crop": crop,
        "iwc": iwc,
        "stress_threshold": stress_threshold,
        "n_forecast": n_forecast,
    }


def _run_single_candidate_job(job: tuple[int, int]) -> tuple[int, int, float, int, float]:
    """Run one candidate × ensemble-member AquaCrop simulation.

    Returns
    -------
    tuple
        (candidate_index, member_id, max_stress, n_stress_days, cc_loss)
    """
    cand_idx, m_id = job
    ctx = _CANDIDATE_WORKER_CTX

    cand = ctx["candidates"][cand_idx]
    weather = ctx["member_weathers"][m_id]
    full_end = ctx["member_end_dates"][m_id]

    irr_mgmt = _build_candidate_irrigation(ctx["parcel"], cand, ctx["today"])

    model = AquaCropModel(
        sim_start_time=ctx["sim_start"].strftime("%Y/%m/%d"),
        sim_end_time=full_end.strftime("%Y/%m/%d"),
        weather_df=weather,
        soil=ctx["soil"],
        crop=ctx["crop"],
        initial_water_content=ctx["iwc"],
        irrigation_management=irr_mgmt,
    )
    model.run_model(till_termination=True)

    # Avoid constructing AquaCropResult and avoid calling get_water_storage().
    # The optimizer only needs stress and canopy-loss metrics.
    cg = model.get_crop_growth()
    wf = model.get_water_flux()

    n_total = min(len(cg), len(wf))
    if n_total > 1 and cg["dap"].iloc[n_total - 1] == 0:
        n_total -= 1

    n_forecast = ctx["n_forecast"]
    fc_start = max(0, n_total - n_forecast)

    tr = wf["Tr"].values[fc_start:n_total]
    tr_pot = wf["TrPot"].values[fc_start:n_total]
    ks = np.where(tr_pot > 0.01, tr / tr_pot, 1.0)

    max_stress = float(1.0 - np.nanmin(ks)) if len(ks) > 0 else 0.0
    n_stress_days = int((ks < ctx["stress_threshold"]).sum())

    if "canopy_cover_ns" in cg and "canopy_cover" in cg:
        cc_ns = cg["canopy_cover_ns"].values[fc_start:n_total]
        cc = cg["canopy_cover"].values[fc_start:n_total]
        cc_loss = float(np.nanmean(cc_ns - cc))
    else:
        cc_loss = 0.0

    return cand_idx, m_id, max_stress, n_stress_days, cc_loss


def evaluate_candidates_ensemble(
    historical_forcing: DailyForcing,
    arome_forcing: DailyForcing,
    member_forcings: dict[int, DailyForcing],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    today: date,
    candidates: list[IrrigationCandidate] | None = None,
    workers: int | None = 3,
    stress_threshold: float = 0.9,
) -> list[CandidateResult]:
    """Evaluate irrigation candidates across the IFS ENS ensemble.

    Optimized: weather DataFrames (including ET₀) are pre-computed once
    per member.  The historical + AROME portion (Jan 1 → today+2) is
    shared by all candidates — only the irrigation schedule varies.

    Immediate runtime fix:
    - build all candidate × member jobs once
    - create only one process pool
    - send only small job tuples to workers
    - avoid full AquaCropResult/water_storage extraction inside optimizer jobs
    """
    if candidates is None:
        candidates = build_candidates(parcel)

    if not candidates:
        return []
    if not member_forcings:
        return []

    # ------------------------------------------------------------------
    # Pre-compute shared objects (computed once, reused for all runs)
    # ------------------------------------------------------------------
    soil = soil_to_aquacrop(soil_profile)
    crop = parcel_to_crop(parcel)
    iwc = InitialWaterContent(value=["FC"])

    # Base weather: historical + AROME (includes ET₀ computation)
    base_forcing = historical_forcing.concat(arome_forcing)
    base_weather = forcing_to_weather(base_forcing, terrain, parcel.lat)

    # Per-member weather: base + IFS ENS forecast.
    # ET₀ is computed once per member here, not once per candidate.
    member_weathers: dict[int, pd.DataFrame] = {}
    member_end_dates: dict[int, date] = {}

    for m_id, m_forcing in member_forcings.items():
        m_weather = forcing_to_weather(m_forcing, terrain, parcel.lat)

        # Concat with base taking precedence on overlapping dates
        full = (
            pd.concat([base_weather, m_weather])
            .drop_duplicates("Date", keep="first")
            .sort_values("Date")
            .reset_index(drop=True)
        )
        member_weathers[m_id] = full
        member_end_dates[m_id] = pd.Timestamp(m_forcing.dates[-1]).date()

    n_forecast = next(iter(member_forcings.values())).n_days
    member_ids = list(member_weathers.keys())

    jobs: list[tuple[int, int]] = [
        (cand_idx, m_id) for cand_idx in range(len(candidates)) for m_id in member_ids
    ]

    n_jobs = len(jobs)
    if workers is None or workers <= 0:
        n_workers = min(mp.cpu_count(), n_jobs)
    else:
        n_workers = min(workers, mp.cpu_count(), n_jobs)

    logger.info(
        "Evaluating %d candidates × %d members = %d AquaCrop runs "
        "(%d worker%s, weather pre-computed, %d historical + %d forecast days)",
        len(candidates),
        len(member_forcings),
        n_jobs,
        n_workers,
        "s" if n_workers != 1 else "",
        base_forcing.n_days,
        n_forecast,
    )

    # candidate_member_results[candidate_index][member_id] =
    #     (max_stress, n_stress_days, cc_loss)
    candidate_member_results: list[dict[int, tuple[float, int, float]]] = [{} for _ in candidates]

    initargs = (
        candidates,
        member_weathers,
        member_end_dates,
        parcel,
        today,
        sim_start,
        soil,
        crop,
        iwc,
        stress_threshold,
        n_forecast,
    )

    if n_workers == 1:
        _init_candidate_worker(*initargs)
        iterator = map(_run_single_candidate_job, jobs)
        for cand_idx, m_id, max_stress, n_stress_days, cc_loss in iterator:
            candidate_member_results[cand_idx][m_id] = (
                max_stress,
                n_stress_days,
                cc_loss,
            )
    else:
        # AquaCrop jobs are relatively heavy.  A small chunksize keeps load
        # balancing good while reducing IPC overhead versus chunksize=1.
        chunksize = max(1, n_jobs // (n_workers * 8))
        with mp.Pool(
            processes=n_workers,
            initializer=_init_candidate_worker,
            initargs=initargs,
        ) as pool:
            for cand_idx, m_id, max_stress, n_stress_days, cc_loss in pool.imap_unordered(
                _run_single_candidate_job,
                jobs,
                chunksize=chunksize,
            ):
                candidate_member_results[cand_idx][m_id] = (
                    max_stress,
                    n_stress_days,
                    cc_loss,
                )

    # ------------------------------------------------------------------
    # Aggregate per candidate
    # ------------------------------------------------------------------
    results: list[CandidateResult] = []

    for ci, cand in enumerate(candidates):
        member_rows = [candidate_member_results[ci][m_id] for m_id in member_ids]

        member_max_stress = [row[0] for row in member_rows]
        member_stress_days = [row[1] for row in member_rows]
        member_yield_impact = [row[2] for row in member_rows]

        arr_stress = np.asarray(member_max_stress, dtype=float)
        arr_days = np.asarray(member_stress_days, dtype=float)

        result = CandidateResult(
            candidate=cand,
            member_max_stress=member_max_stress,
            member_stress_days=member_stress_days,
            member_yield_impact=member_yield_impact,
            mean_max_stress=float(np.nanmean(arr_stress)),
            median_max_stress=float(np.nanmedian(arr_stress)),
            pct_members_avoid_stress=float(np.nanmean(arr_stress < (1.0 - stress_threshold))),
            mean_stress_days=float(np.nanmean(arr_days)),
        )
        results.append(result)

        logger.info(
            "Candidate %d/%d '%s': mean_stress=%.3f, avoid_pct=%.0f%%, mean_stress_days=%.1f",
            ci + 1,
            len(candidates),
            cand.label,
            result.mean_max_stress,
            result.pct_members_avoid_stress * 100,
            result.mean_stress_days,
        )

    # Sort by best outcome: highest pct_members_avoid_stress, then lowest mean_stress
    results.sort(key=lambda r: (-r.pct_members_avoid_stress, r.mean_max_stress))
    return results


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
        model = AquaCropModel(
            sim_start_time=sim_start.strftime("%Y/%m/%d"),
            sim_end_time=full_end.strftime("%Y/%m/%d"),
            weather_df=weather,
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


def _build_candidate_arome_states(
    *,
    reference_model: AquaCropModel,
    candidates: list[IrrigationCandidate],
    today: date,
    sim_start: date,
    ensemble_start_date: date,
    metric_days: int,
    max_dose: float,
) -> dict[int, _TwoLevelBranchState]:
    """Run the common history once, then candidate-specific deterministic days once.

    Output state is positioned at ``ensemble_start_date``, which should be the
    first date in the ensemble-member forcing.  Do not infer this solely from
    ``today + arome_days`` because duplicate/overlap handling can otherwise
    create a one-day scoring mismatch at the boundary.
    """

    today_idx = _advance_model_to_date(
        reference_model,
        sim_start=sim_start,
        branch_date=today,
    )

    ensemble_start_idx = _index_for_model_date(reference_model, ensemble_start_date)
    deterministic_steps = ensemble_start_idx - today_idx
    if deterministic_steps < 0:
        raise ValueError(f"ensemble_start_date={ensemble_start_date} is before today={today}.")
    n_time = len(reference_model._clock_struct.time_span)
    metric_steps = min(metric_days, n_time - ensemble_start_idx)
    if metric_steps <= 0:
        raise RuntimeError(
            f"No metric timesteps available from ensemble_start_date={ensemble_start_date}."
        )
    # AquaCrop private output arrays label completed timesteps with the
    # pre-increment time_step_counter.  Therefore, when the model is positioned
    # at ensemble_start_idx, the output row for the first ensemble timestep is
    # stored with counter ensemble_start_idx - 1.  The full-rerun reference uses
    # public DataFrame positions, so the private-output scorer must shift the
    # metric window back by one internal counter to include the first ensemble
    # day instead of starting at day+3.
    metric_output_start_idx = max(0, ensemble_start_idx - 1)
    candidate_states: dict[int, _TwoLevelBranchState] = {}

    for cand_idx, candidate in enumerate(candidates):
        model = copy.deepcopy(reference_model)

        # Only inject events that belong to the deterministic shared window.
        # Events on the boundary date itself (e.g. day+2 when ensemble starts at
        # day+2) are injected later inside the member-specific branch, before
        # that timestep is processed.

        _add_candidate_to_initialized_schedule(
            model,
            candidate,
            today=today,
            max_dose=max_dose,
            earliest_idx=today_idx,
            latest_idx_exclusive=ensemble_start_idx,
        )

        if deterministic_steps > 0:
            model.run_model(
                num_steps=deterministic_steps,
                till_termination=False,
                initialize_model=False,
                process_outputs=False,
            )

        current_idx = int(model._clock_struct.time_step_counter)
        if current_idx != ensemble_start_idx:
            dates = _model_dates(model)
            current_date = dates[current_idx] if 0 <= current_idx < len(dates) else None
            raise RuntimeError(
                "Candidate deterministic branch alignment failed: "
                f"expected index {ensemble_start_idx} for {ensemble_start_date}, "
                f"but model is at index {current_idx} ({current_date})."
            )

        candidate_states[cand_idx] = _TwoLevelBranchState(
            model=model,
            branch_start_idx=ensemble_start_idx,
            branch_steps=metric_steps,
            metric_start_idx=metric_output_start_idx,
            metric_steps=metric_steps,
            branch_date=ensemble_start_date,
        )

    return candidate_states


def _extract_optimizer_metrics_from_private_outputs(
    model: AquaCropModel,
    state: _TwoLevelBranchState,
    *,
    stress_threshold: float,
) -> tuple[float, int, float]:
    """Extract optimizer metrics directly from AquaCrop private outputs."""
    start = state.metric_start_idx
    stop = start + state.metric_steps

    wf = model._outputs.water_flux
    cg = model._outputs.crop_growth

    if isinstance(wf, pd.DataFrame):
        wf_slice = wf[(wf["time_step_counter"] >= start) & (wf["time_step_counter"] < stop)]
        cg_slice = cg[(cg["time_step_counter"] >= start) & (cg["time_step_counter"] < stop)]

        tr = wf_slice["Tr"].to_numpy(dtype=float)
        tr_pot = wf_slice["TrPot"].to_numpy(dtype=float)
        cc_loss_arr = cg_slice["canopy_cover_ns"].to_numpy(dtype=float) - cg_slice[
            "canopy_cover"
        ].to_numpy(dtype=float)
    else:
        wf_arr = np.asarray(wf)
        cg_arr = np.asarray(cg)

        if wf_arr.size == 0 or cg_arr.size == 0:
            return 1.0, state.metric_steps, 1.0

        # Use AquaCrop's time_step_counter column instead of raw row positions.
        # This avoids off-by-one errors at branch boundaries and avoids including
        # any terminal/zero rows that AquaCrop may append after the last timestep.
        wf_mask = (wf_arr[:, 0] >= start) & (wf_arr[:, 0] < stop)
        cg_mask = (cg_arr[:, 0] >= start) & (cg_arr[:, 0] < stop)
        wf_slice = wf_arr[wf_mask]
        cg_slice = cg_arr[cg_mask]

        if wf_slice.size == 0 or cg_slice.size == 0:
            return 1.0, state.metric_steps, 1.0

        # Raw AquaCrop output columns, from aquacrop.timestep:
        # water_flux: 15=Tr, 16=TrPot
        # crop_growth: 6=canopy_cover, 7=canopy_cover_ns
        tr = wf_slice[:, 15].astype(float)
        tr_pot = wf_slice[:, 16].astype(float)
        cc_loss_arr = cg_slice[:, 7].astype(float) - cg_slice[:, 6].astype(float)

    if len(tr) == 0:
        return 1.0, state.metric_steps, 1.0

    ks = np.where(tr_pot > 0.01, tr / tr_pot, 1.0)
    max_stress = float(1.0 - np.nanmin(ks))
    n_stress_days = int(np.sum(ks < stress_threshold))
    cc_loss = float(np.nanmean(cc_loss_arr)) if len(cc_loss_arr) else 0.0

    return max_stress, n_stress_days, cc_loss


_TWO_LEVEL_WORKER_CTX: dict[str, Any] = {}


def _init_two_level_worker(
    candidate_states: dict[int, _TwoLevelBranchState],
    member_future_weather: dict[int, _MemberFutureWeather],
    candidates: list[IrrigationCandidate],
    today: date,
    max_dose: float,
    stress_threshold: float,
) -> None:
    """Initialize process-local context for two-level branching workers."""
    global _TWO_LEVEL_WORKER_CTX
    _TWO_LEVEL_WORKER_CTX = {
        "candidate_states": candidate_states,
        "member_future_weather": member_future_weather,
        "candidates": candidates,
        "today": today,
        "max_dose": float(max_dose),
        "stress_threshold": float(stress_threshold),
    }


def _run_two_level_candidate_member_job(
    job: tuple[int, int],
) -> tuple[int, int, float, int, float]:
    """Run one candidate × member from the candidate-specific AROME state."""
    cand_idx, member_id = job
    ctx = _TWO_LEVEL_WORKER_CTX

    base_state = ctx["candidate_states"][cand_idx]
    member_weather = ctx["member_future_weather"][member_id]

    model = copy.deepcopy(base_state.model)
    # At this point the model is already at ensemble_start.  Replace only the
    # internal weather array so that today+arome_days onward follows the member.
    model._weather = member_weather.weather_array

    state = _TwoLevelBranchState(
        model=model,
        branch_start_idx=base_state.branch_start_idx,
        branch_steps=base_state.branch_steps,
        metric_start_idx=base_state.metric_start_idx,
        metric_steps=base_state.metric_steps,
        branch_date=base_state.branch_date,
    )

    # Inject events on/after the ensemble boundary only after the member weather
    # has been selected.  This fixes boundary candidates such as day+2 when the
    # ensemble also starts at day+2.
    _add_candidate_to_initialized_schedule(
        model,
        ctx["candidates"][cand_idx],
        today=ctx["today"],
        max_dose=ctx["max_dose"],
        earliest_idx=state.branch_start_idx,
    )

    if state.branch_steps > 0:
        model.run_model(
            num_steps=state.branch_steps,
            till_termination=False,
            initialize_model=False,
            process_outputs=False,
        )

    max_stress, n_stress_days, cc_loss = _extract_optimizer_metrics_from_private_outputs(
        model,
        state,
        stress_threshold=ctx["stress_threshold"],
    )

    return cand_idx, member_id, max_stress, n_stress_days, cc_loss


def evaluate_candidates_ensemble_branching_2level(
    historical_forcing: DailyForcing,
    arome_forcing: DailyForcing,
    member_forcings: dict[int, DailyForcing],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    today: date,
    candidates: list[IrrigationCandidate] | None = None,
    workers: int | None = 3,
    stress_threshold: float = 0.9,
    arome_days: int = 2,
) -> list[CandidateResult]:
    """Evaluate candidates using two-level AquaCrop branching.

    Pipeline:
    1. Build full weather per member: historical + deterministic AROME + member.
    2. Initialize one reference full-weather model and run sim_start -> today - 1.
    3. For each candidate, add its full future schedule and run the shared
       deterministic AROME window once: today -> today + arome_days - 1.
    4. For each candidate × member, copy that candidate state, swap in the
       member-specific internal weather array, run ensemble_start -> full_end,
       and score the same final IFS horizon as the original evaluator.

    This is a private-internals prototype.  Keep the original
    evaluate_candidates_ensemble() available for equivalence checks.
    """
    if candidates is None:
        candidates = build_candidates(parcel)

    if not candidates or not member_forcings:
        return []

    crop = parcel_to_crop(parcel)
    soil = soil_to_aquacrop(soil_profile, min_depth_m=crop.Zmax, extrapolate_below_profile=True)
    iwc = InitialWaterContent(value=["FC"])

    base_forcing = historical_forcing.concat(arome_forcing)
    base_weather = forcing_to_weather(base_forcing, terrain, parcel.lat)

    member_weathers: dict[int, pd.DataFrame] = {}
    member_end_dates: dict[int, date] = {}

    for member_id, member_forcing in member_forcings.items():
        member_weather = forcing_to_weather(member_forcing, terrain, parcel.lat)
        full_weather = (
            pd.concat([base_weather, member_weather])
            .drop_duplicates("Date", keep="first")
            .sort_values("Date")
            .reset_index(drop=True)
        )
        member_weathers[member_id] = full_weather
        member_end_dates[member_id] = pd.Timestamp(member_forcing.dates[-1]).date()

    metric_days = next(iter(member_forcings.values())).n_days
    member_start_dates = {
        member_id: pd.Timestamp(member_forcings[member_id].dates[0]).date()
        for member_id in member_weathers
    }
    unique_member_start_dates = set(member_start_dates.values())
    if len(unique_member_start_dates) != 1:
        raise ValueError(
            "Two-level branching requires all ensemble members to start on the same date. "
            f"Got: {member_start_dates}"
        )
    ensemble_start_date = next(iter(unique_member_start_dates))
    inferred_arome_days = (pd.Timestamp(ensemble_start_date) - pd.Timestamp(today).normalize()).days
    if inferred_arome_days != arome_days:
        logger.warning(
            "arome_days=%d but first ensemble member date implies %d deterministic days "
            "(%s -> %s). Using the member forcing date as source of truth.",
            arome_days,
            inferred_arome_days,
            today,
            ensemble_start_date,
        )

    max_dose = float(parcel.irrigation.get("max_dose_mm", 40.0))
    member_ids = list(member_weathers.keys())
    reference_member_id = member_ids[0]

    # Initialize members once to get AquaCrop-prepared weather arrays.
    member_future_weather = _prepare_member_future_weather(
        member_weathers=member_weathers,
        member_end_dates=member_end_dates,
        parcel=parcel,
        sim_start=sim_start,
        soil=soil,
        crop=crop,
        iwc=iwc,
    )

    reference_model = AquaCropModel(
        sim_start_time=sim_start.strftime("%Y/%m/%d"),
        sim_end_time=member_end_dates[reference_member_id].strftime("%Y/%m/%d"),
        weather_df=member_weathers[reference_member_id],
        soil=soil,
        crop=crop,
        initial_water_content=iwc,
        irrigation_management=parcel_to_irrigation(parcel),
    )

    # Initialize reference before time-span compatibility check.
    reference_model._initialize()
    _assert_compatible_member_time_spans(reference_model, member_future_weather)

    # Rebuild reference_model after the compatibility check because _initialize()
    # above was only needed to inspect the time axis; _build_candidate_arome_states
    # expects an unadvanced model and will initialize/advance it itself.
    reference_model = AquaCropModel(
        sim_start_time=sim_start.strftime("%Y/%m/%d"),
        sim_end_time=member_end_dates[reference_member_id].strftime("%Y/%m/%d"),
        weather_df=member_weathers[reference_member_id],
        soil=soil,
        crop=crop,
        initial_water_content=iwc,
        irrigation_management=parcel_to_irrigation(parcel),
    )

    candidate_states = _build_candidate_arome_states(
        reference_model=reference_model,
        candidates=candidates,
        today=today,
        sim_start=sim_start,
        ensemble_start_date=ensemble_start_date,
        metric_days=metric_days,
        max_dose=max_dose,
    )

    jobs = [
        (cand_idx, member_id) for cand_idx in range(len(candidates)) for member_id in member_ids
    ]
    n_jobs = len(jobs)

    if workers is None or workers <= 0:
        n_workers = min(mp.cpu_count(), n_jobs)
    else:
        n_workers = min(int(workers), mp.cpu_count(), n_jobs)

    logger.info(
        "Evaluating %d candidates × %d members = %d AquaCrop branches "
        "(two-level branching, arome_days=%d, workers=%d)",
        len(candidates),
        len(member_ids),
        n_jobs,
        arome_days,
        n_workers,
    )

    candidate_member_results: list[dict[int, tuple[float, int, float]]] = [{} for _ in candidates]

    initargs = (
        candidate_states,
        member_future_weather,
        candidates,
        today,
        max_dose,
        stress_threshold,
    )

    if n_workers == 1:
        _init_two_level_worker(*initargs)
        iterator = map(_run_two_level_candidate_member_job, jobs)
        for cand_idx, member_id, max_stress, n_days, cc_loss in iterator:
            candidate_member_results[cand_idx][member_id] = (max_stress, n_days, cc_loss)
    else:
        # Prefer fork on WSL/Linux to avoid pickling many already-branched models.
        mp_ctx = (
            mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        )
        chunksize = max(1, n_jobs // (n_workers * 8))

        with mp_ctx.Pool(
            processes=n_workers,
            initializer=_init_two_level_worker,
            initargs=initargs,
        ) as pool:
            for cand_idx, member_id, max_stress, n_days, cc_loss in pool.imap_unordered(
                _run_two_level_candidate_member_job,
                jobs,
                chunksize=chunksize,
            ):
                candidate_member_results[cand_idx][member_id] = (max_stress, n_days, cc_loss)

    results: list[CandidateResult] = []

    for ci, candidate in enumerate(candidates):
        rows = [candidate_member_results[ci][member_id] for member_id in member_ids]
        member_max_stress = [row[0] for row in rows]
        member_stress_days = [row[1] for row in rows]
        member_yield_impact = [row[2] for row in rows]

        arr_stress = np.asarray(member_max_stress, dtype=float)
        arr_days = np.asarray(member_stress_days, dtype=float)

        result = CandidateResult(
            candidate=candidate,
            member_max_stress=member_max_stress,
            member_stress_days=member_stress_days,
            member_yield_impact=member_yield_impact,
            mean_max_stress=float(np.nanmean(arr_stress)),
            median_max_stress=float(np.nanmedian(arr_stress)),
            pct_members_avoid_stress=float(np.nanmean(arr_stress < (1.0 - stress_threshold))),
            mean_stress_days=float(np.nanmean(arr_days)),
        )
        results.append(result)

        logger.info(
            "Candidate %d/%d '%s': mean_stress=%.3f, avoid_pct=%.0f%%, mean_stress_days=%.1f",
            ci + 1,
            len(candidates),
            candidate.label,
            result.mean_max_stress,
            result.pct_members_avoid_stress * 100,
            result.mean_stress_days,
        )

    results.sort(key=lambda r: (-r.pct_members_avoid_stress, r.mean_max_stress))
    return results


# Backward-compatible shorter alias for the private-internals prototype.
evaluate_candidates_ensemble_branching = evaluate_candidates_ensemble_branching_2level


def _evaluate_candidates_ensemble_full_reference_serial(
    *,
    historical_forcing: DailyForcing,
    arome_forcing: DailyForcing,
    member_forcings: dict[int, DailyForcing],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    today: date,
    candidates: list[IrrigationCandidate],
    stress_threshold: float,
) -> list[CandidateResult]:
    """Small, serial full-rerun reference implementation for validation only.

    This intentionally avoids the older multiprocessing evaluator so that the
    private-branching comparison is not affected by pool/stale-member bugs.
    """
    soil = soil_to_aquacrop(soil_profile)
    crop = parcel_to_crop(parcel)
    iwc = InitialWaterContent(value=["FC"])

    base_forcing = historical_forcing.concat(arome_forcing)
    base_weather = forcing_to_weather(base_forcing, terrain, parcel.lat)

    member_weathers: dict[int, pd.DataFrame] = {}
    member_end_dates: dict[int, date] = {}

    for member_id, member_forcing in member_forcings.items():
        member_weather = forcing_to_weather(member_forcing, terrain, parcel.lat)
        full_weather = (
            pd.concat([base_weather, member_weather])
            .drop_duplicates("Date", keep="first")
            .sort_values("Date")
            .reset_index(drop=True)
        )
        member_weathers[member_id] = full_weather
        member_end_dates[member_id] = pd.Timestamp(member_forcing.dates[-1]).date()

    metric_days = next(iter(member_forcings.values())).n_days
    member_ids = list(member_weathers)
    results: list[CandidateResult] = []

    for candidate in candidates:
        member_max_stress: list[float] = []
        member_stress_days: list[int] = []
        member_yield_impact: list[float] = []

        for member_id in member_ids:
            irr_mgmt = _build_candidate_irrigation(parcel, candidate, today)
            model = AquaCropModel(
                sim_start_time=sim_start.strftime("%Y/%m/%d"),
                sim_end_time=member_end_dates[member_id].strftime("%Y/%m/%d"),
                weather_df=member_weathers[member_id],
                soil=soil,
                crop=crop,
                initial_water_content=iwc,
                irrigation_management=irr_mgmt,
            )
            model.run_model(till_termination=True)

            cg = model.get_crop_growth()
            wf = model.get_water_flux()

            n_total = min(len(cg), len(wf))
            if n_total > 1 and cg["dap"].iloc[n_total - 1] == 0:
                n_total -= 1

            fc_start = max(0, n_total - metric_days)
            tr = wf["Tr"].values[fc_start:n_total]
            tr_pot = wf["TrPot"].values[fc_start:n_total]
            ks = np.where(tr_pot > 0.01, tr / tr_pot, 1.0)

            max_stress = float(1.0 - np.nanmin(ks)) if len(ks) else 0.0
            n_stress_days = int((ks < stress_threshold).sum())
            cc_loss = float(
                (
                    cg["canopy_cover_ns"].values[fc_start:n_total]
                    - cg["canopy_cover"].values[fc_start:n_total]
                ).mean()
            )

            member_max_stress.append(max_stress)
            member_stress_days.append(n_stress_days)
            member_yield_impact.append(cc_loss)

        arr_stress = np.asarray(member_max_stress, dtype=float)
        arr_days = np.asarray(member_stress_days, dtype=float)
        results.append(
            CandidateResult(
                candidate=candidate,
                member_max_stress=member_max_stress,
                member_stress_days=member_stress_days,
                member_yield_impact=member_yield_impact,
                mean_max_stress=float(np.nanmean(arr_stress)),
                median_max_stress=float(np.nanmedian(arr_stress)),
                pct_members_avoid_stress=float(np.nanmean(arr_stress < (1.0 - stress_threshold))),
                mean_stress_days=float(np.nanmean(arr_days)),
            )
        )

    results.sort(key=lambda r: (-r.pct_members_avoid_stress, r.mean_max_stress))
    return results


def compare_full_vs_branching_2level_on_subset(
    *,
    historical_forcing: DailyForcing,
    arome_forcing: DailyForcing,
    member_forcings: dict[int, DailyForcing],
    parcel: ParcelConfig,
    terrain: TerrainParams,
    soil_profile: SoilProfile,
    sim_start: date,
    today: date,
    candidates: list[IrrigationCandidate],
    n_candidates: int = 10,
    n_members: int = 5,
    stress_threshold: float = 0.9,
    arome_days: int = 2,
) -> pd.DataFrame:
    """Compare original full-rerun evaluator with two-level branching.

    Use this before making evaluate_candidates_ensemble_branching_2level the
    default path in the pipeline.
    """
    sample_candidates = candidates[:n_candidates]
    sample_member_ids = list(member_forcings)[:n_members]
    sample_member_forcings = {
        member_id: member_forcings[member_id] for member_id in sample_member_ids
    }

    full_results = _evaluate_candidates_ensemble_full_reference_serial(
        historical_forcing=historical_forcing,
        arome_forcing=arome_forcing,
        member_forcings=sample_member_forcings,
        parcel=parcel,
        terrain=terrain,
        soil_profile=soil_profile,
        sim_start=sim_start,
        today=today,
        candidates=sample_candidates,
        stress_threshold=stress_threshold,
    )

    branch_results = evaluate_candidates_ensemble_branching_2level(
        historical_forcing=historical_forcing,
        arome_forcing=arome_forcing,
        member_forcings=sample_member_forcings,
        parcel=parcel,
        terrain=terrain,
        soil_profile=soil_profile,
        sim_start=sim_start,
        today=today,
        candidates=sample_candidates,
        workers=1,
        stress_threshold=stress_threshold,
        arome_days=arome_days,
    )

    full_by_label = {result.candidate.label: result for result in full_results}
    branch_by_label = {result.candidate.label: result for result in branch_results}

    rows = []
    for label in sorted(set(full_by_label) & set(branch_by_label)):
        full = full_by_label[label]
        branch = branch_by_label[label]
        rows.append(
            {
                "candidate": label,
                "full_mean_max_stress": full.mean_max_stress,
                "branch_mean_max_stress": branch.mean_max_stress,
                "abs_delta_mean_max_stress": abs(full.mean_max_stress - branch.mean_max_stress),
                "full_mean_stress_days": full.mean_stress_days,
                "branch_mean_stress_days": branch.mean_stress_days,
                "abs_delta_mean_stress_days": abs(full.mean_stress_days - branch.mean_stress_days),
                "full_avoid_pct": full.pct_members_avoid_stress,
                "branch_avoid_pct": branch.pct_members_avoid_stress,
                "abs_delta_avoid_pct": abs(
                    full.pct_members_avoid_stress - branch.pct_members_avoid_stress
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_delta_mean_max_stress", ascending=False)


def recommend_from_optimizer(
    results: list[CandidateResult],
    water_cost_eur_mm: float = 2.5,
    top_x: int = 1,
) -> list[dict]:
    """Pick the best irrigation action from optimizer results.

    Selection logic:
    1. If no-irrigation already avoids stress in >80% of members → don't irrigate
    2. Otherwise, pick the cheapest action that avoids stress in >70% of members
       (cheapest = lowest total water, then fewest events)
    3. If no action avoids stress in >70%, pick the one that minimizes mean stress

    Returns a dict with the recommendation and comparison.
    """
    no_irr = next((r for r in results if r.candidate.is_rainfed), None)

    if no_irr and no_irr.pct_members_avoid_stress > 0.80:
        return [
            {
                "action": "No irrigation needed",
                "reason": f"{no_irr.pct_members_avoid_stress:.0%} of members avoid stress without irrigation",
                "candidate": no_irr.candidate,
                "avoid_stress_pct": no_irr.pct_members_avoid_stress,
                "mean_stress_days": no_irr.mean_stress_days,
                "all_results": results,
            }
        ]

    # Find cheapest action that avoids stress in >70% of members
    good = [r for r in results if r.pct_members_avoid_stress > 0.70 and not r.candidate.is_rainfed]
    if good:
        # Sort by total water (cheapest), then fewest events, then earliest start
        good.sort(
            key=lambda r: (r.candidate.total_mm, r.candidate.n_events, r.candidate.day_offset)
        )
        best_results = good[:top_x]
        no_irr_days = no_irr.mean_stress_days if no_irr else float("nan")
        return [
            {
                "action": best.candidate.label,
                "reason": (
                    f"{best.pct_members_avoid_stress:.0%} of members avoid stress. "
                    f"Mean stress days: {best.mean_stress_days:.1f} vs "
                    f"{no_irr_days:.1f} without irrigation"
                ),
                "candidate": best.candidate,
                "avoid_stress_pct": best.pct_members_avoid_stress,
                "mean_stress_days": best.mean_stress_days,
                "cost_eur_ha": best.candidate.total_mm * water_cost_eur_mm,
                "all_results": results,
            }
            for best in best_results
        ]

    # Nothing avoids stress well — pick action that minimizes stress
    top_results = results[:top_x]

    return [
        {
            "action": f"{best.candidate.label} (limited benefit)",
            "reason": (
                f"No action avoids stress in >70% of members. Best option: "
                f"{best.pct_members_avoid_stress:.0%} avoid stress, "
                f"mean {best.mean_stress_days:.1f} stress days"
            ),
            "candidate": best.candidate,
            "avoid_stress_pct": best.pct_members_avoid_stress,
            "mean_stress_days": best.mean_stress_days,
            "all_results": results,
        }
        for best in top_results
    ]
