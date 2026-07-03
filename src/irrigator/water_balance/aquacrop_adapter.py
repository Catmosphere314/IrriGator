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

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

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



def parcel_to_crop(parcel: ParcelConfig) -> Crop:
    """Convert parcel crop config to AquaCrop Crop object.

    Uses AquaCrop's built-in Maize parameters (GDD thresholds,
    canopy expansion coefficients, root deepening rate, etc.).
    """
    crop_cfg = parcel.crop
    crop_type = crop_cfg.get("type", "grain_maize")

    if crop_type not in {"grain_maize", "maize"}:
        raise ValueError(
            f"Unsupported AquaCrop crop: {crop_type}. Currently only grain_maize is supported."
        )

    planting = pd.Timestamp(crop_cfg["planting_date"])
    planting_mmdd = planting.strftime("%m/%d")

    harvest = crop_cfg.get("expected_harvest")
    harvest_mmdd = pd.Timestamp(harvest).strftime("%m/%d") if harvest else None

    return Crop("Maize", planting_date=planting_mmdd, harvest_date=harvest_mmdd)


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
    soil = soil_to_aquacrop(soil_profile)
    crop = parcel_to_crop(parcel)

    iwc = initial_water_content or InitialWaterContent(value=["FC"])
    irr = irrigation_management or parcel_to_irrigation(parcel)

    model = AquaCropModel(
        sim_start_time=sim_start.strftime("%Y/%m/%d"),
        sim_end_time=sim_end.strftime("%Y/%m/%d"),
        weather_df=weather,
        soil=soil,
        crop=crop,
        initial_water_content=iwc,
        irrigation_management=irr,
    )

    model.run_model(till_termination=True)

    result = AquaCropResult(
        final_results=model.get_simulation_results(),
        crop_growth=model.get_crop_growth(),
        water_flux=model.get_water_flux(),
        water_storage=model.get_water_storage(),
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
    soil = soil_to_aquacrop(soil_profile)
    crop = parcel_to_crop(parcel)
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

        model = AquaCropModel(
            sim_start_time=sim_start.strftime("%Y/%m/%d"),
            sim_end_time=full_end.strftime("%Y/%m/%d"),
            weather_df=weather,
            soil=soil,
            crop=crop,
            initial_water_content=iwc,
            irrigation_management=parcel_to_irrigation(parcel),
        )
        model.run_model(till_termination=True)

        result = AquaCropResult(
            final_results=model.get_simulation_results(),
            crop_growth=model.get_crop_growth(),
            water_flux=model.get_water_flux(),
            water_storage=model.get_water_storage(),
        )

        stress = result.daily_stress

        # Extract forecast portion (last n_forecast days, excluding terminal zero row)
        n_total = len(stress)
        if n_total > 1 and stress["dap"].iloc[-1] == 0:
            n_total -= 1

        forecast_start_idx = max(0, n_total - n_forecast)
        forecast_slice = stress.iloc[forecast_start_idx:n_total]

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
    max_dose = irr_cfg.get("max_dose_mm",35)
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


def recommend_from_optimizer(
    results: list[CandidateResult],
    water_cost_eur_mm: float = 2.5,
) -> dict:
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
        return {
            "action": "No irrigation needed",
            "reason": f"{no_irr.pct_members_avoid_stress:.0%} of members avoid stress without irrigation",
            "candidate": no_irr.candidate,
            "avoid_stress_pct": no_irr.pct_members_avoid_stress,
            "mean_stress_days": no_irr.mean_stress_days,
            "all_results": results,
        }

    # Find cheapest action that avoids stress in >70% of members
    good = [r for r in results if r.pct_members_avoid_stress > 0.70 and not r.candidate.is_rainfed]
    if good:
        # Sort by total water (cheapest), then fewest events, then earliest start
        good.sort(
            key=lambda r: (r.candidate.total_mm, r.candidate.n_events, r.candidate.day_offset)
        )
        best = good[0]
        no_irr_days = no_irr.mean_stress_days if no_irr else float("nan")
        return {
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

    # Nothing avoids stress well — pick action that minimizes stress
    best = results[0]  # already sorted by best outcome
    return {
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
