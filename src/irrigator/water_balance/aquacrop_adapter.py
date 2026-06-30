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

    weather = pd.DataFrame({
        "MinTemp": t_min,
        "MaxTemp": t_max,
        "Precipitation": forcing.precip_mm.astype(np.float64),
        "ReferenceET": np.clip(et0, 0.0, None),
        "Date": pd.to_datetime(forcing.dates),
    })

    # Replace NaN with 0 for precipitation and ET0
    weather["Precipitation"] = weather["Precipitation"].fillna(0.0).clip(lower=0.0)
    weather["ReferenceET"] = weather["ReferenceET"].fillna(0.1).clip(lower=0.01)
    weather["MinTemp"] = weather["MinTemp"].fillna(method="ffill").fillna(10.0)
    weather["MaxTemp"] = weather["MaxTemp"].fillna(method="ffill").fillna(20.0)

    return weather


def soil_to_aquacrop(profile: SoilProfile) -> Soil:
    """Convert IrriGator SoilProfile to AquaCrop Soil object.

    Requires theta_s (saturation water content) in the profile.
    If not available, estimates it from field capacity.
    """
    dz = [(bottom - top) / 100.0 for top, bottom in profile.z_layers_cm]
    soil = Soil("custom", dz=dz)

    for i, (top, bottom) in enumerate(profile.z_layers_cm):
        thickness_m = (bottom - top) / 100.0

        th_wp = profile.theta_wp[i]
        th_fc = profile.theta_fc[i]

        # Saturation water content
        if hasattr(profile, "theta_s") and profile.theta_s is not None:
            if isinstance(profile.theta_s, list):
                th_s = profile.theta_s[i]
            else:
                th_s = profile.theta_s
        else:
            # Fallback: estimate from field capacity
            th_s = min(0.60, th_fc + 0.12)
            logger.warning(
                "Layer %d: theta_s not available, estimated %.3f from FC=%.3f. "
                "Add THS to SOIL_VARIABLES in esdac_loader.py for accurate values.",
                i, th_s, th_fc,
            )

        # Saturated hydraulic conductivity: SoilProfile stores cm/day,
        # AquaCrop expects mm/day
        ksat_mm_day = profile.k_sat[i] * 10.0

        # Penetrability: 100% unless restricted layer
        penetrability = 100

        soil.add_layer(thickness_m, th_wp, th_fc, th_s, ksat_mm_day, penetrability)

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
            f"Unsupported AquaCrop crop: {crop_type}. "
            f"Currently only grain_maize is supported."
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
        rows.append({
            "Date": pd.Timestamp(item["date"]),
            "Depth": float(item["amount_mm"]),
        })

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
    final_results: pd.DataFrame    # season summary (yield, irrigation)
    crop_growth: pd.DataFrame      # daily: canopy, biomass, root depth, GDD
    water_flux: pd.DataFrame       # daily: Tr, TrPot, Es, IrrDay, Wr, DeepPerc
    water_storage: pd.DataFrame    # daily: soil moisture per compartment

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

        return pd.DataFrame({
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
        })


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
        sim_start, sim_end, n_days,
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

    For each member:
    1. Concatenate historical forcing (Jan 1 → today) with member forecast
    2. Run AquaCrop for the full period
    3. Extract the forecast portion (today → end) of the results

    This ensures crop state at the forecast start is consistent
    across all members and the stress-growth coupling is preserved.

    Parameters
    ----------
    historical_forcing : DailyForcing from Jan 1 → today
    member_forcings : {member_id: DailyForcing} for each IFS ENS member
    parcel : parcel configuration
    terrain : terrain parameters
    soil_profile : soil hydraulic properties
    sim_start : simulation start (typically Jan 1)
    today : current date (forecast branches from here)
    stress_threshold : Ks below this counts as "stressed"
    """
    daily_stats: list[AquaCropEnsembleStats] = []

    # Determine forecast end from first member
    first_forcing = next(iter(member_forcings.values()))
    forecast_end = pd.Timestamp(first_forcing.dates[-1]).date()
    n_hist = len(historical_forcing.dates)
    n_forecast = first_forcing.n_days

    logger.info(
        "Running AquaCrop ensemble: %d members, %s → %s (forecast: %s → %s)",
        len(member_forcings), sim_start, forecast_end, today, forecast_end,
    )

    # Run each member
    all_member_ks: dict[int, np.ndarray] = {}
    all_member_cc: dict[int, np.ndarray] = {}
    all_member_wr: dict[int, np.ndarray] = {}
    all_member_precip: dict[int, np.ndarray] = {}
    member_dates = None

    for member_id, member_forcing in member_forcings.items():
        # Concatenate: historical (Jan 1 → today) + forecast (today+1 → end)
        full_forcing = historical_forcing.concat(member_forcing)
        full_end = pd.Timestamp(full_forcing.dates[-1]).date()

        # Run AquaCrop for full period (rainfed during forecast)
        result = run_aquacrop(
            forcing=full_forcing,
            parcel=parcel,
            terrain=terrain,
            soil_profile=soil_profile,
            sim_start=sim_start,
            sim_end=full_end,
            irrigation_management=IrrigationManagement(irrigation_method=0),
        )

        stress = result.daily_stress

        # Extract forecast portion (last n_forecast days, excluding terminal zero row)
        n_total = len(stress)
        # AquaCrop adds a zero row at the end — skip it
        if n_total > 1 and stress["dap"].iloc[-1] == 0:
            n_total -= 1

        forecast_start_idx = max(0, n_total - n_forecast)
        forecast_slice = stress.iloc[forecast_start_idx:n_total]

        all_member_ks[member_id] = forecast_slice["ks"].values
        all_member_cc[member_id] = forecast_slice["canopy_cover"].values
        all_member_wr[member_id] = forecast_slice["wr_mm"].values
        all_member_precip[member_id] = forecast_slice["precip_mm"].values

        if member_dates is None:
            member_dates = pd.to_datetime(
                first_forcing.dates[:len(forecast_slice)]
            )

    # Aggregate ensemble statistics per day
    n_days = min(len(v) for v in all_member_ks.values())

    for d in range(n_days):
        ks_arr = np.array([all_member_ks[m][d] for m in all_member_ks])
        cc_arr = np.array([all_member_cc[m][d] for m in all_member_cc])
        wr_arr = np.array([all_member_wr[m][d] for m in all_member_wr])
        pr_arr = np.array([all_member_precip[m][d] for m in all_member_precip])

        day_date = member_dates[d] if d < len(member_dates) else None

        daily_stats.append(AquaCropEnsembleStats(
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
        ))

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
        "Optimization complete: SMT=%s, yield=%.1f t/ha, "
        "irrigation=%d mm, profit=€%.0f/ha",
        [f"{s:.0f}" for s in optimal_smt],
        metrics["yield_t_ha"],
        metrics["seasonal_irrigation_mm"],
        metrics["profit_eur_ha"],
    )

    return metrics
