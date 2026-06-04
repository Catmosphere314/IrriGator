"""Crop phenology: GDD accumulation, growth stage, Kc, and root depth.

Tracks maize development through Growing Degree Days (GDD) and provides
the time-varying crop coefficient (Kc) and rooting depth (Z_r) that
the water balance needs at each daily timestep.

The crop state evolves daily:
    GDD(t) = GDD(t-1) + max(0, (Tmax + Tmin)/2 - T_base)
    stage(t) = f(GDD(t))
    Kc(t) = f(stage(t))
    Z_r(t) = f(stage(t))

References
----------
- FAO-56 (Allen et al., 1998): Kc values, stage definitions
- Arvalis: maturity group GDD thresholds for French maize varieties
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import yaml

logger = logging.getLogger(__name__)

# Default T_base for maize GDD
T_BASE = 10.0
T_UPPER = 30.0  # GDD contribution capped above this

# FAO-56 Kc values for grain maize
KC_INITIAL = 0.30
KC_MID = 1.20
KC_END = 0.35

# Default rooting depth
Z_ROOT_INITIAL = 0.15  # m at emergence
Z_ROOT_MAX = 1.20  # m at mid-season


@dataclass
class CropState:
    """Crop state at one point in time.

    Attributes
    ----------
    date : current date
    gdd : accumulated GDD since planting [°C·days]
    stage : growth stage name
    kc : crop coefficient (dimensionless)
    z_root : current rooting depth [m]
    is_growing : True if between emergence and maturity
    """

    date: date
    gdd: float
    stage: str
    kc: float
    z_root: float
    is_growing: bool


@dataclass
class CropParams:
    """Crop parameters loaded from config.

    GDD thresholds define stage transitions.
    """

    planting_date: date
    t_base: float
    t_upper: float
    # GDD thresholds
    gdd_emergence: float  # GDD to emerge (~50-80)
    gdd_end_initial: float  # end of initial stage
    gdd_end_development: float  # tasseling
    gdd_end_midseason: float  # end of grain fill
    gdd_maturity: float  # physiological maturity
    # Kc values
    kc_initial: float
    kc_mid: float
    kc_end: float
    # Root depth
    z_root_initial: float
    z_root_max: float
    # Stress
    p: float  # depletion fraction (FAO-56)

    @classmethod
    def from_config(
        cls, parcel_crop: dict, crop_params_path: str = "configs/crop_parameters.yaml"
    ) -> CropParams:
        """Build CropParams from parcel config + crop parameters YAML."""
        # Load crop parameters
        with open(crop_params_path) as f:
            crop_cfg = yaml.safe_load(f)["grain_maize"]

        maturity_group = parcel_crop.get("maturity_group", "C1")
        gdd_thresholds = crop_cfg["gdd_by_maturity_group"][maturity_group]

        planting_str = parcel_crop["planting_date"]
        if isinstance(planting_str, str):
            planting = date.fromisoformat(planting_str)
        else:
            planting = planting_str

        return cls(
            planting_date=planting,
            t_base=crop_cfg.get("t_base_c", T_BASE),
            t_upper=crop_cfg.get("t_upper_c", T_UPPER),
            gdd_emergence=60,  # ~60 GDD to emerge, not in FAO-56 but standard
            gdd_end_initial=gdd_thresholds["end_initial"],
            gdd_end_development=gdd_thresholds["end_development"],
            gdd_end_midseason=gdd_thresholds["end_midseason"],
            gdd_maturity=gdd_thresholds["maturity"],
            kc_initial=crop_cfg["kc"]["initial"],
            kc_mid=crop_cfg["kc"]["mid"],
            kc_end=crop_cfg["kc"]["end"],
            z_root_initial=crop_cfg["rooting_depth"]["at_emergence_m"],
            z_root_max=crop_cfg["rooting_depth"]["max_m"],
            p=crop_cfg["stress"]["p"],
        )


# ---------------------------------------------------------------------------
# Daily GDD
# ---------------------------------------------------------------------------


def daily_gdd(
    t_min: float, t_max: float, t_base: float = T_BASE, t_upper: float = T_UPPER
) -> float:
    """Compute GDD for one day.

    Uses the standard method: GDD = max(0, (Tmax + Tmin)/2 - T_base)
    with Tmax capped at T_upper and Tmin floored at T_base.
    """
    t_max_c = min(t_max, t_upper)
    t_min_c = max(t_min, t_base)
    t_mean = (t_max_c + t_min_c) / 2.0
    return max(0.0, t_mean - t_base)


# ---------------------------------------------------------------------------
# Stage, Kc, and root depth from GDD
# ---------------------------------------------------------------------------


def _get_stage(gdd: float, params: CropParams) -> str:
    """Determine growth stage from accumulated GDD."""
    if gdd < params.gdd_emergence:
        return "pre_emergence"
    elif gdd < params.gdd_end_initial:
        return "initial"
    elif gdd < params.gdd_end_development:
        return "development"
    elif gdd < params.gdd_end_midseason:
        return "mid"
    elif gdd < params.gdd_maturity:
        return "late"
    else:
        return "mature"


def _get_kc(gdd: float, params: CropParams) -> float:
    """Interpolate Kc based on GDD and growth stage.

    Kc is constant during initial and mid stages, and linearly
    interpolated during development and late stages.
    """
    if gdd < params.gdd_emergence:
        return 0.0  # no crop yet
    elif gdd < params.gdd_end_initial:
        return params.kc_initial
    elif gdd < params.gdd_end_development:
        # Linear ramp from kc_initial to kc_mid
        frac = (gdd - params.gdd_end_initial) / (
            params.gdd_end_development - params.gdd_end_initial
        )
        return params.kc_initial + frac * (params.kc_mid - params.kc_initial)
    elif gdd < params.gdd_end_midseason:
        return params.kc_mid
    elif gdd < params.gdd_maturity:
        # Linear decline from kc_mid to kc_end
        frac = (gdd - params.gdd_end_midseason) / (params.gdd_maturity - params.gdd_end_midseason)
        return params.kc_mid + frac * (params.kc_end - params.kc_mid)
    else:
        return params.kc_end


def _get_root_depth(gdd: float, params: CropParams) -> float:
    """Root depth increases linearly from emergence to end of development,
    then stays constant.
    """
    if gdd < params.gdd_emergence:
        return 0.0
    elif gdd < params.gdd_end_development:
        frac = (gdd - params.gdd_emergence) / (params.gdd_end_development - params.gdd_emergence)
        return params.z_root_initial + frac * (params.z_root_max - params.z_root_initial)
    else:
        return params.z_root_max


def advance_crop(
    current_date: date,
    t_min: float,
    t_max: float,
    gdd_prev: float,
    params: CropParams,
) -> CropState:
    """Advance crop state by one day.

    Parameters
    ----------
    current_date : today's date
    t_min, t_max : daily temperatures [°C]
    gdd_prev : accumulated GDD up to yesterday
    params : crop parameters

    Returns
    -------
    Updated CropState for today.
    """
    # Only accumulate GDD after planting
    print(params.planting_date)
    if current_date < params.planting_date:
        return CropState(
            date=current_date,
            gdd=0,
            stage="not_planted",
            kc=0,
            z_root=0,
            is_growing=False,
        )

    gdd_today = daily_gdd(t_min, t_max, params.t_base, params.t_upper)
    gdd_total = gdd_prev + gdd_today

    stage = _get_stage(gdd_total, params)
    kc = _get_kc(gdd_total, params)
    z_root = _get_root_depth(gdd_total, params)
    is_growing = stage not in ("not_planted", "pre_emergence", "mature")

    return CropState(
        date=current_date,
        gdd=gdd_total,
        stage=stage,
        kc=kc,
        z_root=z_root,
        is_growing=is_growing,
    )
