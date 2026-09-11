"""AquaCrop water-stress diagnostics and irrigation trigger helpers.

The optimizer should not infer agronomic stress from a single arbitrary
``Tr / TrPot`` threshold.  AquaCrop already defines crop-specific depletion
thresholds for canopy expansion, stomatal closure, early senescence and
pollination.  This module reconstructs those daily thresholds and provides the
root-zone depletion diagnostics needed to use them as irrigation-date triggers.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd


STRESS_PROCESSES = {
    "expansion": 0,
    "stomatal": 1,
    "senescence": 2,
    "pollination": 3,
}

TRIGGER_PROCESSES = {"auto", *STRESS_PROCESSES}


def dynamic_water_stress_thresholds(
    crop,
    et0,
    *,
    t_early_sen=None,
) -> pd.DataFrame:
    """Compute AquaCrop daily water-stress depletion thresholds.

    Parameters
    ----------
    crop
        AquaCrop ``Crop`` object.
    et0
        Daily reference evapotranspiration, mm/day.
    t_early_sen
        Optional daily AquaCrop early-senescence counter.  If supplied, the
        state-dependent adjustment to the senescence threshold is reproduced.

    Returns
    -------
    pandas.DataFrame
        Thresholds expressed as fractions of total available water (TAW), where
        0 is field capacity and 1 is wilting point.
    """
    et0 = np.asarray(et0, dtype=float)

    if et0.ndim != 1:
        raise ValueError("et0 must be one-dimensional")

    n = len(et0)

    p_up_base = np.asarray(crop.p_up, dtype=float)
    p_lo_base = np.asarray(crop.p_lo, dtype=float)

    if p_up_base.shape != (4,) or p_lo_base.shape != (4,):
        raise ValueError("Expected AquaCrop crop.p_up and crop.p_lo to contain 4 thresholds.")

    p_up = np.tile(p_up_base, (n, 1))
    p_lo = np.tile(p_lo_base, (n, 1))

    # AquaCrop ET0 adjustment: canopy expansion, stomatal closure and
    # senescence only. Pollination is not ET0-adjusted.
    if int(crop.ETadj) == 1:
        et_term = 0.04 * (5.0 - et0[:, None])

        original_up = p_up[:, :3].copy()
        original_lo = p_lo[:, :3].copy()

        p_up[:, :3] = original_up + et_term * np.log10(10.0 - 9.0 * original_up)
        p_lo[:, :3] = original_lo + et_term * np.log10(10.0 - 9.0 * original_lo)

    # AquaCrop further modifies the senescence onset threshold once early
    # senescence has actually begun.
    if t_early_sen is not None:
        t_early_sen = np.asarray(t_early_sen, dtype=float)

        if len(t_early_sen) != n:
            raise ValueError("t_early_sen and et0 must have the same length")

        early_senescence = t_early_sen > 0
        p_up[early_senescence, 2] *= 1.0 - float(crop.beta) / 100.0

    p_up = np.clip(p_up, 0.0, 1.0)
    p_lo = np.clip(p_lo, 0.0, 1.0)

    return pd.DataFrame(
        {
            # Stress onset
            "p_exp_start": p_up[:, 0],
            "p_sto_start": p_up[:, 1],
            "p_sen_start": p_up[:, 2],
            "p_pol_start": p_up[:, 3],
            # Maximum stress
            "p_exp_full": p_lo[:, 0],
            "p_sto_full": p_lo[:, 1],
            "p_sen_full": p_lo[:, 2],
            "p_pol_full": p_lo[:, 3],
        }
    )


def _integrated_storage(
    soil_profile: pd.DataFrame,
    depths_m: np.ndarray,
    theta_column: str,
) -> np.ndarray:
    """Integrate a volumetric water-content threshold to each root depth."""
    profile = soil_profile.copy()

    required = {"dzsum", "dz", theta_column}
    missing = required - set(profile.columns)
    if missing:
        raise ValueError(f"AquaCrop soil profile is missing required columns: {sorted(missing)}")

    profile = profile.sort_values("dzsum")
    top = (profile["dzsum"] - profile["dz"]).to_numpy(dtype=float)
    bottom = profile["dzsum"].to_numpy(dtype=float)
    theta = profile[theta_column].to_numpy(dtype=float)

    profile_depth = float(bottom[-1])
    result = np.empty(len(depths_m), dtype=float)

    for i, depth in enumerate(np.asarray(depths_m, dtype=float)):
        depth = float(np.clip(depth, 0.0, profile_depth))
        overlap = np.clip(
            np.minimum(bottom, depth) - top,
            0.0,
            bottom - top,
        )
        result[i] = float(np.sum(theta * overlap * 1000.0))

    return result


def root_zone_depletion_fraction(
    *,
    wr_mm,
    z_root_m,
    soil_profile: pd.DataFrame,
    crop_zmin_m: float,
) -> np.ndarray:
    """Return root-zone depletion ``Dr / TAW`` from AquaCrop outputs.

    ``Wr`` is AquaCrop root-zone water storage.  Field-capacity and wilting-point
    storage are integrated over the same dynamic root depth so the diagnostic is
    directly comparable with AquaCrop's ``p_up`` stress thresholds.
    """
    wr_mm = np.asarray(wr_mm, dtype=float)
    z_root_m = np.asarray(z_root_m, dtype=float)

    if wr_mm.ndim != 1 or z_root_m.ndim != 1:
        raise ValueError("wr_mm and z_root_m must be one-dimensional")
    if len(wr_mm) != len(z_root_m):
        raise ValueError("wr_mm and z_root_m must have the same length")

    profile_depth = float(soil_profile["dzsum"].max())
    roots = np.clip(
        np.maximum(z_root_m, float(crop_zmin_m)),
        0.0,
        profile_depth,
    )

    wr_fc = _integrated_storage(soil_profile, roots, "th_fc")
    wr_wp = _integrated_storage(soil_profile, roots, "th_wp")
    taw = wr_fc - wr_wp

    depletion = np.divide(
        wr_fc - wr_mm,
        taw,
        out=np.full(len(wr_mm), np.nan, dtype=float),
        where=taw > 1e-12,
    )

    return depletion


def active_water_stress_trigger_thresholds(
    crop,
    et0,
    *,
    gdd_cum,
    dap,
    process: str = "auto",
    margin_fraction: float = 0.0,
) -> pd.DataFrame:
    """Select the biologically relevant AquaCrop stress-onset threshold.

    The returned ``trigger_threshold`` is a fraction of TAW.  In ``auto`` mode
    the process changes with crop development:

    * vegetative/canopy development -> canopy-expansion threshold;
    * flowering/pollination -> the more conservative of pollination and
      stomatal thresholds;
    * grain filling/full canopy -> stomatal threshold;
    * senescence -> the more conservative of senescence and stomatal thresholds.

    ``margin_fraction`` is an optional *depletion* safety margin.  A positive
    value triggers earlier, e.g. 0.03 triggers 3 percentage points of TAW before
    AquaCrop's stress-onset threshold.  The default 0 uses AquaCrop's threshold
    directly and introduces no additional arbitrary ``Ks`` calibration.
    """
    if process not in TRIGGER_PROCESSES:
        raise ValueError(f"process must be one of {sorted(TRIGGER_PROCESSES)}; got {process!r}")
    if not 0.0 <= margin_fraction < 1.0:
        raise ValueError("margin_fraction must be in [0, 1).")

    gdd_cum = np.asarray(gdd_cum, dtype=float)
    dap = np.asarray(dap, dtype=float)
    et0 = np.asarray(et0, dtype=float)

    if not (gdd_cum.ndim == dap.ndim == et0.ndim == 1):
        raise ValueError("gdd_cum, dap and et0 must be one-dimensional")
    if not (len(gdd_cum) == len(dap) == len(et0)):
        raise ValueError("gdd_cum, dap and et0 must have the same length")

    thresholds = dynamic_water_stress_thresholds(crop, et0)
    n = len(et0)

    trigger = np.full(n, np.nan, dtype=float)
    process_name = np.full(n, "inactive", dtype=object)

    active = np.isfinite(dap) & (dap > 0)

    if process != "auto":
        source_column = {
            "expansion": "p_exp_start",
            "stomatal": "p_sto_start",
            "senescence": "p_sen_start",
            "pollination": "p_pol_start",
        }[process]
        trigger[active] = thresholds.loc[active, source_column].to_numpy(dtype=float)
        process_name[active] = process
    else:
        # AquaCrop phenology inputs are GDD when CalendarType=2 and calendar
        # days when CalendarType=1.  The same stage logic therefore works for
        # both representations.
        stage = gdd_cum if int(crop.CalendarType) == 2 else dap

        hi_start = float(getattr(crop, "HIstart", np.inf))
        flowering = float(getattr(crop, "Flowering", 0.0))
        senescence = float(getattr(crop, "Senescence", np.inf))

        canopy_dev_end = float(getattr(crop, "CanopyDevEnd", 0.0))
        if not np.isfinite(canopy_dev_end) or canopy_dev_end <= 0:
            # ``CanopyDevEnd`` is calculated internally by AquaCrop and can be
            # zero on the user Crop object.  HIstart is a conservative fallback
            # for fruit/grain crops: before yield formation, protecting canopy
            # expansion is preferable to waiting for stomatal limitation.
            canopy_dev_end = hi_start

        crop_type = int(getattr(crop, "CropType", 0))
        has_flowering = crop_type == 3 and np.isfinite(flowering) and flowering > 0
        flowering_end = hi_start + flowering if has_flowering else hi_start

        for i in np.flatnonzero(active):
            s = float(stage[i])

            # During flowering, protect whichever of pollination or stomatal
            # response starts first.
            if has_flowering and hi_start <= s < flowering_end:
                candidates = {
                    "pollination": float(thresholds.iloc[i]["p_pol_start"]),
                    "stomatal": float(thresholds.iloc[i]["p_sto_start"]),
                }
                chosen = min(candidates, key=candidates.get)
                trigger[i] = candidates[chosen]
                process_name[i] = chosen

            elif s < canopy_dev_end:
                trigger[i] = float(thresholds.iloc[i]["p_exp_start"])
                process_name[i] = "expansion"

            elif s >= senescence:
                candidates = {
                    "senescence": float(thresholds.iloc[i]["p_sen_start"]),
                    "stomatal": float(thresholds.iloc[i]["p_sto_start"]),
                }
                chosen = min(candidates, key=candidates.get)
                trigger[i] = candidates[chosen]
                process_name[i] = chosen

            else:
                trigger[i] = float(thresholds.iloc[i]["p_sto_start"])
                process_name[i] = "stomatal"

    finite = np.isfinite(trigger)
    trigger[finite] = np.clip(trigger[finite] - float(margin_fraction), 0.0, 1.0)

    result = thresholds.copy()
    result["trigger_threshold"] = trigger
    result["trigger_process"] = process_name
    return result


def first_water_stress_trigger_date(
    *,
    dates: Sequence[date],
    crop,
    soil_profile: pd.DataFrame,
    wr_mm,
    z_root_m,
    et0,
    gdd_cum,
    dap,
    not_before: date | None = None,
    process: str = "auto",
    margin_fraction: float = 0.0,
) -> tuple[date | None, str | None]:
    """Return the first AquaCrop depletion-threshold crossing.

    This is intended as an irrigation *date proposal* heuristic.  Candidate
    acceptance remains a separate optimization constraint (yield for a full
    harvest horizon, crop-state preservation for a partial forecast horizon).
    """
    dates = list(dates)
    n = len(dates)

    arrays = [wr_mm, z_root_m, et0, gdd_cum, dap]
    if any(len(np.asarray(values)) != n for values in arrays):
        raise ValueError("All stress-trigger inputs must share the same length")

    depletion = root_zone_depletion_fraction(
        wr_mm=wr_mm,
        z_root_m=z_root_m,
        soil_profile=soil_profile,
        crop_zmin_m=float(getattr(crop, "Zmin", 0.0)),
    )

    thresholds = active_water_stress_trigger_thresholds(
        crop,
        et0,
        gdd_cum=gdd_cum,
        dap=dap,
        process=process,
        margin_fraction=margin_fraction,
    )

    trigger = thresholds["trigger_threshold"].to_numpy(dtype=float)
    processes = thresholds["trigger_process"].to_numpy(dtype=object)

    for i, d in enumerate(dates):
        if not_before is not None and d < not_before:
            continue
        if not np.isfinite(depletion[i]) or not np.isfinite(trigger[i]):
            continue
        if depletion[i] >= trigger[i]:
            return d, str(processes[i])

    return None, None
