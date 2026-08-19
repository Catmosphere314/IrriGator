"""AquaCrop water-stress diagnostics.

Utilities for reconstructing the dynamic AquaCrop soil-water depletion
thresholds from Crop parameters and daily reference evapotranspiration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


STRESS_PROCESSES = {
    "expansion": 0,
    "stomatal": 1,
    "senescence": 2,
    "pollination": 3,
}


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
        AquaCrop Crop object.

    et0
        Daily reference evapotranspiration, mm/day.

    t_early_sen
        Optional daily AquaCrop early-senescence counter.

        If supplied, the state-dependent adjustment to the
        senescence threshold is reproduced.

    Returns
    -------
    pd.DataFrame
        Thresholds expressed as fractions of TAW:

        0 = field capacity
        1 = wilting point
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

    # AquaCrop ET0 adjustment:
    # canopy expansion, stomatal closure and senescence only.
    if int(crop.ETadj) == 1:
        et_term = 0.04 * (5.0 - et0[:, None])

        original_up = p_up[:, :3].copy()
        original_lo = p_lo[:, :3].copy()

        p_up[:, :3] = original_up + et_term * np.log10(10.0 - 9.0 * original_up)

        p_lo[:, :3] = original_lo + et_term * np.log10(10.0 - 9.0 * original_lo)

    # AquaCrop further modifies the senescence onset threshold
    # once early senescence has actually begun.
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
