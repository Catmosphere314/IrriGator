"""Water balance state dataclass.

FAO-56 tracks root zone depletion (Dr) rather than absolute water
content.  This is cleaner because:
- Dr = 0 → soil at field capacity (full)
- Dr = TAW → soil at wilting point (empty)
- Stress starts when Dr > RAW = p × TAW
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class WaterBalanceState:
    """Complete water balance state for one day at one parcel.

    Water quantities in mm.  Depletion-based tracking per FAO-56.

    Attributes
    ----------
    date : the day
    et0 : reference evapotranspiration [mm]
    kc : crop coefficient
    etc_pot : potential crop ET = Kc × ET0 [mm]
    etc_act : actual crop ET = Ks × Kc × ET0 [mm]
    precip : precipitation [mm]
    irrigation : irrigation applied [mm]
    runoff : surface runoff [mm]
    drainage : deep percolation [mm]
    depletion : root zone depletion Dr [mm] (0 = at FC, TAW = at WP)
    taw : total available water (FC − WP) × Z_r × 1000 [mm]
    raw : readily available water = p × TAW [mm]
    stress_coeff : Ks [0−1] (1 = no stress)
    z_root : rooting depth [m]
    gdd : accumulated GDD [°C·days]
    crop_stage : growth stage name
    """

    date: date
    et0: float
    kc: float
    etc_pot: float
    etc_act: float
    precip: float
    irrigation: float
    runoff: float
    drainage: float
    depletion: float
    taw: float
    raw: float
    stress_coeff: float
    z_root: float
    gdd: float
    crop_stage: str

    @property
    def fraction_available(self) -> float:
        """Fraction of available water remaining [0−1].
        1.0 = at field capacity, 0.0 = at wilting point."""
        if self.taw <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - self.depletion / self.taw))

    @property
    def is_stressed(self) -> bool:
        return self.stress_coeff < 1.0
