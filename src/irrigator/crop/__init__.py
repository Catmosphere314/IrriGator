"""IrriGator Block 4 — Crop phenology.

Tracks maize growth via GDD accumulation and provides time-varying
Kc (crop coefficient) and Z_r (root depth) for the water balance.
"""

from irrigator.crop.phenology import CropParams, CropState, advance_crop, daily_gdd

__all__ = ["CropParams", "CropState", "advance_crop", "daily_gdd"]
