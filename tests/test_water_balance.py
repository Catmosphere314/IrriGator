"""Tests for ET0 computation — Block 3."""


def test_et0_reasonable_range(sample_daily_forcing):
    """ET0 on a hot summer day in Dordogne should be roughly 5-8 mm/day."""
    # TODO: implement once src/irrigator/water_balance/et0.py exists
    # from irrigator.water_balance.et0 import compute_et0
    # et0 = compute_et0(**sample_daily_forcing)
    # assert 3.0 < et0 < 10.0, f"ET0 = {et0} mm/day outside plausible range"
    pass


def test_et0_winter_low():
    """ET0 in January should be well below 2 mm/day."""
    # TODO
    pass


def test_gdd_accumulation(sample_daily_forcing):
    """GDD for a day with Tmin=16.5, Tmax=32 should be ~14.25 (base 10)."""
    t_mean = (sample_daily_forcing["t_min"] + sample_daily_forcing["t_max"]) / 2
    gdd = max(0, t_mean - 10.0)
    assert abs(gdd - 14.25) < 0.01


def test_soil_water_balance_conservation(sample_soil_profile):
    """Water balance should conserve: W(t) = W(t-1) + P + I - ETc - D - R."""
    # TODO: implement once bucket model exists
    pass
