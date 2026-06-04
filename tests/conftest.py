"""Shared fixtures for IrriGator tests."""

import datetime

import pytest


@pytest.fixture
def sample_daily_forcing():
    """Typical summer day in Dordogne for testing ET0 and water balance."""
    return {
        "date": datetime.date(2026, 7, 15),
        "t_min": 16.5,  # °C
        "t_max": 32.0,  # °C
        "t_mean": 24.25,  # °C
        "dewpoint": 15.0,  # °C
        "wind_speed": 1.8,  # m/s at 2m
        "pressure": 101.0,  # kPa
        "rs": 25.0,  # MJ/m²/day
        "precip": 0.0,  # mm
    }


@pytest.fixture
def sample_soil_profile():
    """Alluvial soil typical of Dordogne valley, single-layer simplification."""
    return {
        "theta_fc": 0.30,  # m³/m³
        "theta_wp": 0.13,  # m³/m³
        "z_root": 1.0,  # m
        "total_awc": 170.0,  # mm = (0.30 - 0.13) * 1000
        "geology_class": "alluvium",
        "source": "eu_soilhydrogrids",
    }


@pytest.fixture
def sample_crop_state():
    """Mid-season maize at tasseling."""
    return {
        "planting_date": datetime.date(2026, 4, 20),
        "current_date": datetime.date(2026, 7, 15),
        "gdd_accumulated": 850.0,
        "stage": "mid",
        "kc": 1.20,
        "root_depth_m": 1.0,
    }
