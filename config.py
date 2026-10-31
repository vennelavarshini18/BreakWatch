from __future__ import annotations

SEASONS = {"kharif": (6, 7, 8, 9, 10), "rabi": (11, 12, 1, 2), "summer": (3, 4, 5)}


def season_of(month: int) -> str:
    month = int(month)
    if month in SEASONS["kharif"]:
        return "kharif"
    if month in SEASONS["rabi"]:
        return "rabi"
    if month in SEASONS["summer"]:
        return "summer"
    raise ValueError(f"Invalid month: {month}")


BASELINE_HALF_WINDOW_DAYS = 10
BASELINE_START_YEAR = {"weather": 2010, "modis_ndvi": 2010, "s2": 2018, "s1": 2018}
MIN_BASELINE_YEARS = {"weather": 5, "modis_ndvi": 5, "s2": 3, "s1": 3}
MIN_BASELINE_OBS = 6
SIGMA_FLOOR = {
    "s2_ndvi": 0.03,
    "s2_ndmi": 0.03,
    "modis_ndvi": 0.03,
    "s1_vh": 0.5,
    "s1_vv": 0.5,
    "soil_moist": 0.01,
}
SENSOR_SIGN = {
    "s2_ndvi": -1,
    "s2_ndmi": -1,
    "modis_ndvi": -1,
    "soil_moist": -1,
    "s1_vh": -1,
    "s1_vv": -1,
}
SENSOR_PROVISIONAL = {"s1_vh", "s1_vv"}


def baseline_group(sensor: str) -> str:
    if sensor.startswith("s2_"):
        return "s2"
    if sensor.startswith("s1_"):
        return "s1"
    if sensor == "modis_ndvi":
        return "modis_ndvi"
    if sensor == "soil_moist":
        return "weather"
    raise ValueError(f"Unsupported sensor: {sensor}")


MODIS_COMPOSITE_DAYS = 16
SM_DAILY_MIN_HOURS = 18
FORECAST_HORIZON_DAYS = 14
STRESS_THRESHOLD = 1.0
RISK_P_WATCH = 0.2
RISK_P_HIGH = 0.5
INSUFFICIENT_STD = 0.9
TRAIN_YEARS = (2015, 2020)
CALIB_YEARS = (2021, 2022)
TEST_YEARS = (2023, 2025)
SKILL_MIN = 0.05
COVERAGE_MIN = 0.75
N_TEST_MIN = 200
OPTICAL_MIN_VALID_FRAC = 0.4
STALE_DAYS = 12
MAX_FIELD_HA = 500
MIN_FIELD_HA = 1
HTTP_TIMEOUT_S = 8
HTTP_BACKOFFS_S = (1, 3)
