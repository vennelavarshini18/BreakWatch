from datetime import date

import pytest

from contracts import FieldInput, ObsBundle, SensorObs, WeatherDay


def make_field(as_of: date, mode: str = "live") -> FieldInput:
    return FieldInput(
        polygon={
            "type": "Polygon",
            "coordinates": [
                [[78.0, 17.0], [78.1, 17.0], [78.1, 17.1], [78.0, 17.1], [78.0, 17.0]]
            ],
        },
        as_of=as_of,
        mode=mode,
    )


def test_obs_after_as_of_rejected():
    field = make_field(date(2024, 1, 15))
    with pytest.raises(ValueError):
        ObsBundle(
            field=field,
            obs=[
                SensorObs(
                    sensor="modis_ndvi",
                    day=date(2024, 1, 16),
                    value=0.5,
                    valid_frac=1.0,
                    n_baseline=5,
                    evidence_id="ev_001",
                )
            ],
        )


def test_weather_after_as_of_rejected():
    field = make_field(date(2024, 1, 15))
    with pytest.raises(ValueError):
        ObsBundle(
            field=field,
            weather_daily=[
                WeatherDay(
                    day=date(2024, 1, 16),
                    precip_mm=12.0,
                    evidence_id="ev_001",
                )
            ],
        )


def test_forecast_row_rejected_in_replay_and_accepted_in_live():
    replay_field = make_field(date(2024, 1, 15), mode="replay")
    with pytest.raises(ValueError):
        ObsBundle(
            field=replay_field,
            weather_daily=[
                WeatherDay(
                    day=date(2024, 1, 15),
                    precip_mm=12.0,
                    is_forecast=True,
                    evidence_id="ev_001",
                )
            ],
        )

    live_field = make_field(date(2024, 1, 15), mode="live")
    bundle = ObsBundle(
        field=live_field,
        weather_daily=[
            WeatherDay(
                day=date(2024, 1, 15),
                precip_mm=12.0,
                is_forecast=True,
                evidence_id="ev_001",
            )
        ],
    )
    assert bundle.weather_daily[0].is_forecast is True
