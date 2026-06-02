from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
REQUEST_TIMEOUT_SECONDS = 30


class ForecastError(Exception):
    """Base exception for forecast-building failures."""


class DataUnavailableError(ForecastError):
    """Raised when an external API cannot provide required data."""


class ModelInputError(ForecastError):
    """Raised when model input cannot be assembled correctly."""


# These are the only raw features your selected model needs
PREDICT_FEATURE_COLUMNS = [
    "cloud_cover_total",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "aerosol_optical_depth",
    "temperature",
    "dew_point",
    "pressure_6h_before",
    "pressure_trend",
    "relative_humidity",
    "pm25",
    "pm10",
    "wind_speed",
    "wind_direction",
]


@dataclass
class ForecastBuildResult:
    X_ready: pd.DataFrame
    sunset_time_today: str
    snapshot: dict[str, Any]
    missing_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "X_ready": self.X_ready,
            "sunset_time_today": self.sunset_time_today,
            "snapshot": self.snapshot,
            "missing_fields": self.missing_fields,
        }


def _build_retry_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        allowed_methods=frozenset(["GET"]),
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1.0,
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _find_closest_row(df: pd.DataFrame, target_dt: datetime) -> pd.Series:
    if df.empty:
        raise ModelInputError("DataFrame is empty.")

    temp = df.copy()
    temp["seconds_from_target"] = (
        temp["time_dt"] - pd.Timestamp(target_dt)
    ).abs().dt.total_seconds()

    idx = temp["seconds_from_target"].idxmin()
    return temp.loc[idx]


def _find_row_at_or_before(df: pd.DataFrame, target_dt: datetime) -> Optional[pd.Series]:
    if df.empty:
        return None

    eligible = df[df["time_dt"] <= pd.Timestamp(target_dt)].copy()
    if eligible.empty:
        return None

    eligible["seconds_from_target"] = (
        pd.Timestamp(target_dt) - eligible["time_dt"]
    ).dt.total_seconds()

    idx = eligible["seconds_from_target"].idxmin()
    return eligible.loc[idx]


def _log_request_failure(api_name: str, attempt: int, exc: Exception) -> None:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    print(
        f"[forecast] {api_name} request failed "
        f"attempt={attempt}/3 status={status_code} error={exc}"
    )


def _request_json_with_retries(
    url: str,
    params: dict[str, Any],
    api_name: str,
) -> dict[str, Any]:
    session = _build_retry_session()
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            response = session.get(
                url,
                params=params,
                timeout=(10, REQUEST_TIMEOUT_SECONDS),
            )

            if response.status_code >= 400:
                body_preview = response.text[:300].replace("\n", " ")
                raise requests.HTTPError(
                    f"{api_name} error status={response.status_code} body={body_preview}",
                    response=response,
                )

            return response.json()

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
            requests.RequestException,
        ) as exc:
            last_error = exc
            _log_request_failure(api_name=api_name, attempt=attempt, exc=exc)

            if attempt < 3:
                continue

    raise DataUnavailableError(
        f"Failed to fetch data from {api_name} after retries: {last_error}"
    )


def _require_keys(data: dict[str, Any], required_keys: list[str], context: str) -> None:
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise DataUnavailableError(f"Missing keys in {context}: {missing}")


def _fetch_weather_forecast(
    latitude: float,
    longitude: float,
    target_date: str,
    timezone: str,
) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "start_date": target_date,
        "end_date": target_date,
        "hourly": ",".join(
            [
                "cloud_cover",
                "cloud_cover_low",
                "cloud_cover_mid",
                "cloud_cover_high",
                "temperature_2m",
                "dew_point_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "daily": "sunset",
    }

    data = _request_json_with_retries(
        url=OPEN_METEO_FORECAST_URL,
        params=params,
        api_name="open-meteo forecast",
    )

    _require_keys(data, ["hourly", "daily"], "weather forecast response")
    _require_keys(data["hourly"], ["time"], "weather forecast hourly")
    _require_keys(data["daily"], ["sunset"], "weather forecast daily")

    return data


def _fetch_air_quality_forecast(
    latitude: float,
    longitude: float,
    target_date: str,
    timezone: str,
) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "start_date": target_date,
        "end_date": target_date,
        "hourly": ",".join(
            [
                "pm2_5",
                "pm10",
                "aerosol_optical_depth",
            ]
        ),
    }

    data = _request_json_with_retries(
        url=OPEN_METEO_AIR_QUALITY_URL,
        params=params,
        api_name="open-meteo air-quality",
    )

    _require_keys(data, ["hourly"], "air quality response")
    _require_keys(
        data["hourly"],
        ["time", "pm2_5", "pm10", "aerosol_optical_depth"],
        "air quality hourly",
    )

    return data


def _build_weather_df(hourly_data: dict[str, list[Any]]) -> pd.DataFrame:
    required = [
        "time",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "temperature_2m",
        "dew_point_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
    ]
    missing = [k for k in required if k not in hourly_data]
    if missing:
        raise ModelInputError(f"Weather hourly data missing keys: {missing}")

    df = pd.DataFrame(
        {
            "time": hourly_data["time"],
            "cloud_cover_total": hourly_data["cloud_cover"],
            "cloud_cover_low": hourly_data["cloud_cover_low"],
            "cloud_cover_mid": hourly_data["cloud_cover_mid"],
            "cloud_cover_high": hourly_data["cloud_cover_high"],
            "temperature": hourly_data["temperature_2m"],
            "dew_point": hourly_data["dew_point_2m"],
            "relative_humidity": hourly_data["relative_humidity_2m"],
            "pressure": hourly_data["surface_pressure"],
            "wind_speed": hourly_data["wind_speed_10m"],
            "wind_direction": hourly_data["wind_direction_10m"],
        }
    )
    df["time_dt"] = pd.to_datetime(df["time"])
    return df


def _build_air_quality_df(hourly_data: dict[str, list[Any]]) -> pd.DataFrame:
    required = ["time", "pm2_5", "pm10", "aerosol_optical_depth"]
    missing = [k for k in required if k not in hourly_data]
    if missing:
        raise ModelInputError(f"Air quality hourly data missing keys: {missing}")

    df = pd.DataFrame(
        {
            "time": hourly_data["time"],
            "pm25": hourly_data["pm2_5"],
            "pm10": hourly_data["pm10"],
            "aerosol_optical_depth": hourly_data["aerosol_optical_depth"],
        }
    )
    df["time_dt"] = pd.to_datetime(df["time"])
    return df


def _build_feature_row(
    sunset_dt: datetime,
    weather_df: pd.DataFrame,
    air_df: pd.DataFrame,
) -> dict[str, Any]:
    sunset_weather = _find_closest_row(weather_df, sunset_dt)
    sunset_air = _find_closest_row(air_df, sunset_dt)

    pressure_6h_before_row = _find_row_at_or_before(weather_df, sunset_dt - timedelta(hours=6))

    pressure_now = _safe_float(sunset_weather["pressure"])
    pressure_6h_before = (
        _safe_float(pressure_6h_before_row["pressure"])
        if pressure_6h_before_row is not None
        else None
    )

    row = {
        "cloud_cover_total": _safe_float(sunset_weather["cloud_cover_total"]),
        "cloud_cover_low": _safe_float(sunset_weather["cloud_cover_low"]),
        "cloud_cover_mid": _safe_float(sunset_weather["cloud_cover_mid"]),
        "cloud_cover_high": _safe_float(sunset_weather["cloud_cover_high"]),
        "aerosol_optical_depth": _safe_float(sunset_air["aerosol_optical_depth"]),
        "temperature": _safe_float(sunset_weather["temperature"]),
        "dew_point": _safe_float(sunset_weather["dew_point"]),
        "pressure_6h_before": pressure_6h_before,
        "pressure_trend": (
            float(pressure_now - pressure_6h_before)
            if pressure_now is not None and pressure_6h_before is not None
            else None
        ),
        "relative_humidity": _safe_float(sunset_weather["relative_humidity"]),
        "pm25": _safe_float(sunset_air["pm25"]),
        "pm10": _safe_float(sunset_air["pm10"]),
        "wind_speed": _safe_float(sunset_weather["wind_speed"]),
        "wind_direction": _safe_float(sunset_weather["wind_direction"]),
    }

    return row


def _make_ready_dataframe(
    feature_row: dict[str, Any],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    missing_fields: list[str] = []

    ready_row: dict[str, Any] = {}
    for col in feature_columns:
        value = feature_row.get(col, None)
        ready_row[col] = value
        if value is None:
            missing_fields.append(col)

    X_ready = pd.DataFrame([ready_row], columns=feature_columns)
    return X_ready, missing_fields


def build_today_model_input(
    city_name: str,
    latitude: float,
    longitude: float,
    feature_columns: Optional[list[str]] = None,
    timezone: str = "auto",
    target_date: Optional[str] = None,
    fail_on_missing: bool = True,
) -> dict[str, Any]:
    """
    Returns a ready-to-predict one-row DataFrame containing only the raw
    features needed by the selected model pipeline.

    Output:
    {
        "X_ready": pd.DataFrame,       # exact prediction columns, exact order
        "sunset_time_today": str,      # ISO datetime
        "snapshot": dict,              # raw feature row + metadata
        "missing_fields": list[str],   # missing required model columns
    }
    """
    if feature_columns is None:
        feature_columns = PREDICT_FEATURE_COLUMNS

    if target_date is None:
        target_date = date.today().isoformat()

    weather_data = _fetch_weather_forecast(
        latitude=latitude,
        longitude=longitude,
        target_date=target_date,
        timezone=timezone,
    )
    air_data = _fetch_air_quality_forecast(
        latitude=latitude,
        longitude=longitude,
        target_date=target_date,
        timezone=timezone,
    )

    sunset_values = weather_data["daily"].get("sunset", [])
    if not sunset_values:
        raise DataUnavailableError("No sunset time returned for requested date.")

    sunset_dt = _parse_iso_datetime(sunset_values[0])

    weather_df = _build_weather_df(weather_data["hourly"])
    air_df = _build_air_quality_df(air_data["hourly"])

    feature_row = _build_feature_row(
        sunset_dt=sunset_dt,
        weather_df=weather_df,
        air_df=air_df,
    )

    X_ready, missing_fields = _make_ready_dataframe(
        feature_row=feature_row,
        feature_columns=feature_columns,
    )

    snapshot = {
        "city_name": city_name,
        "latitude": latitude,
        "longitude": longitude,
        "prediction_date": target_date,
        "sunset_time_today": sunset_dt.isoformat(),
        **feature_row,
    }

    if fail_on_missing and missing_fields:
        raise ModelInputError(
            "Missing required model features for today's forecast: "
            + ", ".join(missing_fields)
        )

    result = ForecastBuildResult(
        X_ready=X_ready,
        sunset_time_today=sunset_dt.isoformat(),
        snapshot=snapshot,
        missing_fields=missing_fields,
    )
    return result.to_dict()


if __name__ == "__main__":
    result = build_today_model_input(
        city_name="Tel Aviv",
        latitude=32.0853,
        longitude=34.7818,
        timezone="auto",
        fail_on_missing=False,
    )

    print("Sunset time today:", result["sunset_time_today"])
    print("\nMissing fields:", result["missing_fields"])
    print("\nReady input:")
    print(result["X_ready"].to_string(index=False))
