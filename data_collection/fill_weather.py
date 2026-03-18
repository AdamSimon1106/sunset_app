import math
import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import timedelta, timezone
from dateutil import parser as dateparser
from typing import Optional, Dict, Any, Tuple

from astral import Observer
from astral.sun import azimuth, elevation, sunset

# ============================================================
# CONFIG
# ============================================================
EXCEL_PATH = "sunset_project_v2.xlsx"
SHEET_NAME = 0

SAVE_EVERY_N_ROWS = 50
SLEEP_BETWEEN_ROWS = 0.05
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
BASE_SLEEP = 1.2

# Sample point toward the sunset
SUNSET_DIR_DISTANCE_KM = 25.0

# Prefer European CAMS domain where available
AQ_PRIMARY_DOMAIN = "cams_europe"
AQ_FALLBACK_DOMAIN = "auto"

# If True, skip rows where all fillable cols are already non-empty
SKIP_ALREADY_FILLED = True

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

ARCHIVE_HOURLY_VARS = [
    "weather_code",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "relative_humidity_2m",
    "dew_point_2m",
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
]

AIR_HOURLY_VARS = [
    "pm10",
    "pm2_5",
    "aerosol_optical_depth",
    "dust",
]

SESSION = requests.Session()


# ============================================================
# HELPERS
# ============================================================
def _get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                sleep_s = BASE_SLEEP * attempt
                print(f"[retry {attempt}/{MAX_RETRIES}] {url} failed: {e} | sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)
    raise RuntimeError(f"GET failed after {MAX_RETRIES} attempts: {last_err}")


def _ensure_utc_dt(x) -> pd.Timestamp:
    dt = dateparser.isoparse(str(x))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt.astimezone(timezone.utc))


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None


def _nearest_hour_index(times, target_utc: pd.Timestamp) -> int:
    best_i = 0
    best_abs = None
    target_py = target_utc.to_pydatetime()

    for i, t in enumerate(times):
        dt = dateparser.isoparse(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        diff = abs((dt - target_py).total_seconds())
        if best_abs is None or diff < best_abs:
            best_abs = diff
            best_i = i
    return best_i


def _sum_last_n(hours_arr, idx: int, n: int) -> Optional[float]:
    start = idx - (n - 1)
    if start < 0:
        return None

    vals = []
    for v in hours_arr[start: idx + 1]:
        fv = _safe_float(v)
        if fv is None:
            return None
        vals.append(fv)

    return float(sum(vals))


def absolute_humidity_gm3(temp_c: Optional[float], rh_pct: Optional[float]) -> Optional[float]:
    if temp_c is None or rh_pct is None:
        return None

    es_hpa = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
    e_hpa = es_hpa * (rh_pct / 100.0)
    ah = 216.7 * (e_hpa / (temp_c + 273.15))
    return float(ah)


def cloud_type_from_layers(
    weather_code: Optional[float],
    low: Optional[float],
    mid: Optional[float],
    high: Optional[float],
    total: Optional[float],
) -> Optional[str]:
    if weather_code in {45, 48}:
        return "fog"

    low = low or 0.0
    mid = mid or 0.0
    high = high or 0.0

    if total is None:
        return None
    if total < 10:
        return "clear"

    values = {"low": low, "mid": mid, "high": high}
    dominant = max(values, key=values.get)
    dom_val = values[dominant]

    if total >= 85 and all(v >= 50 for v in values.values()):
        return "overcast_multi_layer"
    if dom_val < 15:
        return "mixed"
    return dominant


def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float) -> Tuple[float, float]:
    R = 6371.0
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    d = distance_km / R

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) +
        math.cos(lat1) * math.sin(d) * math.cos(brng)
    )

    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )

    lon2 = (lon2 + 3 * math.pi) % (2 * math.pi) - math.pi
    return math.degrees(lat2), math.degrees(lon2)


def compute_smoke_proxy(
    pm25: Optional[float],
    pm10: Optional[float],
    dust: Optional[float],
    aod: Optional[float],
) -> Optional[float]:
    """
    Transparent proxy, not a direct smoke measurement.

    Intuition:
    - PM2.5 captures fine particles more associated with haze/smoke
    - subtract some dust contribution so desert dust doesn't dominate
    - scale modestly by AOD so thick aerosol columns count more
    """
    pm25 = _safe_float(pm25)
    pm10 = _safe_float(pm10)
    dust = _safe_float(dust) or 0.0
    aod = _safe_float(aod) or 0.0

    if pm25 is None and pm10 is None and aod == 0.0:
        return None

    fine = pm25 if pm25 is not None else 0.6 * (pm10 or 0.0)
    coarse_gap = max((pm10 or fine) - fine, 0.0)

    # Prefer fine aerosol, discount dust-driven coarse load
    smoke = fine - 0.20 * coarse_gap - 0.35 * dust

    # Light AOD scaling; keeps value interpretable and stable
    smoke *= (1.0 + 0.60 * aod)

    return float(max(smoke, 0.0))


def save_excel(df: pd.DataFrame, path: str, sheet_name=0) -> None:
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        target_sheet = writer.book.sheetnames[sheet_name] if isinstance(sheet_name, int) else sheet_name
        df.to_excel(writer, index=False, sheet_name=target_sheet)


# ============================================================
# FETCHERS
# ============================================================
def fetch_archive_weather(lat: float, lon: float, dt_utc: pd.Timestamp) -> Dict[str, Optional[float]]:
    # Pull previous day too so rain_last_12h / pressure_6h_before are safe near midnight
    start_date = (dt_utc.date() - timedelta(days=1)).isoformat()
    end_date = dt_utc.date().isoformat()

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(ARCHIVE_HOURLY_VARS),
        "timezone": "GMT",
        "timeformat": "iso8601",
        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    data = _get_json(OPEN_METEO_ARCHIVE_URL, params)
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise RuntimeError("No hourly times returned from archive API")

    idx = _nearest_hour_index(times, dt_utc)

    def get_var(name: str) -> Optional[float]:
        arr = hourly.get(name)
        if arr is None or idx >= len(arr):
            return None
        return _safe_float(arr[idx])

    out = {
        "weather_code": get_var("weather_code"),
        "cloud_cover_total": get_var("cloud_cover"),
        "cloud_cover_low": get_var("cloud_cover_low"),
        "cloud_cover_mid": get_var("cloud_cover_mid"),
        "cloud_cover_high": get_var("cloud_cover_high"),
        "relative_humidity": get_var("relative_humidity_2m"),
        "dew_point": get_var("dew_point_2m"),
        "temperature": get_var("temperature_2m"),
        "wind_speed": get_var("wind_speed_10m"),
        "wind_direction": get_var("wind_direction_10m"),
        "pressure": get_var("surface_pressure"),
    }

    precip_arr = hourly.get("precipitation") or []
    pressure_arr = hourly.get("surface_pressure") or []

    out["rain_last_6h"] = _sum_last_n(precip_arr, idx, 6)
    out["rain_last_12h"] = _sum_last_n(precip_arr, idx, 12)

    p_now = _safe_float(pressure_arr[idx]) if idx < len(pressure_arr) else None
    p_prev6 = _safe_float(pressure_arr[idx - 6]) if idx - 6 >= 0 and idx - 6 < len(pressure_arr) else None

    out["pressure_6h_before"] = p_prev6
    out["pressure_trend"] = None if (p_now is None or p_prev6 is None) else float(p_now - p_prev6)

    out["absolute_humidity"] = absolute_humidity_gm3(out["temperature"], out["relative_humidity"])
    out["cloud_type"] = cloud_type_from_layers(
        out["weather_code"],
        out["cloud_cover_low"],
        out["cloud_cover_mid"],
        out["cloud_cover_high"],
        out["cloud_cover_total"],
    )

    return out


def fetch_air_quality(lat: float, lon: float, dt_utc: pd.Timestamp, domain: str) -> Dict[str, Optional[float]]:
    start_date = dt_utc.date().isoformat()
    end_date = dt_utc.date().isoformat()

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(AIR_HOURLY_VARS),
        "timezone": "GMT",
        "timeformat": "iso8601",
        "domains": domain,
    }

    data = _get_json(OPEN_METEO_AIR_URL, params)
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return {
            "pm25": None,
            "pm10": None,
            "aerosol_optical_depth": None,
            "dust": None,
            "smoke": None,
        }

    idx = _nearest_hour_index(times, dt_utc)

    def get_var(name: str) -> Optional[float]:
        arr = hourly.get(name)
        if arr is None or idx >= len(arr):
            return None
        return _safe_float(arr[idx])

    pm25 = get_var("pm2_5")
    pm10 = get_var("pm10")
    aod = get_var("aerosol_optical_depth")
    dust = get_var("dust")

    return {
        "pm25": pm25,
        "pm10": pm10,
        "aerosol_optical_depth": aod,
        "dust": dust,
        "smoke": compute_smoke_proxy(pm25, pm10, dust, aod),
    }


def fetch_air_quality_with_fallback(lat: float, lon: float, dt_utc: pd.Timestamp) -> Dict[str, Optional[float]]:
    try:
        result = fetch_air_quality(lat, lon, dt_utc, AQ_PRIMARY_DOMAIN)

        # If CAMS Europe gives almost nothing, fall back
        populated = sum(v is not None for v in [
            result.get("pm25"),
            result.get("pm10"),
            result.get("aerosol_optical_depth"),
            result.get("dust"),
        ])
        if populated >= 2:
            return result
    except Exception as e:
        print(f"[aq primary fallback] {e}")

    return fetch_air_quality(lat, lon, dt_utc, AQ_FALLBACK_DOMAIN)


def compute_solar_fields(lat: float, lon: float, elevation_m: float, dt_utc: pd.Timestamp) -> Dict[str, Optional[float]]:
    try:
        obs = Observer(latitude=lat, longitude=lon, elevation=elevation_m or 0.0)
        solar_el = float(elevation(obs, dt_utc.to_pydatetime()))
        solar_az = float(azimuth(obs, dt_utc.to_pydatetime()))
        sunset_dt = sunset(obs, date=dt_utc.date(), tzinfo=timezone.utc)
        sunset_dir = float(azimuth(obs, sunset_dt))

        return {
            "solar_elevation": solar_el,
            "solar_azimuth": solar_az,
            "sunset_direction": sunset_dir,
        }
    except Exception:
        return {
            "solar_elevation": None,
            "solar_azimuth": None,
            "sunset_direction": None,
        }


def fetch_all_for_point(lat: float, lon: float, elevation_m: float, dt_utc: pd.Timestamp) -> Dict[str, Any]:
    weather = fetch_archive_weather(lat, lon, dt_utc)
    air = fetch_air_quality_with_fallback(lat, lon, dt_utc)
    solar = compute_solar_fields(lat, lon, elevation_m, dt_utc)

    out = {}
    out.update(weather)
    out.update(air)
    out.update(solar)

    # Open-Meteo historical weather docs do not document visibility here
    out["visibility"] = None

    return out


def fetch_row_values(lat: float, lon: float, elevation_m: float, dt_utc: pd.Timestamp) -> Dict[str, Any]:
    base = fetch_all_for_point(lat, lon, elevation_m, dt_utc)

    sunset_dir = base.get("sunset_direction")
    if sunset_dir is None:
        base.update({
            "cloud_cover_total_sunset_dir": None,
            "cloud_cover_low_sunset_dir": None,
            "cloud_cover_mid_sunset_dir": None,
            "cloud_cover_high_sunset_dir": None,
            "cloud_type_sunset_dir": None,
            "aerosol_optical_depth_sunset_dir": None,
            "pm25_sunset_dir": None,
            "pm10_sunset_dir": None,
            "dust_sunset_dir": None,
            "smoke_sunset_dir": None,
            "visibility_sunset_dir": None,
            "rain_last_6h_sunset_dir": None,
            "rain_last_12h_sunset_dir": None,
        })
        return base

    dir_lat, dir_lon = destination_point(lat, lon, sunset_dir, SUNSET_DIR_DISTANCE_KM)
    sunset_point = fetch_all_for_point(dir_lat, dir_lon, elevation_m, dt_utc)

    base.update({
        "cloud_cover_total_sunset_dir": sunset_point.get("cloud_cover_total"),
        "cloud_cover_low_sunset_dir": sunset_point.get("cloud_cover_low"),
        "cloud_cover_mid_sunset_dir": sunset_point.get("cloud_cover_mid"),
        "cloud_cover_high_sunset_dir": sunset_point.get("cloud_cover_high"),
        "cloud_type_sunset_dir": sunset_point.get("cloud_type"),
        "aerosol_optical_depth_sunset_dir": sunset_point.get("aerosol_optical_depth"),
        "pm25_sunset_dir": sunset_point.get("pm25"),
        "pm10_sunset_dir": sunset_point.get("pm10"),
        "dust_sunset_dir": sunset_point.get("dust"),
        "smoke_sunset_dir": sunset_point.get("smoke"),
        "visibility_sunset_dir": sunset_point.get("visibility"),
        "rain_last_6h_sunset_dir": sunset_point.get("rain_last_6h"),
        "rain_last_12h_sunset_dir": sunset_point.get("rain_last_12h"),
    })

    return base


# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    required = ["datetime", "lat", "lon"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column '{c}'")

    fillable_cols = [
        "sunset_direction",
        "cloud_cover_total",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "cloud_type",
        "relative_humidity",
        "absolute_humidity",
        "dew_point",
        "temperature",
        "pm25",
        "pm10",
        "dust",
        "smoke",
        "aerosol_optical_depth",
        "visibility",
        "wind_speed",
        "wind_direction",
        "rain_last_6h",
        "rain_last_12h",
        "pressure",
        "pressure_6h_before",
        "pressure_trend",
        "solar_elevation",
        "solar_azimuth",
        "cloud_cover_total_sunset_dir",
        "cloud_cover_low_sunset_dir",
        "cloud_cover_mid_sunset_dir",
        "cloud_cover_high_sunset_dir",
        "cloud_type_sunset_dir",
        "aerosol_optical_depth_sunset_dir",
        "pm25_sunset_dir",
        "pm10_sunset_dir",
        "dust_sunset_dir",
        "smoke_sunset_dir",
        "visibility_sunset_dir",
        "rain_last_6h_sunset_dir",
        "rain_last_12h_sunset_dir",
    ]

    def set_if_exists(row_idx: int, col: str, val):
        if col in df.columns:
            df.at[row_idx, col] = val

    def should_skip_row(row_idx: int) -> bool:
        if not SKIP_ALREADY_FILLED:
            return False

        present = 0
        filled = 0
        for c in fillable_cols:
            if c in df.columns:
                present += 1
                if not pd.isna(df.at[row_idx, c]):
                    filled += 1
        return present > 0 and filled == present

    updated_since_save = 0

    for i in tqdm(range(len(df)), desc="Filling weather v4"):
        if should_skip_row(i):
            continue

        dt_raw = df.at[i, "datetime"]
        lat = df.at[i, "lat"]
        lon = df.at[i, "lon"]

        if pd.isna(dt_raw) or pd.isna(lat) or pd.isna(lon):
            continue

        elevation_m = 0.0
        if "elevation_m" in df.columns and not pd.isna(df.at[i, "elevation_m"]):
            elevation_m = float(df.at[i, "elevation_m"])

        dt_utc = _ensure_utc_dt(dt_raw)
        lat_f = float(lat)
        lon_f = float(lon)

        try:
            vals = fetch_row_values(lat_f, lon_f, elevation_m, dt_utc)
        except Exception as e:
            print(f"[row {i}] failed: {e}")
            continue

        for col in fillable_cols:
            set_if_exists(i, col, vals.get(col))

        updated_since_save += 1

        if updated_since_save >= SAVE_EVERY_N_ROWS:
            save_excel(df, EXCEL_PATH, SHEET_NAME)
            print(f"[checkpoint] saved after row {i}")
            updated_since_save = 0

        time.sleep(SLEEP_BETWEEN_ROWS)

    save_excel(df, EXCEL_PATH, SHEET_NAME)
    print(f"Done. Updated file: {EXCEL_PATH}")


if __name__ == "__main__":
    main()