import math
import os
import time
import zipfile
from pathlib import Path
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import requests
import xarray as xr
from dateutil import parser as dateparser
from tqdm import tqdm

from astral import Observer
from astral.sun import azimuth, elevation, sunset
import cdsapi

# ============================================================
# CONFIG
# ============================================================
EXCEL_PATH = "sunset_project_v2.xlsx"
SHEET_NAME = 0

CACHE_DIR = Path("cams_cache")
CACHE_DIR.mkdir(exist_ok=True)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
BASE_SLEEP = 1.2

SAVE_EVERY_N_ROWS = 50
SLEEP_BETWEEN_ROWS = 0.03

# distance to sample in the direction of sunset
SUNSET_DIR_DISTANCE_KM = 25.0

# Set this to False if you want to overwrite existing filled values
SKIP_ALREADY_FILLED = True

# ------------------------------------------------------------
# Open-Meteo weather variables
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# CAMS Europe reanalysis variable names
# NOTE:
# - Variable names in ADS can evolve.
# - These are the intended names for the reanalysis request.
# - If one name fails in your ADS account, copy the exact name
#   from the dataset page's "Show API request code".
# ------------------------------------------------------------
CAMS_EU_DATASET = "cams-europe-air-quality-reanalyses"
CAMS_EU_MODEL = "ensemble"

CAMS_EU_VARIABLES = [
    "particulate_matter_2.5um",   # -> pm25
    "particulate_matter_10um",    # -> pm10
    "pm10_wildfires",             # -> smoke (best direct-ish proxy if available)
]

# ------------------------------------------------------------
# CAMS global EAC4 variable names
# These are documented optical-depth variables on EAC4.
# ------------------------------------------------------------
CAMS_GLOBAL_DATASET = "cams-global-reanalysis-eac4"

CAMS_GLOBAL_VARIABLES = [
    "total_aerosol_optical_depth_550nm",           # -> aerosol_optical_depth
    "dust_aerosol_optical_depth_550nm",            # -> dust proxy
    "black_carbon_aerosol_optical_depth_550nm",    # -> smoke component
    "organic_matter_aerosol_optical_depth_550nm",  # -> smoke component
]

SESSION = requests.Session()


# ============================================================
# GENERAL HELPERS
# ============================================================
def _get_json(url: str, params: dict) -> dict:
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


def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None


def _sum_last_n(hours_arr, idx: int, n: int):
    start = idx - (n - 1)
    if start < 0:
        return None
    window = hours_arr[start: idx + 1]
    vals = []
    for v in window:
        fv = _safe_float(v)
        if fv is None:
            return None
        vals.append(fv)
    return float(sum(vals))


def absolute_humidity_gm3(temp_c, rh_pct):
    """
    Absolute humidity in g/m^3
    """
    temp_c = _safe_float(temp_c)
    rh_pct = _safe_float(rh_pct)
    if temp_c is None or rh_pct is None:
        return None

    es_hpa = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
    e_hpa = es_hpa * (rh_pct / 100.0)
    ah = 216.7 * (e_hpa / (temp_c + 273.15))
    return float(ah)


def cloud_type_from_layers(weather_code, low, mid, high, total):
    weather_code = _safe_float(weather_code)
    low = _safe_float(low) or 0.0
    mid = _safe_float(mid) or 0.0
    high = _safe_float(high) or 0.0
    total = _safe_float(total)

    if weather_code in {45, 48}:
        return "fog"

    if total is None:
        return None

    if total < 10:
        return "clear"

    layer_vals = {"low": low, "mid": mid, "high": high}
    dominant = max(layer_vals, key=layer_vals.get)
    dom_val = layer_vals[dominant]

    if total >= 85 and all(v >= 50 for v in layer_vals.values()):
        return "overcast_multi_layer"
    if dom_val < 15:
        return "mixed"

    return dominant


def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float):
    """
    Great-circle destination point
    """
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


def save_excel(df: pd.DataFrame, path: str, sheet_name=0):
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        target_sheet = writer.book.sheetnames[sheet_name] if isinstance(sheet_name, int) else sheet_name
        df.to_excel(writer, index=False, sheet_name=target_sheet)


# ============================================================
# OPEN-METEO WEATHER
# ============================================================
def _nearest_hour_index(times, target_utc: pd.Timestamp) -> int:
    target_py = target_utc.to_pydatetime()
    diffs = []
    for t in times:
        dt = dateparser.isoparse(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        diffs.append(abs((dt - target_py).total_seconds()))
    return int(np.argmin(diffs))


def fetch_open_meteo_weather(lat: float, lon: float, dt_utc: pd.Timestamp) -> dict:
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
        raise RuntimeError("No hourly data returned from Open-Meteo")

    idx = _nearest_hour_index(times, dt_utc)

    def get_var(name):
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
    out["rain_last_6h"] = _sum_last_n(precip_arr, idx, 6)
    out["rain_last_12h"] = _sum_last_n(precip_arr, idx, 12)

    pressure_arr = hourly.get("surface_pressure") or []
    p_now = _safe_float(pressure_arr[idx]) if idx < len(pressure_arr) else None
    p_prev6 = _safe_float(pressure_arr[idx - 6]) if idx - 6 >= 0 and (idx - 6) < len(pressure_arr) else None

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


# ============================================================
# SOLAR GEOMETRY
# ============================================================
def compute_solar_fields(lat: float, lon: float, elevation_m: float, dt_utc: pd.Timestamp) -> dict:
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


# ============================================================
# CAMS DOWNLOAD / CACHE
# ============================================================
def _extract_single_netcdf_from_zip(zip_path: Path, output_nc: Path) -> Path:
    if output_nc.exists():
        return output_nc

    with zipfile.ZipFile(zip_path, "r") as zf:
        nc_files = [n for n in zf.namelist() if n.lower().endswith(".nc")]
        if not nc_files:
            raise RuntimeError(f"No .nc file found in zip: {zip_path}")
        member = nc_files[0]
        zf.extract(member, output_nc.parent)
        extracted = output_nc.parent / member
        extracted.replace(output_nc)

    return output_nc


def _month_date_range(year: int, month: int) -> str:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(1)
    return f"{start.date().isoformat()}/{end.date().isoformat()}"


def _ads_client() -> cdsapi.Client:
    # Uses ~/.cdsapirc
    return cdsapi.Client()


def ensure_cams_global_month(year: int, month: int) -> Path:
    nc_path = CACHE_DIR / f"cams_global_eac4_{year}_{month:02d}.nc"
    zip_path = CACHE_DIR / f"cams_global_eac4_{year}_{month:02d}.zip"

    if nc_path.exists():
        return nc_path

    client = _ads_client()
    request = {
        "variable": CAMS_GLOBAL_VARIABLES,
        "date": [_month_date_range(year, month)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf_zip",
    }

    print(f"[download] CAMS global EAC4 {year}-{month:02d}")
    client.retrieve(CAMS_GLOBAL_DATASET, request, str(zip_path))
    _extract_single_netcdf_from_zip(zip_path, nc_path)
    return nc_path


def ensure_cams_europe_month(year: int, month: int) -> Path:
    nc_path = CACHE_DIR / f"cams_europe_reanalysis_{year}_{month:02d}.nc"
    zip_path = CACHE_DIR / f"cams_europe_reanalysis_{year}_{month:02d}.zip"

    if nc_path.exists():
        return nc_path

    client = _ads_client()

    # This request structure may need one small adjustment on your ADS page
    # depending on the exact current field names shown in "Show API request code".
    request = {
        "date": [_month_date_range(year, month)],
        "type": ["validated_reanalysis", "interim_reanalysis"],
        "format": "netcdf_zip",
        "variable": CAMS_EU_VARIABLES,
        "model": [CAMS_EU_MODEL],
        "time": [f"{h:02d}:00" for h in range(24)],
        "level": ["0"],
    }

    print(f"[download] CAMS Europe reanalysis {year}-{month:02d}")
    client.retrieve(CAMS_EU_DATASET, request, str(zip_path))
    _extract_single_netcdf_from_zip(zip_path, nc_path)
    return nc_path


# ============================================================
# XARRAY SAMPLING
# ============================================================
def _open_dataset_cached(nc_path: Path) -> xr.Dataset:
    return xr.open_dataset(nc_path)


def _find_time_dim(ds: xr.Dataset) -> str:
    for c in ["time", "valid_time", "forecast_reference_time"]:
        if c in ds.coords or c in ds.dims:
            return c
    raise RuntimeError(f"Could not find time coordinate in dataset: {list(ds.coords)}")


def _find_lat_lon_names(ds: xr.Dataset):
    lat_name = None
    lon_name = None
    for c in ["latitude", "lat"]:
        if c in ds.coords or c in ds.dims:
            lat_name = c
            break
    for c in ["longitude", "lon"]:
        if c in ds.coords or c in ds.dims:
            lon_name = c
            break
    if lat_name is None or lon_name is None:
        raise RuntimeError(f"Could not find lat/lon coordinates in dataset: {list(ds.coords)}")
    return lat_name, lon_name


def _normalize_lon_for_dataset(lon: float, ds: xr.Dataset, lon_name: str) -> float:
    lon_vals = ds[lon_name].values
    if np.nanmin(lon_vals) >= 0 and lon < 0:
        return lon % 360
    if np.nanmin(lon_vals) < 0 and lon > 180:
        return ((lon + 180) % 360) - 180
    return lon


def _select_nearest(ds: xr.Dataset, lat: float, lon: float, dt_utc: pd.Timestamp) -> xr.Dataset:
    time_name = _find_time_dim(ds)
    lat_name, lon_name = _find_lat_lon_names(ds)
    lon2 = _normalize_lon_for_dataset(lon, ds, lon_name)

    sub = ds.sel(
        {
            lat_name: lat,
            lon_name: lon2,
            time_name: np.datetime64(dt_utc.to_datetime64()),
        },
        method="nearest",
    )
    return sub


def _read_first_existing(ds_or_da, candidate_names):
    for name in candidate_names:
        if name in ds_or_da:
            try:
                val = ds_or_da[name].values
                if np.ndim(val) > 0:
                    val = np.ravel(val)[0]
                return _safe_float(val)
            except Exception:
                pass
    return None


def sample_cams_global(ds: xr.Dataset, lat: float, lon: float, dt_utc: pd.Timestamp) -> dict:
    sub = _select_nearest(ds, lat, lon, dt_utc)

    aod = _read_first_existing(sub, [
        "total_aerosol_optical_depth_550nm",
        "aod550",
    ])
    dust = _read_first_existing(sub, [
        "dust_aerosol_optical_depth_550nm",
        "duaod550",
    ])
    bc = _read_first_existing(sub, [
        "black_carbon_aerosol_optical_depth_550nm",
        "bcaod550",
    ])
    om = _read_first_existing(sub, [
        "organic_matter_aerosol_optical_depth_550nm",
        "omaod550",
    ])

    smoke_proxy = None
    if bc is not None or om is not None:
        smoke_proxy = (1.3 * (bc or 0.0)) + (0.7 * (om or 0.0))

    return {
        "aerosol_optical_depth": aod,
        "dust": dust,
        "smoke_proxy_global": smoke_proxy,
    }


def sample_cams_europe(ds: xr.Dataset, lat: float, lon: float, dt_utc: pd.Timestamp) -> dict:
    sub = _select_nearest(ds, lat, lon, dt_utc)

    pm25 = _read_first_existing(sub, [
        "particulate_matter_2.5um",
        "pm2p5",
        "pm25",
    ])
    pm10 = _read_first_existing(sub, [
        "particulate_matter_10um",
        "pm10",
    ])
    pm10_wf = _read_first_existing(sub, [
        "pm10_wildfires",
        "pm10wf",
    ])

    return {
        "pm25": pm25,
        "pm10": pm10,
        "smoke_direct": pm10_wf,
    }


# ============================================================
# COMPOSITE FETCH
# ============================================================
def fetch_all_sources_for_point(
    lat: float,
    lon: float,
    elevation_m: float,
    dt_utc: pd.Timestamp,
    ds_global: xr.Dataset,
    ds_europe: xr.Dataset,
) -> dict:
    weather = fetch_open_meteo_weather(lat, lon, dt_utc)
    solar = compute_solar_fields(lat, lon, elevation_m, dt_utc)
    cams_global = sample_cams_global(ds_global, lat, lon, dt_utc)
    cams_europe = sample_cams_europe(ds_europe, lat, lon, dt_utc)

    smoke = cams_europe["smoke_direct"]
    if smoke is None:
        smoke = cams_global["smoke_proxy_global"]

    out = {}
    out.update(weather)
    out.update(solar)
    out["pm25"] = cams_europe["pm25"]
    out["pm10"] = cams_europe["pm10"]
    out["aerosol_optical_depth"] = cams_global["aerosol_optical_depth"]
    out["dust"] = cams_global["dust"]
    out["smoke"] = smoke

    return out


def fetch_row_values(
    lat: float,
    lon: float,
    elevation_m: float,
    dt_utc: pd.Timestamp,
    ds_global: xr.Dataset,
    ds_europe: xr.Dataset,
) -> dict:
    base = fetch_all_sources_for_point(lat, lon, elevation_m, dt_utc, ds_global, ds_europe)

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

    sunset_point = fetch_all_sources_for_point(dir_lat, dir_lon, elevation_m, dt_utc, ds_global, ds_europe)

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
        # Open-Meteo historical archive page does not expose visibility in the
        # documented historical endpoint you are already using, so keep None here.
        "visibility_sunset_dir": None,
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
            raise RuntimeError(f"Missing required column: {c}")

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
        existing = []
        for c in fillable_cols:
            if c in df.columns:
                v = df.at[row_idx, c]
                existing.append(not pd.isna(v))
        return len(existing) > 0 and all(existing)

    # Precompute months needed
    valid_rows = []
    month_keys = set()

    for i in range(len(df)):
        dt_raw = df.at[i, "datetime"]
        lat = df.at[i, "lat"]
        lon = df.at[i, "lon"]
        if pd.isna(dt_raw) or pd.isna(lat) or pd.isna(lon):
            continue
        dt_utc = _ensure_utc_dt(dt_raw)
        valid_rows.append((i, dt_utc))
        month_keys.add((dt_utc.year, dt_utc.month))

    # Download/cache monthly files once
    global_datasets = {}
    europe_datasets = {}

    for year, month in sorted(month_keys):
        global_nc = ensure_cams_global_month(year, month)
        europe_nc = ensure_cams_europe_month(year, month)
        global_datasets[(year, month)] = _open_dataset_cached(global_nc)
        europe_datasets[(year, month)] = _open_dataset_cached(europe_nc)

    updated_since_save = 0

    for i, dt_utc in tqdm(valid_rows, desc="Filling v3 weather"):
        if should_skip_row(i):
            continue

        lat = float(df.at[i, "lat"])
        lon = float(df.at[i, "lon"])
        elevation_m = 0.0
        if "elevation_m" in df.columns and not pd.isna(df.at[i, "elevation_m"]):
            elevation_m = float(df.at[i, "elevation_m"])

        ds_global = global_datasets[(dt_utc.year, dt_utc.month)]
        ds_europe = europe_datasets[(dt_utc.year, dt_utc.month)]

        try:
            vals = fetch_row_values(lat, lon, elevation_m, dt_utc, ds_global, ds_europe)
        except Exception as e:
            print(f"[row {i}] failed: {e}")
            continue

        # visibility is not filled by this v3 unless you add a separate source
        vals.setdefault("visibility", None)

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