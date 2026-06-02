from datetime import datetime, timezone, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from math import sqrt, atan2, degrees
from dataclasses import dataclass, field
import asyncio
import aiohttp
import pytz
from astral import Observer, LocationInfo
from astral.sun import sunset, azimuth, sun
from geopy.distance import distance
from geopy import Point

# ======================== CONSTANTS ========================

EARTH_RADIUS             = 6371 * 1000
ELEVATION_ENDPOINT       = "https://api.open-elevation.com/api/v1/lookup"
OPEN_METEO_FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MAX_RETRIES              = 5
EYE_LEVEL_M              = 50
TERRAIN_SAMPLES          = 20
WEATHER_POINTS           = 4

ELEVATION_CONCURRENCY    = 10
WEATHER_CONCURRENCY      = 20
TLV_LAT, TLV_LON = 32.0885752, 34.7704678
requests_list = [
        {"lat": 32.0885752, "lon": 34.7704678, "tz": "Asia/Jerusalem"},
    ]


# ======================== FIELD MAPPINGS ========================

@dataclass
class MeteoFieldMap:
    """Canonical -> actual API field name mapping for one endpoint."""
    endpoint: str

    # Cloud cover
    cloudcover_low:   str = "cloud_cover_low"
    cloudcover_mid:   str = "cloud_cover_mid"
    cloudcover_high:  str = "cloud_cover_high"
    cloudcover_total: str = "cloud_cover"

    # Temperature / humidity
    temperature_2m:       str = "temperature_2m"
    apparent_temperature: str = "apparent_temperature"
    dewpoint_2m:          str = "dew_point_2m"
    relative_humidity_2m: str = "relative_humidity_2m"

    # Wind
    windspeed_10m:     str = "wind_speed_10m"
    winddirection_10m: str = "wind_direction_10m"
    windgusts_10m:     str = "wind_gusts_10m"

    # Precipitation & pressure
    precipitation:             str = "precipitation"
    precipitation_probability: str = "precipitation_probability"
    surface_pressure:          str = "surface_pressure"

    # Visibility & radiation
    visibility:          str = "visibility"
    shortwave_radiation: str = "shortwave_radiation"

    def cloud_fields(self) -> str:
        return ",".join([
            self.cloudcover_low,
            self.cloudcover_mid,
            self.cloudcover_high,
            self.cloudcover_total,
        ])

    def weather_fields(self) -> str:
        return ",".join([
            self.temperature_2m,
            self.apparent_temperature,
            self.dewpoint_2m,
            self.relative_humidity_2m,
            self.windspeed_10m,
            self.winddirection_10m,
            self.windgusts_10m,
            self.precipitation,
            self.precipitation_probability,
            self.surface_pressure,
            self.visibility,
            self.shortwave_radiation,
            self.cloudcover_low,
            self.cloudcover_mid,
            self.cloudcover_high,
            self.cloudcover_total,
        ])


# Both forecast and archive now use the same snake_case field names.
# We keep two instances so endpoint URLs differ, but field names are identical.
FORECAST_MAP = MeteoFieldMap(endpoint=OPEN_METEO_FORECAST_URL)
ARCHIVE_MAP  = MeteoFieldMap(endpoint=OPEN_METEO_ARCHIVE_URL)

AIR_QUALITY_FIELDS = ",".join([
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
    "european_aqi",
    "us_aqi",
])


# ======================== HELPERS ========================

def is_past(dt_utc: datetime) -> bool:
    now = datetime.now(timezone.utc)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.date() < now.date()


def get_field_map(dt_utc: datetime) -> MeteoFieldMap:
    return ARCHIVE_MAP if is_past(dt_utc) else FORECAST_MAP


def _extract_hour(hourly: dict, key: str, target: str) -> Optional[float]:
    """Pull value for `target` hour string from an Open-Meteo hourly dict."""
    times = hourly.get("time", [])
    values = hourly.get(key, [])
    if not values or target not in times:
        return None
    idx = times.index(target)
    v = values[idx]
    return float(v) if v is not None else None


# ======================== CLOUD TYPE INFERENCE ========================

def infer_cloud_types(
    low_pct: Optional[float],
    mid_pct: Optional[float],
    high_pct: Optional[float],
    total_pct: Optional[float],
    precipitation: Optional[float],
    temp_2m: Optional[float],
) -> list[str]:
    types: list[str] = []
    low    = low_pct   or 0.0
    mid    = mid_pct   or 0.0
    high   = high_pct  or 0.0
    total  = total_pct or 0.0
    precip = precipitation or 0.0

    if high >= 60:
        types.append("Cirrostratus")
    elif high >= 30:
        types.append("Cirrus")
    if high >= 40 and mid < 20:
        types.append("Cirrocumulus")

    if mid >= 70:
        types.append("Altostratus")
    elif mid >= 25:
        types.append("Altocumulus")

    if low >= 80 and precip > 0.5:
        if temp_2m is not None and temp_2m > 10 and mid >= 30:
            types.append("Cumulonimbus")
        else:
            types.append("Nimbostratus")
    elif low >= 70:
        types.append("Stratus")
    elif low >= 30:
        types.append("Stratocumulus")
    elif low >= 10 and total < 50:
        types.append("Cumulus")

    if not types or total < 10:
        types = ["Clear sky"]

    return types


# ======================== GEOMETRY ========================

def find_destination_point(origin: Point, dist_m: float, azimuth_deg: float) -> Point:
    dest = distance(meters=dist_m).destination(origin, bearing=azimuth_deg)
    return Point(dest.latitude, dest.longitude)


def get_distance_to_horizon(elevation_m: float) -> float:
    effective_elevation = max(elevation_m, 0.0) + EYE_LEVEL_M
    return sqrt(2 * EARTH_RADIUS * effective_elevation)


def get_azimuth_to_sunset(user_location: Point, elevation_m: float, dt_utc: datetime) -> float:
    obs = Observer(latitude=user_location.latitude, longitude=user_location.longitude, elevation=elevation_m)
    sunset_dt = sunset(obs, date=dt_utc.date(), tzinfo=pytz.utc)
    return float(azimuth(obs, sunset_dt))


# ======================== ELEVATION API ========================

async def get_elevation_m(session: aiohttp.ClientSession, sem: asyncio.Semaphore, location: Point) -> Optional[float]:
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(
                    ELEVATION_ENDPOINT,
                    params={"locations": f"{location.latitude},{location.longitude}"}
                ) as r:
                    r.raise_for_status()
                    data = await r.json()
                    results = data.get("results") or []
                    if not results:
                        return None
                    elev = results[0].get("elevation")
                    return float(elev) if elev is not None else None
            except Exception as e:
                print(f"[elevation error] attempt={attempt} lat={location.latitude} lon={location.longitude} err={e}")
                await asyncio.sleep(0.5 * attempt)
    print(f"[elevation failed] lat={location.latitude} lon={location.longitude}")
    return 50


async def get_elevations_batch(session: aiohttp.ClientSession, sem: asyncio.Semaphore, points: list[Point]) -> list[Optional[float]]:
    locations_str = "|".join(f"{p.latitude},{p.longitude}" for p in points)
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(ELEVATION_ENDPOINT, params={"locations": locations_str}) as r:
                    r.raise_for_status()
                    data = await r.json()
                    results = data.get("results") or []
                    return [float(res["elevation"]) if res.get("elevation") is not None else None for res in results]
            except Exception as e:
                print(f"[batch elevation error] attempt={attempt} err={e}")
                await asyncio.sleep(0.5 * attempt)
    print(f"[batch elevation failed]")
    return [None] * len(points)


# ======================== VISIBLE HORIZON ========================

async def get_visible_horizon(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    user_location: Point,
    viewer_elevation_m: float,
    azimuth_to_sunset: float,
    geometric_horizon_m: float
) -> dict:
    step             = geometric_horizon_m / TERRAIN_SAMPLES
    sample_points    = [find_destination_point(user_location, step * i, azimuth_to_sunset) for i in range(1, TERRAIN_SAMPLES + 1)]
    sample_distances = [step * i for i in range(1, TERRAIN_SAMPLES + 1)]

    elevations = await get_elevations_batch(session, sem, sample_points)

    max_angle   = -float("inf")
    horizon_idx = 0
    for i, (d, elev) in enumerate(zip(sample_distances, elevations)):
        if elev is None:
            continue
        angle = degrees(atan2(elev - viewer_elevation_m, d))
        if angle > max_angle:
            max_angle   = angle
            horizon_idx = i

    return {
        "visible_horizon_dist_m": sample_distances[horizon_idx],
        "blocking_angle_deg":     max_angle,
    }


# ======================== CLOUD + WEATHER DATA ========================

async def _fetch_hourly(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    endpoint: str,
    params: dict,
    target: str,
    label: str = "weather",
) -> Optional[dict]:
    """Low-level helper: fetch one Open-Meteo hourly request and return the hourly dict."""
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(endpoint, params=params) as r:
                    r.raise_for_status()
                    data   = await r.json()
                    hourly = data.get("hourly", {})
                    if target not in hourly.get("time", []):
                        print(f"[{label}] hour {target} not in response.")
                        return None
                    return hourly
            except Exception as e:
                print(f"[{label} error] attempt={attempt} err={e}")
                await asyncio.sleep(0.5 * attempt)
    print(f"[{label} failed]")
    return None


async def fetch_weather_data(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    location: Point,
    dt_utc: datetime,
    include_full_weather: bool = False,
) -> Optional[dict]:
    fm     = get_field_map(dt_utc)
    fields = fm.weather_fields() if include_full_weather else fm.cloud_fields()
    target = dt_utc.strftime("%Y-%m-%dT%H:00")

    base_params = {
        "latitude":   location.latitude,
        "longitude":  location.longitude,
        "hourly":     fields,
        "start_date": dt_utc.strftime("%Y-%m-%d"),
        "end_date":   (dt_utc + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone":   "UTC",
    }

    # Primary: ECMWF for accurate cloud layers (forecast only, ~10-day horizon)
    ecmwf_hourly = await _fetch_hourly(
        session, sem, fm.endpoint,
        {**base_params, "models": "ecmwf_ifs025"},
        target, label="ecmwf",
    )

    # Fallback to ICON if ECMWF unavailable (archive or beyond 10-day horizon)
    icon_hourly: Optional[dict] = None
    if ecmwf_hourly is None:
        ecmwf_hourly = await _fetch_hourly(
            session, sem, fm.endpoint,
            base_params, target, label="icon_fallback",
        )
        if ecmwf_hourly is None:
            return None
    elif include_full_weather:
        # Fetch ICON separately only to get visibility, which ECMWF does not provide
        icon_hourly = await _fetch_hourly(
            session, sem, fm.endpoint,
            base_params, target, label="icon_visibility",
        )

    def get(hourly: dict, key: str) -> Optional[float]:
        return _extract_hour(hourly, key, target)

    low   = get(ecmwf_hourly, fm.cloudcover_low)
    mid   = get(ecmwf_hourly, fm.cloudcover_mid)
    high  = get(ecmwf_hourly, fm.cloudcover_high)
    total = get(ecmwf_hourly, fm.cloudcover_total)

    # ECMWF quirk: total can be non-zero even when all three layers are 0 due to
    # internal maximum-overlap cloud fraction computation. Use effective_total
    # as max(reported_total, max_layer) so cloud type inference stays consistent.
    layer_max       = max(low or 0.0, mid or 0.0, high or 0.0)
    effective_total = max(total or 0.0, layer_max)

    precip = get(ecmwf_hourly, fm.precipitation) if include_full_weather else None
    temp   = get(ecmwf_hourly, fm.temperature_2m) if include_full_weather else None

    result: dict = {
        "clouds": {
            "low_pct":   low,
            "mid_pct":   mid,
            "high_pct":  high,
            "total_pct": total,
            "types":     infer_cloud_types(low, mid, high, effective_total, precip, temp),
        }
    }

    if include_full_weather:
        # Visibility from ICON (ECMWF does not expose it); everything else from ECMWF
        visibility = get(icon_hourly, fm.visibility) if icon_hourly else None
        result["weather"] = {
            "temp_c":                  temp,
            "apparent_temp_c":         get(ecmwf_hourly, fm.apparent_temperature),
            "dewpoint_c":              get(ecmwf_hourly, fm.dewpoint_2m),
            "humidity_pct":            get(ecmwf_hourly, fm.relative_humidity_2m),
            "wind_speed_kmh":          get(ecmwf_hourly, fm.windspeed_10m),
            "wind_dir_deg":            get(ecmwf_hourly, fm.winddirection_10m),
            "wind_gusts_kmh":          get(ecmwf_hourly, fm.windgusts_10m),
            "precip_mm":               precip,
            "precip_prob_pct":         get(ecmwf_hourly, fm.precipitation_probability),
            "pressure_hpa":            get(ecmwf_hourly, fm.surface_pressure),
            "visibility_m":            visibility,
            "shortwave_radiation_wm2": get(ecmwf_hourly, fm.shortwave_radiation),
        }

    return result


# ======================== AIR QUALITY DATA ========================

async def fetch_air_quality(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    location: Point,
    dt_utc: datetime,
) -> Optional[dict]:
    if is_past(dt_utc):
        return None

    params = {
        "latitude":   location.latitude,
        "longitude":  location.longitude,
        "hourly":     AIR_QUALITY_FIELDS,
        "start_date": dt_utc.strftime("%Y-%m-%d"),
        "end_date":   (dt_utc + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone":   "UTC",
    }

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(OPEN_METEO_AIR_QUALITY_URL, params=params) as r:
                    r.raise_for_status()
                    data   = await r.json()
                    hourly = data.get("hourly", {})
                    target = dt_utc.strftime("%Y-%m-%dT%H:00")

                    if target not in hourly.get("time", []):
                        print(f"[air quality] hour {target} not found.")
                        return None

                    def get(key: str) -> Optional[float]:
                        return _extract_hour(hourly, key, target)

                    return {
                        "pm10_μg_m3":            get("pm10"),
                        "pm2_5_μg_m3":           get("pm2_5"),
                        "co_μg_m3":              get("carbon_monoxide"),
                        "no2_μg_m3":             get("nitrogen_dioxide"),
                        "so2_μg_m3":             get("sulphur_dioxide"),
                        "o3_μg_m3":              get("ozone"),
                        "aerosol_optical_depth": get("aerosol_optical_depth"),
                        "dust_μg_m3":            get("dust"),
                        "uv_index":              get("uv_index"),
                        "european_aqi":          get("european_aqi"),
                        "us_aqi":                get("us_aqi"),
                    }

            except Exception as e:
                print(f"[air quality error] attempt={attempt} lat={location.latitude} lon={location.longitude} err={e}")
                await asyncio.sleep(0.5 * attempt)

    print(f"[air quality failed] lat={location.latitude} lon={location.longitude}")
    return None


# ======================== MAIN PIPELINE ========================

async def process_user(
    session: aiohttp.ClientSession,
    elev_sem: asyncio.Semaphore,
    weather_sem: asyncio.Semaphore,
    user_location: Point,
    dt_utc: datetime
) -> dict:

    user_elevation = await get_elevation_m(session, elev_sem, user_location)
    if user_elevation is None:
        return {"error": "Could not fetch elevation"}

    az                = get_azimuth_to_sunset(user_location, user_elevation, dt_utc)
    geometric_horizon = get_distance_to_horizon(user_elevation)

    horizon_info         = await get_visible_horizon(session, elev_sem, user_location, user_elevation, az, geometric_horizon)
    visible_horizon_dist = horizon_info["visible_horizon_dist_m"]
    blocking_angle       = horizon_info["blocking_angle_deg"]

    step           = visible_horizon_dist / (WEATHER_POINTS - 1)
    weather_points = [user_location] + [find_destination_point(user_location, step * i, az) for i in range(1, WEATHER_POINTS)]

    weather_tasks = [
        fetch_weather_data(session, weather_sem, p, dt_utc, include_full_weather=(i == 0))
        for i, p in enumerate(weather_points)
    ]

    air_quality_task = fetch_air_quality(session, weather_sem, user_location, dt_utc)

    *weather_results, air_quality = await asyncio.gather(*weather_tasks, air_quality_task)

    points_data = []
    for i, (point, result) in enumerate(zip(weather_points, weather_results)):
        entry = {
            "point_index": i,
            "lat":         point.latitude,
            "lon":         point.longitude,
            "distance_m":  round(step * i),
            "clouds":      result["clouds"] if result else None,
        }
        if i == 0 and result and "weather" in result:
            entry["weather"] = result["weather"]
        points_data.append(entry)

    viewer_data = points_data[0] if points_data else {}

    return {
        "viewer": {
            "lat":         user_location.latitude,
            "lon":         user_location.longitude,
            "elevation_m": user_elevation,
        },
        "azimuth_deg":            az,
        "visible_horizon_dist_m": visible_horizon_dist,
        "blocking_angle_deg":     blocking_angle,
        "date":                   dt_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "source":                 "archive" if is_past(dt_utc) else "forecast",
        "viewer_weather":         viewer_data.get("weather"),
        "viewer_clouds":          viewer_data.get("clouds"),
        "air_quality":            air_quality,
        "points":                 points_data,
    }


async def process_many(requests_list: list[dict]) -> list[dict]:
    elev_sem    = asyncio.Semaphore(ELEVATION_CONCURRENCY)
    weather_sem = asyncio.Semaphore(WEATHER_CONCURRENCY)
    connector   = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_user(session, elev_sem, weather_sem, Point(r["lat"], r["lon"]), r["dt_utc"])
            for r in requests_list
        ]
        return await asyncio.gather(*tasks)


def get_today_sunset_utc(lat: float, lon: float, tz: str = "Asia/Jerusalem") -> datetime:
    """Returns today's sunset time as a naive UTC datetime."""
    location = LocationInfo(latitude=lat, longitude=lon, timezone=tz)
    s = sun(location.observer, tzinfo=timezone.utc)
    return s["sunset"].astimezone(timezone.utc).replace(tzinfo=None)


# ======================== ENTRY POINT ========================

async def main(requests_list: list[dict]) -> str:
    """
    requests_list: [{"lat": float, "lon": float, "tz": str}, ...]
    Each entry may optionally include "dt_utc" (datetime); if omitted, today's sunset is used.
    """
    resolved = []
    for r in requests_list:
        dt_utc = r.get("dt_utc") or get_today_sunset_utc(r["lat"], r["lon"], tz=r.get("tz", "UTC"))
        print(f"Sunset UTC for ({r['lat']}, {r['lon']}): {dt_utc}")
        resolved.append({"lat": r["lat"], "lon": r["lon"], "dt_utc": dt_utc})

    results = await process_many(resolved)

    lines = []

    for result in results:
        if "error" in result:
            lines.append(f"Error: {result['error']}")
            continue

        lines.append(f"\n{'=' * 60}")
        lines.append(f"Viewer: {result['viewer']}  |  {result['date']}  [{result['source']}]")
        lines.append(
            f"Azimuth: {result['azimuth_deg']:.1f}°  |  Horizon: {result['visible_horizon_dist_m']:.0f}m  |  Blocking: {result['blocking_angle_deg']:.2f}°")

        w = result.get("viewer_weather")
        if w:
            lines.append(f"\n── Surface Weather ──")
            lines.append(f"  Temperature:       {w['temp_c']}°C  (feels like {w['apparent_temp_c']}°C)")
            lines.append(f"  Dewpoint:          {w['dewpoint_c']}°C")
            lines.append(f"  Humidity:          {w['humidity_pct']}%")
            lines.append(
                f"  Wind:              {w['wind_speed_kmh']} km/h @ {w['wind_dir_deg']}°  (gusts {w['wind_gusts_kmh']} km/h)")
            lines.append(f"  Precipitation:     {w['precip_mm']} mm  (prob {w['precip_prob_pct']}%)")
            lines.append(f"  Pressure:          {w['pressure_hpa']} hPa")
            lines.append(f"  Visibility:        {w['visibility_m']} m")
            lines.append(f"  Solar radiation:   {w['shortwave_radiation_wm2']} W/m²")

        c = result.get("viewer_clouds")
        if c:
            lines.append(f"\n── Clouds at Viewer ──")
            lines.append(f"  Low (<2 km):       {c['low_pct']}%")
            lines.append(f"  Mid (2–7 km):      {c['mid_pct']}%")
            lines.append(f"  High (>7 km):      {c['high_pct']}%")
            lines.append(f"  Total cover:       {c['total_pct']}%")
            lines.append(f"  Cloud types:       {', '.join(c['types'])}")

        aq = result.get("air_quality")
        if aq:
            lines.append(f"\n── Air Quality / Pollution ──")
            lines.append(f"  European AQI:      {aq['european_aqi']}  |  US AQI: {aq['us_aqi']}")
            lines.append(f"  PM10:              {aq['pm10_μg_m3']} μg/m³")
            lines.append(f"  PM2.5:             {aq['pm2_5_μg_m3']} μg/m³")
            lines.append(f"  NO₂:               {aq['no2_μg_m3']} μg/m³")
            lines.append(f"  O₃:                {aq['o3_μg_m3']} μg/m³")
            lines.append(f"  SO₂:               {aq['so2_μg_m3']} μg/m³")
            lines.append(f"  CO:                {aq['co_μg_m3']} μg/m³")
            lines.append(f"  Dust:              {aq['dust_μg_m3']} μg/m³")
            lines.append(f"  Aerosol opt. depth:{aq['aerosol_optical_depth']}")
            lines.append(f"  UV index:          {aq['uv_index']}")
        else:
            lines.append(f"\n  [air quality: not available for past dates]")

        lines.append(f"\n── Cloud Cover Along Sunset Azimuth ──")
        for pt in result["points"]:
            c = pt["clouds"]
            if c is None:
                lines.append(f"  [{pt['point_index']}] {pt['distance_m']:6d}m  ->  fetch failed")
            else:
                types_str = ", ".join(c["types"])
                lines.append(
                    f"  [{pt['point_index']}] {pt['distance_m']:6d}m  ->  "
                    f"low={c['low_pct']}%  mid={c['mid_pct']}%  high={c['high_pct']}%  "
                    f"total={c['total_pct']}%  |  {types_str}"
                )

    forecast_str = "\n".join(lines)
    print(forecast_str)
    return forecast_str


if __name__ == "__main__":
    asyncio.run(main(requests_list))