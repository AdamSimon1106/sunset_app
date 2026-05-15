from datetime import datetime, timezone
from typing import Optional
from math import sqrt, atan2, degrees
from dataclasses import dataclass, field
import asyncio
import aiohttp
import pytz
from astral import Observer
from astral.sun import sunset, azimuth
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


# ======================== FIELD MAPPINGS ========================
# Open-Meteo uses different field names between forecast and archive endpoints.
# We centralise all mappings here so fetch logic stays clean.

@dataclass
class MeteoFieldMap:
    """Canonical -> actual API field name mapping for one endpoint."""
    endpoint: str

    # Cloud cover
    cloudcover_low:  str = "cloudcover_low"
    cloudcover_mid:  str = "cloudcover_mid"
    cloudcover_high: str = "cloudcover_high"
    cloudcover_total: str = "cloudcover"

    # Temperature / humidity
    temperature_2m:       str = "temperature_2m"
    apparent_temperature: str = "apparent_temperature"
    dewpoint_2m:          str = "dewpoint_2m"
    relative_humidity_2m: str = "relative_humidity_2m"

    # Wind
    windspeed_10m:    str = "windspeed_10m"
    winddirection_10m: str = "winddirection_10m"
    windgusts_10m:    str = "windgusts_10m"

    # Precipitation & pressure
    precipitation:         str = "precipitation"
    precipitation_probability: str = "precipitation_probability"
    surface_pressure:      str = "surface_pressure"

    # Visibility & radiation
    visibility:            str = "visibility"
    shortwave_radiation:   str = "shortwave_radiation"

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


# Forecast endpoint uses camelCase-ish names; archive uses snake_case
FORECAST_MAP = MeteoFieldMap(endpoint=OPEN_METEO_FORECAST_URL)

ARCHIVE_MAP = MeteoFieldMap(
    endpoint=OPEN_METEO_ARCHIVE_URL,
    cloudcover_low="cloud_cover_low",
    cloudcover_mid="cloud_cover_mid",
    cloudcover_high="cloud_cover_high",
    cloudcover_total="cloud_cover",
    apparent_temperature="apparent_temperature",   # same in archive
    dewpoint_2m="dew_point_2m",
    relative_humidity_2m="relative_humidity_2m",   # same
    windspeed_10m="wind_speed_10m",
    winddirection_10m="wind_direction_10m",
    windgusts_10m="wind_gusts_10m",
    precipitation="precipitation",
    precipitation_probability="precipitation_probability",
    surface_pressure="surface_pressure",
    visibility="visibility",
    shortwave_radiation="shortwave_radiation",
)

# Air quality variables (forecast only — CAMS does not archive)
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
# Derived from standard WMO altitude bands and cloud-cover thresholds.
# This is a purely local heuristic — no additional API call needed.
#
# Low  (< 2 km): Stratus, Stratocumulus, Nimbostratus, Cumulonimbus, Cumulus
# Mid  (2–7 km): Altostratus, Altocumulus
# High (> 7 km): Cirrus, Cirrostratus, Cirrocumulus

def infer_cloud_types(
    low_pct: Optional[float],
    mid_pct: Optional[float],
    high_pct: Optional[float],
    total_pct: Optional[float],
    precipitation: Optional[float],
    temp_2m: Optional[float],
) -> list[str]:
    """
    Returns a list of likely cloud type names based on coverage percentages
    and surface conditions. Order reflects decreasing likelihood.
    """
    types: list[str] = []
    low   = low_pct   or 0.0
    mid   = mid_pct   or 0.0
    high  = high_pct  or 0.0
    total = total_pct or 0.0
    precip = precipitation or 0.0

    # ── High clouds (cirrus family) ──────────────────────────────────────
    if high >= 60:
        types.append("Cirrostratus")
    elif high >= 30:
        types.append("Cirrus")
    if high >= 40 and mid < 20:
        types.append("Cirrocumulus")

    # ── Mid clouds (alto family) ─────────────────────────────────────────
    if mid >= 70:
        types.append("Altostratus")
    elif mid >= 25:
        types.append("Altocumulus")

    # ── Low clouds ───────────────────────────────────────────────────────
    if low >= 80 and precip > 0.5:
        # Heavy low cloud + precipitation → Nimbostratus (or Cb if convective)
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

    # ── Clear sky ────────────────────────────────────────────────────────
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
    return None


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

async def fetch_weather_data(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    location: Point,
    dt_utc: datetime,
    include_full_weather: bool = False,
) -> Optional[dict]:
    """
    Fetch cloud cover (all altitudes) and — when include_full_weather=True —
    the complete surface weather snapshot for a single point at the given hour.

    Returns a dict with keys:
        clouds: { low, mid, high, total, types }
        weather (only if include_full_weather=True): { temp_c, apparent_temp_c,
            dewpoint_c, humidity_pct, wind_speed_kmh, wind_dir_deg,
            wind_gusts_kmh, precip_mm, precip_prob_pct, pressure_hpa,
            visibility_m, shortwave_radiation_wm2 }
    """
    fm = get_field_map(dt_utc)
    fields = fm.weather_fields() if include_full_weather else fm.cloud_fields()

    params = {
        "latitude":   location.latitude,
        "longitude":  location.longitude,
        "hourly":     fields,
        "start_date": dt_utc.strftime("%Y-%m-%d"),
        "end_date":   dt_utc.strftime("%Y-%m-%d"),
        "timezone":   "UTC",
    }

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(fm.endpoint, params=params) as r:
                    r.raise_for_status()
                    data   = await r.json()
                    hourly = data.get("hourly", {})
                    target = dt_utc.strftime("%Y-%m-%dT%H:00")

                    if target not in hourly.get("time", []):
                        print(f"[weather data] hour {target} not in response. available: {hourly.get('time', [])[:3]}...")
                        return None

                    def get(key: str) -> Optional[float]:
                        return _extract_hour(hourly, key, target)

                    low   = get(fm.cloudcover_low)
                    mid   = get(fm.cloudcover_mid)
                    high  = get(fm.cloudcover_high)
                    total = get(fm.cloudcover_total)
                    precip = get(fm.precipitation) if include_full_weather else None
                    temp   = get(fm.temperature_2m) if include_full_weather else None

                    result: dict = {
                        "clouds": {
                            "low_pct":   low,
                            "mid_pct":   mid,
                            "high_pct":  high,
                            "total_pct": total,
                            "types":     infer_cloud_types(low, mid, high, total, precip, temp),
                        }
                    }

                    if include_full_weather:
                        result["weather"] = {
                            "temp_c":               temp,
                            "apparent_temp_c":      get(fm.apparent_temperature),
                            "dewpoint_c":           get(fm.dewpoint_2m),
                            "humidity_pct":         get(fm.relative_humidity_2m),
                            "wind_speed_kmh":       get(fm.windspeed_10m),
                            "wind_dir_deg":         get(fm.winddirection_10m),
                            "wind_gusts_kmh":       get(fm.windgusts_10m),
                            "precip_mm":            precip,
                            "precip_prob_pct":      get(fm.precipitation_probability),
                            "pressure_hpa":         get(fm.surface_pressure),
                            "visibility_m":         get(fm.visibility),
                            "shortwave_radiation_wm2": get(fm.shortwave_radiation),
                        }

                    return result

            except Exception as e:
                print(f"[weather fetch error] attempt={attempt} lat={location.latitude} lon={location.longitude} err={e}")
                await asyncio.sleep(0.5 * attempt)

    print(f"[weather fetch failed] lat={location.latitude} lon={location.longitude}")
    return None


# ======================== AIR QUALITY DATA ========================

async def fetch_air_quality(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    location: Point,
    dt_utc: datetime,
) -> Optional[dict]:
    """
    Fetch air quality / pollution data for a single location at the given hour.
    CAMS only provides forecasts (not archive), so returns None for past dates.

    Returns: { pm10, pm2_5, co_μg_m3, no2_μg_m3, so2_μg_m3, o3_μg_m3,
               aerosol_optical_depth, dust_μg_m3, uv_index,
               european_aqi, us_aqi }
    """
    if is_past(dt_utc):
        return None  # CAMS air quality archive not available via this endpoint

    params = {
        "latitude":   location.latitude,
        "longitude":  location.longitude,
        "hourly":     AIR_QUALITY_FIELDS,
        "start_date": dt_utc.strftime("%Y-%m-%d"),
        "end_date":   dt_utc.strftime("%Y-%m-%d"),
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
                        "pm10_μg_m3":             get("pm10"),
                        "pm2_5_μg_m3":            get("pm2_5"),
                        "co_μg_m3":               get("carbon_monoxide"),
                        "no2_μg_m3":              get("nitrogen_dioxide"),
                        "so2_μg_m3":              get("sulphur_dioxide"),
                        "o3_μg_m3":               get("ozone"),
                        "aerosol_optical_depth":  get("aerosol_optical_depth"),
                        "dust_μg_m3":             get("dust"),
                        "uv_index":               get("uv_index"),
                        "european_aqi":           get("european_aqi"),
                        "us_aqi":                 get("us_aqi"),
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

    # ── Weather points along the sunset azimuth ──────────────────────────
    step           = visible_horizon_dist / (WEATHER_POINTS - 1)
    weather_points = [user_location] + [find_destination_point(user_location, step * i, az) for i in range(1, WEATHER_POINTS)]

    # User's own location (index 0) gets full weather; other points cloud-only.
    weather_tasks = [
        fetch_weather_data(session, weather_sem, p, dt_utc, include_full_weather=(i == 0))
        for i, p in enumerate(weather_points)
    ]

    # Air quality runs concurrently alongside weather fetches, same semaphore.
    air_quality_task = fetch_air_quality(session, weather_sem, user_location, dt_utc)

    # Fire all fetches in parallel.
    *weather_results, air_quality = await asyncio.gather(*weather_tasks, air_quality_task)

    # ── Assemble point data ───────────────────────────────────────────────
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

    # Convenience alias: pull viewer weather + clouds to top level
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


# ======================== EXAMPLE ========================

async def main():
    requests_list = [
        {"lat": 32.08, "lon": 34.77, "dt_utc": datetime(2026, 5, 15, 16, 0, 0)},  # Tel Aviv, future
    ]

    results = await process_many(requests_list)

    for result in results:
        if "error" in result:
            print(f"Error: {result['error']}")
            continue

        print(f"\n{'='*60}")
        print(f"Viewer: {result['viewer']}  |  {result['date']}  [{result['source']}]")
        print(f"Azimuth: {result['azimuth_deg']:.1f}°  |  Horizon: {result['visible_horizon_dist_m']:.0f}m  |  Blocking: {result['blocking_angle_deg']:.2f}°")

        # ── Surface weather ──────────────────────────────────────────────
        w = result.get("viewer_weather")
        if w:
            print(f"\n── Surface Weather ──")
            print(f"  Temperature:       {w['temp_c']}°C  (feels like {w['apparent_temp_c']}°C)")
            print(f"  Dewpoint:          {w['dewpoint_c']}°C")
            print(f"  Humidity:          {w['humidity_pct']}%")
            print(f"  Wind:              {w['wind_speed_kmh']} km/h @ {w['wind_dir_deg']}°  (gusts {w['wind_gusts_kmh']} km/h)")
            print(f"  Precipitation:     {w['precip_mm']} mm  (prob {w['precip_prob_pct']}%)")
            print(f"  Pressure:          {w['pressure_hpa']} hPa")
            print(f"  Visibility:        {w['visibility_m']} m")
            print(f"  Solar radiation:   {w['shortwave_radiation_wm2']} W/m²")

        # ── Clouds ───────────────────────────────────────────────────────
        c = result.get("viewer_clouds")
        if c:
            print(f"\n── Clouds at Viewer ──")
            print(f"  Low (<2 km):       {c['low_pct']}%")
            print(f"  Mid (2–7 km):      {c['mid_pct']}%")
            print(f"  High (>7 km):      {c['high_pct']}%")
            print(f"  Total cover:       {c['total_pct']}%")
            print(f"  Cloud types:       {', '.join(c['types'])}")

        # ── Air quality ──────────────────────────────────────────────────
        aq = result.get("air_quality")
        if aq:
            print(f"\n── Air Quality / Pollution ──")
            print(f"  European AQI:      {aq['european_aqi']}  |  US AQI: {aq['us_aqi']}")
            print(f"  PM10:              {aq['pm10_μg_m3']} μg/m³")
            print(f"  PM2.5:             {aq['pm2_5_μg_m3']} μg/m³")
            print(f"  NO₂:               {aq['no2_μg_m3']} μg/m³")
            print(f"  O₃:                {aq['o3_μg_m3']} μg/m³")
            print(f"  SO₂:               {aq['so2_μg_m3']} μg/m³")
            print(f"  CO:                {aq['co_μg_m3']} μg/m³")
            print(f"  Dust:              {aq['dust_μg_m3']} μg/m³")
            print(f"  Aerosol opt. depth:{aq['aerosol_optical_depth']}")
            print(f"  UV index:          {aq['uv_index']}")
        else:
            print(f"\n  [air quality: not available for past dates]")

        # ── Clouds along the sunset path ─────────────────────────────────
        print(f"\n── Cloud Cover Along Sunset Azimuth ──")
        for pt in result["points"]:
            c = pt["clouds"]
            if c is None:
                print(f"  [{pt['point_index']}] {pt['distance_m']:6d}m  ->  fetch failed")
            else:
                types_str = ", ".join(c["types"])
                print(
                    f"  [{pt['point_index']}] {pt['distance_m']:6d}m  ->  "
                    f"low={c['low_pct']}%  mid={c['mid_pct']}%  high={c['high_pct']}%  "
                    f"total={c['total_pct']}%  |  {types_str}"
                )


if __name__ == "__main__":
    asyncio.run(main()),