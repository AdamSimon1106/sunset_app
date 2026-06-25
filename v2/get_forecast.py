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

# Cloud cover is taken as a consensus across several independent global models rather
# than a single one: any single model is biased here (e.g. ICON over-forecasts a phantom
# marine stratocumulus deck on the eastern-Med coast, while ECMWF under-calls it). Polling
# all of them in one request and taking the median outvotes a lone outlier. Surface weather
# is read from the first available of these (in order).
CLOUD_MODELS = [
    "ecmwf_ifs025",
    "icon_seamless",
    "gfs_seamless",
    "gem_seamless",
    "meteofrance_seamless",
    "ukmo_seamless",
]

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


def _median(nums: list[float]) -> float:
    """Median of a non-empty list (averages the middle pair for even counts)."""
    s = sorted(nums)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _consensus(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float], int]:
    """
    Combine per-model values for one cloud layer into a robust consensus.
    Drops None (model not available for this point/hour), returns
    (median, spread=max-min, n_models). Median rejects a lone outlier — e.g.
    [55, 0, 0, 0, 2, 9] -> median 1.0, where a mean would be 11.0.
    """
    nums = [v for v in values if v is not None]
    if not nums:
        return None, None, 0
    return _median(nums), max(nums) - min(nums), len(nums)


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

    # Primary: poll all CLOUD_MODELS in one request. Open-Meteo suffixes each field with
    # the model name, e.g. "cloud_cover_low_ecmwf_ifs025".
    hourly   = await _fetch_hourly(
        session, sem, fm.endpoint,
        {**base_params, "models": ",".join(CLOUD_MODELS)},
        target, label="multi_model",
    )
    models   = list(CLOUD_MODELS)
    suffixed = True

    def field(base: str, model: str) -> str:
        return f"{base}_{model}" if suffixed else base

    def has_cloud(h: Optional[dict]) -> bool:
        return h is not None and any(
            _extract_hour(h, field(fm.cloudcover_low, m), target) is not None for m in models
        )

    # Last-resort fallback: a single best_match request (unsuffixed fields) if the
    # multi-model call failed or returned nothing usable for this point/hour.
    if not has_cloud(hourly):
        hourly         = await _fetch_hourly(
            session, sem, fm.endpoint, base_params, target, label="best_match_fallback",
        )
        models, suffixed = ["best_match"], False
        if hourly is None:
            return None

    def get(key: str) -> Optional[float]:
        return _extract_hour(hourly, key, target)

    def get_first(base: str) -> Optional[float]:
        """First available model's value for a field (for non-cloud surface fields)."""
        for m in models:
            v = get(field(base, m))
            if v is not None:
                return v
        return None

    def layer(base: str) -> tuple[Optional[float], Optional[float], int]:
        return _consensus([get(field(base, m)) for m in models])

    low,   low_spread,   _      = layer(fm.cloudcover_low)
    mid,   mid_spread,   _      = layer(fm.cloudcover_mid)
    high,  high_spread,  _      = layer(fm.cloudcover_high)
    total, _,            n_total = layer(fm.cloudcover_total)

    n_models = n_total
    spreads  = [s for s in (low_spread, mid_spread, high_spread) if s is not None]
    spread   = max(spreads) if spreads else None

    # total can be non-zero even when all three layers are 0 (max-overlap cloud fraction).
    # Use effective_total = max(reported_total, max_layer) so cloud type inference is consistent.
    layer_max       = max(low or 0.0, mid or 0.0, high or 0.0)
    effective_total = max(total or 0.0, layer_max)

    precip = get_first(fm.precipitation)   if include_full_weather else None
    temp   = get_first(fm.temperature_2m)  if include_full_weather else None

    result: dict = {
        "clouds": {
            "low_pct":   low,
            "mid_pct":   mid,
            "high_pct":  high,
            "total_pct": total,
            "n_models":  n_models,
            "spread_pct": spread,
            "types":     infer_cloud_types(low, mid, high, effective_total, precip, temp),
        }
    }

    if include_full_weather:
        result["weather"] = {
            "temp_c":                  temp,
            "apparent_temp_c":         get_first(fm.apparent_temperature),
            "dewpoint_c":              get_first(fm.dewpoint_2m),
            "humidity_pct":            get_first(fm.relative_humidity_2m),
            "wind_speed_kmh":          get_first(fm.windspeed_10m),
            "wind_dir_deg":            get_first(fm.winddirection_10m),
            "wind_gusts_kmh":          get_first(fm.windgusts_10m),
            "precip_mm":               precip,
            "precip_prob_pct":         get_first(fm.precipitation_probability),
            "pressure_hpa":            get_first(fm.surface_pressure),
            "visibility_m":            get_first(fm.visibility),
            "shortwave_radiation_wm2": get_first(fm.shortwave_radiation),
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


# Minutes before sunset we recommend arriving.
ARRIVAL_LEAD_MIN = 25


def local_sunset_times(dt_utc: datetime, tz: str) -> tuple[str, str]:
    """
    Convert a naive-UTC sunset instant to the location's local wall-clock time,
    DST-correct via ZoneInfo. Returns ("HH:MM" sunset, "HH:MM" recommended arrival).
    """
    local   = dt_utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz))
    arrival = local - timedelta(minutes=ARRIVAL_LEAD_MIN)
    return local.strftime("%H:%M"), arrival.strftime("%H:%M")


# ======================== ENTRY POINT ========================

async def main(requests_list: list[dict]) -> str:
    """
    requests_list: [{"lat": float, "lon": float, "tz": str}, ...]
    Each entry may optionally include "dt_utc" (datetime); if omitted, today's sunset is used.
    """
    resolved = []
    for r in requests_list:
        tz     = r.get("tz", "UTC")
        dt_utc = r.get("dt_utc") or get_today_sunset_utc(r["lat"], r["lon"], tz=tz)
        print(f"Sunset UTC for ({r['lat']}, {r['lon']}): {dt_utc}")
        resolved.append({"lat": r["lat"], "lon": r["lon"], "dt_utc": dt_utc, "tz": tz})

    results = await process_many(resolved)

    lines = []

    for result, r in zip(results, resolved):
        if "error" in result:
            lines.append(f"Error: {result['error']}")
            continue

        # Authoritative local sunset/arrival times, computed here (DST-correct) so the
        # model never has to convert UTC itself.
        sunset_local, arrival_local = local_sunset_times(r["dt_utc"], r["tz"])

        lines.append(f"\n{'=' * 60}")
        lines.append(f"Viewer: {result['viewer']}  |  {result['date']}  [{result['source']}]")
        lines.append(f"Sunset (local): {sunset_local}  |  Recommended arrival: {arrival_local}")
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
            spread = c.get("spread_pct")
            agree  = (f"{c.get('n_models', 0)} models, spread {spread:.0f}%"
                      if spread is not None else f"{c.get('n_models', 0)} models")
            lines.append(f"\n── Clouds at Viewer (consensus of {agree}) ──")
            lines.append(f"  Low (<3 km):       {c['low_pct']}%")
            lines.append(f"  Mid (3–8 km):      {c['mid_pct']}%")
            lines.append(f"  High (>8 km):      {c['high_pct']}%")
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
                spread    = c.get("spread_pct")
                agree     = f"  [{c.get('n_models', 0)}m, ±{spread:.0f}]" if spread is not None else ""
                lines.append(
                    f"  [{pt['point_index']}] {pt['distance_m']:6d}m  ->  "
                    f"low={c['low_pct']}%  mid={c['mid_pct']}%  high={c['high_pct']}%  "
                    f"total={c['total_pct']}%  |  {types_str}{agree}"
                )

    forecast_str = "\n".join(lines)
    print(forecast_str)
    return forecast_str


if __name__ == "__main__":
    asyncio.run(main(requests_list))