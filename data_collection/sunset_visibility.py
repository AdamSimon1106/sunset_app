from datetime import datetime, timezone
from typing import Optional
from math import sqrt, atan2, degrees
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
MAX_RETRIES              = 5
EYE_LEVEL_M              = 50
TERRAIN_SAMPLES          = 20
WEATHER_POINTS           = 4

ELEVATION_CONCURRENCY    = 10
WEATHER_CONCURRENCY      = 20

# Field names differ between endpoints
FORECAST_FIELDS = "cloudcover_low,cloudcover_mid,cloudcover_high"
ARCHIVE_FIELDS  = "cloud_cover_low,cloud_cover_mid,cloud_cover_high"


# ======================== HELPERS ========================

def is_past(dt_utc: datetime) -> bool:
    now = datetime.now(timezone.utc)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.date() < now.date()


def get_meteo_endpoint_and_fields(dt_utc: datetime) -> tuple[str, str, str, str, str]:
    """Returns (endpoint, fields, low_key, mid_key, high_key)"""
    if is_past(dt_utc):
        return OPEN_METEO_ARCHIVE_URL, ARCHIVE_FIELDS, "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"
    else:
        return OPEN_METEO_FORECAST_URL, FORECAST_FIELDS, "cloudcover_low", "cloudcover_mid", "cloudcover_high"


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
    return 200


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


# ======================== CLOUD DATA ========================

async def fetch_cloud_data(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    location: Point,
    dt_utc: datetime
) -> Optional[dict]:
    """
    Fetch low, mid, high cloud cover % for a single point at the given hour.
    Automatically picks correct endpoint AND field names for past vs future.
    Returns: { "low": int, "mid": int, "high": int }
    """
    endpoint, fields, low_key, mid_key, high_key = get_meteo_endpoint_and_fields(dt_utc)

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
                async with session.get(endpoint, params=params) as r:
                    r.raise_for_status()
                    data   = await r.json()
                    hourly = data.get("hourly", {})
                    times  = hourly.get("time", [])
                    target = dt_utc.strftime("%Y-%m-%dT%H:00")

                    if target not in times:
                        print(f"[cloud data] hour {target} not found. endpoint={endpoint} available: {times[:3]}...{times[-3:]}")
                        return None

                    idx = times.index(target)
                    return {
                        "low":  hourly[low_key][idx],
                        "mid":  hourly[mid_key][idx],
                        "high": hourly[high_key][idx],
                    }
            except Exception as e:
                print(f"[cloud fetch error] attempt={attempt} lat={location.latitude} lon={location.longitude} err={e}")
                await asyncio.sleep(0.5 * attempt)

    print(f"[cloud fetch failed] lat={location.latitude} lon={location.longitude}")
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

    cloud_tasks     = [fetch_cloud_data(session, weather_sem, p, dt_utc) for p in weather_points]
    cloud_per_point = await asyncio.gather(*cloud_tasks)

    points_data = []
    for i, (point, clouds) in enumerate(zip(weather_points, cloud_per_point)):
        points_data.append({
            "point_index": i,
            "lat":         point.latitude,
            "lon":         point.longitude,
            "distance_m":  round(step * i),
            "clouds":      clouds
        })

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
        "points":                 points_data
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
        {"lat": 32.0853, "lon": 34.7818, "dt_utc": datetime(2026, 4, 15, 16, 30, 0)},   # past
        {"lat": 40.630182059416654, "lon": 14.384823257952677, "dt_utc": datetime(2026, 5, 9, 17, 0, 0)},  # future
    ]

    results = await process_many(requests_list)

    for result in results:
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
        print(f"\nViewer: {result['viewer']}  |  {result['date']}  [{result['source']}]")
        print(f"Azimuth: {result['azimuth_deg']:.1f}°  |  Horizon: {result['visible_horizon_dist_m']:.0f}m  |  Blocking: {result['blocking_angle_deg']:.2f}°")
        for pt in result["points"]:
            clouds = pt["clouds"]
            if clouds is None:
                print(f"  [{pt['point_index']}] {pt['distance_m']:6d}m  ->  cloud fetch failed")
            else:
                print(f"  [{pt['point_index']}] {pt['distance_m']:6d}m  ->  low={clouds['low']}%  mid={clouds['mid']}%  high={clouds['high']}%")


if __name__ == "__main__":
    asyncio.run(main())