"""
Per-model cloud comparison tool.

The main pipeline (get_forecast.py) collapses the 6 global models into a single
median consensus, which hides *which* model is off. This script fetches the same
Open-Meteo request and prints every model's cloud values side by side, plus the
median/spread the pipeline would actually use — so you can see at a glance which
model is the outlier when the forecast looks wrong.

Usage:
    python3 compare_models.py                  # TLV, today's sunset
    python3 compare_models.py 40.7128 -74.006  # custom lat/lon, today's sunset
    python3 compare_models.py 32.08 34.77 2026-06-25T16:00   # explicit UTC hour
"""

import asyncio
import sys
from datetime import datetime, timezone

import aiohttp
from geopy import Point

from get_forecast import (
    CLOUD_MODELS,
    TLV_LAT,
    TLV_LON,
    WEATHER_CONCURRENCY,
    MeteoFieldMap,
    _consensus,
    _extract_hour,
    _fetch_hourly,
    get_field_map,
    get_today_sunset_utc,
)

# The four cloud layers we compare, as (label, canonical-field) pairs.
LAYERS = [
    ("Low  (<3 km)",  "cloudcover_low"),
    ("Mid  (3–8 km)", "cloudcover_mid"),
    ("High (>8 km)",  "cloudcover_high"),
    ("Total",         "cloudcover_total"),
]


def _fmt(v, width=9):
    return f"{v:{width}.0f}" if v is not None else f"{'·':>{width}}"


async def compare_models(location: Point, dt_utc: datetime) -> str:
    fm: MeteoFieldMap = get_field_map(dt_utc)
    target = dt_utc.strftime("%Y-%m-%dT%H:00")

    params = {
        "latitude":   location.latitude,
        "longitude":  location.longitude,
        "hourly":     fm.cloud_fields(),
        "start_date": dt_utc.strftime("%Y-%m-%d"),
        "end_date":   dt_utc.strftime("%Y-%m-%d"),
        "timezone":   "UTC",
        "models":     ",".join(CLOUD_MODELS),
    }

    sem = asyncio.Semaphore(WEATHER_CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        hourly = await _fetch_hourly(
            session, sem, fm.endpoint, params, target, label="compare",
        )

    lines: list[str] = []
    lines.append(f"\n{'=' * 78}")
    lines.append(
        f"Cloud model comparison  |  ({location.latitude}, {location.longitude})  "
        f"|  {target} UTC  [{fm.endpoint.split('//')[1].split('/')[0]}]"
    )
    lines.append("=" * 78)

    if hourly is None:
        lines.append("  Request failed — no data returned.")
        return "\n".join(lines)

    # Header row: one column per model.
    short = [m.replace("_seamless", "").replace("_ifs025", "")[:8] for m in CLOUD_MODELS]
    header = f"  {'Layer':<14}" + "".join(f"{s:>9}" for s in short) + f"  │{'med':>7}{'spread':>9}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for label, canonical in LAYERS:
        base = getattr(fm, canonical)
        per_model = [
            _extract_hour(hourly, f"{base}_{m}", target) for m in CLOUD_MODELS
        ]
        median, spread, n = _consensus(per_model)
        row = f"  {label:<14}" + "".join(_fmt(v) for v in per_model)
        row += f"  │{_fmt(median, 7)}{_fmt(spread, 9)}"
        lines.append(row)

    # Flag models that disagree strongly with the median on any layer.
    lines.append("  " + "-" * (len(header) - 2))
    outliers = []
    for canonical in (c for _, c in LAYERS):
        base = getattr(fm, canonical)
        per_model = [_extract_hour(hourly, f"{base}_{m}", target) for m in CLOUD_MODELS]
        median, _, _ = _consensus(per_model)
        if median is None:
            continue
        for m, v in zip(CLOUD_MODELS, per_model):
            if v is not None and abs(v - median) >= 40:
                outliers.append(f"{m} ({canonical.replace('cloudcover_', '')}: {v:.0f}% vs median {median:.0f}%)")
    if outliers:
        lines.append("  ⚠ Strong disagreement (≥40% from median):")
        for o in outliers:
            lines.append(f"      • {o}")
    else:
        lines.append("  ✓ Models broadly agree (no layer >40% from median).")

    return "\n".join(lines)


def _parse_args(argv: list[str]) -> tuple[Point, datetime]:
    lat = float(argv[1]) if len(argv) > 1 else TLV_LAT
    lon = float(argv[2]) if len(argv) > 2 else TLV_LON
    if len(argv) > 3:
        dt_utc = datetime.strptime(argv[3], "%Y-%m-%dT%H:%M")
    else:
        dt_utc = get_today_sunset_utc(lat, lon)
    return Point(lat, lon), dt_utc


async def main(argv: list[str]) -> None:
    location, dt_utc = _parse_args(argv)
    print(f"Sunset/target UTC: {dt_utc}")
    print(await compare_models(location, dt_utc))


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
