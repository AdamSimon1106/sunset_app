"""
Per-model cloud accuracy tracker.

Instead of hard-coding which model to trust (the get_forecast.py comment claims
ICON over-forecasts coastal stratocumulus — but on 2026-06-25 ICON was the only
model that called the deck correctly), this module *measures* it.

Workflow:
  1. `log`     — at/near sunset, record every model's predicted cloud layers for
                 the sunset hour. Append-only JSONL, one record per location/day.
  2. `observe` — enter what the sky actually looked like (ground truth, by eye).
  3. `backfill`— for past records with no manual observation, pull ERA5 reanalysis
                 from the archive as an automatic (independent) proxy for truth.
  4. `report`  — aggregate per-model error and rank models. The ranking, not a
                 hand-written comment, is what should eventually drive weighting.

Truth precedence: a manual `observe` always overrides the ERA5 backfill, since a
human looking at the sky beats a reanalysis.

Usage:
    python3 model_accuracy.py log                     # TLV, today's sunset
    python3 model_accuracy.py log 40.71 -74.0         # custom lat/lon
    python3 model_accuracy.py observe 2026-06-25 48 0 0   # actual low mid high (%)
    python3 model_accuracy.py backfill                # fill ERA5 actuals for old rows
    python3 model_accuracy.py report
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import aiohttp
from geopy import Point

from get_forecast import (
    CLOUD_MODELS,
    TLV_LAT,
    TLV_LON,
    WEATHER_CONCURRENCY,
    ARCHIVE_MAP,
    MeteoFieldMap,
    _consensus,
    _extract_hour,
    _fetch_hourly,
    get_field_map,
    get_today_sunset_utc,
)

LOG_PATH = os.path.join(os.path.dirname(__file__), "model_accuracy_log.jsonl")
LAYERS = ["low", "mid", "high"]  # total is derived; we score the three physical layers
_CANON = {
    "low":  "cloudcover_low",
    "mid":  "cloudcover_mid",
    "high": "cloudcover_high",
}


# ======================== STORAGE ========================

def _read_log() -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_log(records: list[dict]) -> None:
    with open(LOG_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _append_log(record: dict) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _key(rec: dict) -> tuple:
    return (rec["date"], round(rec["lat"], 3), round(rec["lon"], 3))


# ======================== LOG PREDICTIONS ========================

async def _fetch_all_models(location: Point, dt_utc: datetime, fm: MeteoFieldMap) -> dict:
    """Return {model: {low, mid, high}} for the target hour, plus the raw hourly dict."""
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
        hourly = await _fetch_hourly(session, sem, fm.endpoint, params, target, label="accuracy_log")

    preds: dict = {}
    if hourly is None:
        return preds
    for m in CLOUD_MODELS:
        preds[m] = {
            layer: _extract_hour(hourly, f"{getattr(fm, _CANON[layer])}_{m}", target)
            for layer in LAYERS
        }
    return preds


async def cmd_log(location: Point, dt_utc: datetime) -> None:
    fm = get_field_map(dt_utc)
    target = dt_utc.strftime("%Y-%m-%dT%H:00")
    preds = await _fetch_all_models(location, dt_utc, fm)
    if not preds:
        print("No prediction data returned — nothing logged.")
        return

    median = {
        layer: _consensus([preds[m][layer] for m in CLOUD_MODELS])[0]
        for layer in LAYERS
    }

    record = {
        "date":         dt_utc.strftime("%Y-%m-%d"),
        "target_hour":  target,
        "lat":          location.latitude,
        "lon":          location.longitude,
        "source":       "archive" if fm is ARCHIVE_MAP else "forecast",
        "predictions":  preds,
        "median":       median,
        "actual":       None,   # {low, mid, high}
        "actual_source": None,  # "observed" | "era5"
    }

    # De-dupe: replace an existing record for the same (date, location).
    records = [r for r in _read_log() if _key(r) != _key(record)]
    records.append(record)
    _write_log(records)
    print(f"Logged {len(CLOUD_MODELS)} models for {record['date']} @ {target} UTC.")
    print(f"  median: low={median['low']} mid={median['mid']} high={median['high']}")


# ======================== GROUND TRUTH ========================

def cmd_observe(date_str: str, low: float, mid: float, high: float,
                lat: float = TLV_LAT, lon: float = TLV_LON) -> None:
    records = _read_log()
    target_key = (date_str, round(lat, 3), round(lon, 3))
    for r in records:
        if _key(r) == target_key:
            r["actual"] = {"low": low, "mid": mid, "high": high}
            r["actual_source"] = "observed"
            _write_log(records)
            print(f"Recorded observed sky for {date_str}: low={low} mid={mid} high={high}")
            return
    print(f"No logged prediction found for {date_str} at ({lat}, {lon}). Run `log` first.")


async def cmd_backfill() -> None:
    """Fill ERA5 reanalysis actuals for past records lacking a manual observation."""
    records = _read_log()
    sem = asyncio.Semaphore(WEATHER_CONCURRENCY)
    filled = 0
    async with aiohttp.ClientSession() as session:
        for r in records:
            if r.get("actual") is not None:
                continue  # already have truth (observed or era5)
            dt_utc = datetime.strptime(r["target_hour"], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            if dt_utc.date() >= datetime.now(timezone.utc).date():
                continue  # not in the past yet — archive won't have it
            fm = ARCHIVE_MAP
            params = {
                "latitude":   r["lat"],
                "longitude":  r["lon"],
                "hourly":     fm.cloud_fields(),
                "start_date": dt_utc.strftime("%Y-%m-%d"),
                "end_date":   dt_utc.strftime("%Y-%m-%d"),
                "timezone":   "UTC",
            }
            hourly = await _fetch_hourly(session, sem, fm.endpoint, params, r["target_hour"], label="era5")
            if hourly is None:
                continue
            actual = {layer: _extract_hour(hourly, getattr(fm, _CANON[layer]), r["target_hour"]) for layer in LAYERS}
            if all(v is None for v in actual.values()):
                continue
            r["actual"] = actual
            r["actual_source"] = "era5"
            filled += 1
    _write_log(records)
    print(f"Backfilled ERA5 actuals for {filled} record(s).")


# ======================== REPORT ========================

def cmd_report() -> None:
    records = [r for r in _read_log() if r.get("actual")]
    if not records:
        print("No records with ground truth yet. Run `observe` or `backfill` first.")
        return

    # Per-model accumulated absolute error and per-day "win" counts.
    abs_err: dict[str, list[float]] = {m: [] for m in CLOUD_MODELS}
    abs_err["median"] = []
    wins: dict[str, int] = {m: 0 for m in CLOUD_MODELS}
    wins["median"] = 0

    for r in records:
        actual = r["actual"]
        contenders = {**r["predictions"], "median": r["median"]}
        day_err: dict[str, float] = {}
        for name, layers in contenders.items():
            errs = [
                abs(layers[l] - actual[l])
                for l in LAYERS
                if layers.get(l) is not None and actual.get(l) is not None
            ]
            if errs:
                mae = sum(errs) / len(errs)
                abs_err[name].append(mae)
                day_err[name] = mae
        if day_err:
            best = min(day_err, key=day_err.get)
            wins[best] += 1

    n_days = len(records)
    truth_mix = {}
    for r in records:
        truth_mix[r["actual_source"]] = truth_mix.get(r["actual_source"], 0) + 1

    print(f"\n{'=' * 62}")
    print(f"Model accuracy over {n_days} day(s)   truth: "
          + ", ".join(f"{k}×{v}" for k, v in truth_mix.items()))
    print("=" * 62)
    print(f"  {'model':<22}{'MAE %':>9}{'days':>7}{'wins':>7}")
    print("  " + "-" * 43)

    ranked = sorted(
        abs_err.items(),
        key=lambda kv: (sum(kv[1]) / len(kv[1])) if kv[1] else 1e9,
    )
    for name, errs in ranked:
        if not errs:
            continue
        mae = sum(errs) / len(errs)
        tag = "  ← consensus" if name == "median" else ""
        print(f"  {name:<22}{mae:9.1f}{len(errs):7d}{wins.get(name, 0):7d}{tag}")

    print("\n  MAE = mean abs error across low/mid/high vs. truth (lower=better).")
    print("  wins = days this model was closest. Once a model consistently beats")
    print("  the median, weight the consensus toward it in get_forecast.py.")


# ======================== CLI ========================

def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "log"

    if cmd == "log":
        lat = float(argv[2]) if len(argv) > 2 else TLV_LAT
        lon = float(argv[3]) if len(argv) > 3 else TLV_LON
        dt_utc = get_today_sunset_utc(lat, lon)
        asyncio.run(cmd_log(Point(lat, lon), dt_utc))

    elif cmd == "observe":
        # observe DATE low mid high [lat lon]
        date_str = argv[2]
        low, mid, high = float(argv[3]), float(argv[4]), float(argv[5])
        lat = float(argv[6]) if len(argv) > 6 else TLV_LAT
        lon = float(argv[7]) if len(argv) > 7 else TLV_LON
        cmd_observe(date_str, low, mid, high, lat, lon)

    elif cmd == "backfill":
        asyncio.run(cmd_backfill())

    elif cmd == "report":
        cmd_report()

    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
