import pandas as pd
from tqdm import tqdm
from typing import Optional
from datetime import timezone
from dateutil import parser as dateparser

from astral import Observer
from astral.sun import elevation, azimuth


EXCEL_PATH = "sunset_project_schema_with_flickr_cols.xlsx"
SHEET_NAME = 0


def _ensure_utc_dt(x):
    dt = dateparser.isoparse(str(x))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_solar(lat: float, lon: float, dt_utc) -> (Optional[float], Optional[float]):
    """
    Returns (solar_elevation_deg, solar_azimuth_deg)
    """
    try:
        obs = Observer(latitude=lat, longitude=lon)
        elev = float(elevation(observer=obs, dateandtime=dt_utc))
        azi = float(azimuth(observer=obs, dateandtime=dt_utc))
        return elev, azi
    except Exception:
        return None, None


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    required = ["datetime", "lat", "lon"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError("Missing required column '{}'".format(c))

    for i in tqdm(range(len(df)), desc="Filling solar angles"):
        dt_raw = df.at[i, "datetime"]
        lat = df.at[i, "lat"]
        lon = df.at[i, "lon"]

        if pd.isna(dt_raw) or pd.isna(lat) or pd.isna(lon):
            continue

        dt_utc = _ensure_utc_dt(dt_raw)
        elev, azi = compute_solar(float(lat), float(lon), dt_utc)

        if "solar_elevation" in df.columns:
            df.at[i, "solar_elevation"] = elev
        if "solar_azimuth" in df.columns:
            df.at[i, "solar_azimuth"] = azi

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, index=False, sheet_name=writer.book.sheetnames[SHEET_NAME])

    print("Done. Updated solar_elevation and solar_azimuth in: {}".format(EXCEL_PATH))


if __name__ == "__main__":
    main()