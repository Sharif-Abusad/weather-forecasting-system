"""
pipeline/preprocess.py
━━━━━━━━━━━━━━━━━━━━━━
Cleans and preprocesses the raw merged ERA5 CSV.

Input  : data/processed/azamgarh_weather_raw.csv  (output of weather_dataset_pipeline.ipynb)
Output : data/processed/azamgarh_weather_clean.csv

Steps:
    1. Load raw CSV and parse datetime
    2. Rename ERA5 variable names to model-friendly names
    3. Unit conversions (Kelvin → °C, metres → mm, fraction → %)
    4. Sort by valid_time and reset index
    5. Handle missing values (linear interpolation)
    6. Remove physical impossibilities (value range checks)
    7. Drop duplicate timestamps
    8. Compute wind speed from u/v components if needed
    9. Save cleaned CSV
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Add project root to path so app.config is importable ─────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import DATA_PROCESSED_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = os.path.join(DATA_PROCESSED_DIR, "azamgarh_weather_raw.csv")
OUTPUT_PATH = os.path.join(DATA_PROCESSED_DIR, "azamgarh_weather_clean.csv")

# ── Physical validity ranges ──────────────────────────────────────────────────
VALID_RANGES = {
    "temperature"       : (-20,  60),   # °C
    "surface_pressure"  : (850, 1080),  # hPa
    "total_cloud_cover" : (0,   1),
    "low_cloud_cover"   : (0,   1),
    "medium_cloud_cover": (0,   1),
    "high_cloud_cover"  : (0,   1),
    "precipitation"     : (0,   300),   # mm
    "humidity"          : (0,   100),   # %
    "wind_speed"        : (0,   60),    # m/s
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def kelvin_to_celsius(k: pd.Series) -> pd.Series:
    return (k - 273.15).round(4)


def metres_to_mm(m: pd.Series) -> pd.Series:
    """ERA5 precipitation is in metres per hour — convert to mm."""
    return (m * 1000).round(4)


def dewpoint_to_humidity(t_c: pd.Series, td_c: pd.Series) -> pd.Series:
    """
    Approximate relative humidity from temperature and dewpoint (both °C).
    Formula: RH ≈ 100 × exp(17.625 × Td / (243.04 + Td)) /
                          exp(17.625 × T  / (243.04 + T))
    """
    numerator   = np.exp(17.625 * td_c  / (243.04 + td_c))
    denominator = np.exp(17.625 * t_c   / (243.04 + t_c))
    rh = 100 * numerator / denominator
    return rh.clip(0, 100).round(2)


def compute_wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    """Wind speed from u (east) and v (north) components."""
    return np.sqrt(u**2 + v**2).round(4)


# ── Column rename map (ERA5 names → model names) ─────────────────────────────
RENAME_MAP = {
    "t2m"   : "temperature_raw",      # 2m temperature in Kelvin
    "sp"    : "surface_pressure_raw", # surface pressure in Pa
    "tcc"   : "total_cloud_cover",
    "lcc"   : "low_cloud_cover",
    "mcc"   : "medium_cloud_cover",
    "hcc"   : "high_cloud_cover",
    "tp"    : "precipitation_raw",    # total precip in metres
    "d2m"   : "dewpoint_raw",         # 2m dewpoint in Kelvin
    "u10"   : "u_wind",               # u-component at 10m
    "v10"   : "v_wind",               # v-component at 10m
    # Some pipelines already output renamed columns — handle both:
    "2m_temperature"          : "temperature_raw",
    "surface_pressure"        : "surface_pressure_raw",
    "total_precipitation"     : "precipitation_raw",
    "2m_dewpoint_temperature" : "dewpoint_raw",
    "10m_u_component_of_wind" : "u_wind",
    "10m_v_component_of_wind" : "v_wind",
}


def load_and_rename(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()
    # Rename ERA5 raw names to intermediate names
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    print(f"  Loaded  : {len(df):,} rows · {df.shape[1]} columns")
    return df


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    time_col = next((c for c in df.columns if "time" in c or "date" in c), None)
    if time_col is None:
        raise ValueError("No datetime column found. Expected 'valid_time' or 'time'.")
    df["valid_time"] = pd.to_datetime(df[time_col], utc=True)
    # Convert to IST (UTC+5:30)
    df["valid_time"] = df["valid_time"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    if time_col != "valid_time":
        df = df.drop(columns=[time_col])
    print(f"  Date range: {df['valid_time'].min()} → {df['valid_time'].max()}")
    return df


def unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    # Temperature: Kelvin → Celsius
    if "temperature_raw" in df.columns:
        df["temperature"] = kelvin_to_celsius(df["temperature_raw"])
        df = df.drop(columns=["temperature_raw"])

    # Pressure: Pa → hPa
    if "surface_pressure_raw" in df.columns:
        df["surface_pressure"] = (df["surface_pressure_raw"] / 100).round(2)
        df = df.drop(columns=["surface_pressure_raw"])

    # Precipitation: metres → mm (clip negatives from floating point)
    if "precipitation_raw" in df.columns:
        df["precipitation"] = metres_to_mm(df["precipitation_raw"]).clip(lower=0)
        df = df.drop(columns=["precipitation_raw"])

    # Humidity from dewpoint
    if "dewpoint_raw" in df.columns and "temperature" in df.columns:
        td_c = kelvin_to_celsius(df["dewpoint_raw"])
        df["humidity"] = dewpoint_to_humidity(df["temperature"], td_c)
        df = df.drop(columns=["dewpoint_raw"])

    # Wind speed from components
    if "u_wind" in df.columns and "v_wind" in df.columns:
        df["wind_speed"] = compute_wind_speed(df["u_wind"], df["v_wind"])
        df = df.drop(columns=["u_wind", "v_wind"])

    print("  Unit conversions complete")
    return df


def sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values("valid_time").drop_duplicates(
        subset="valid_time", keep="first"
    ).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} duplicate timestamps")
    else:
        print("  No duplicate timestamps")
    return df


def check_temporal_continuity(df: pd.DataFrame) -> None:
    diffs  = df["valid_time"].diff().dropna()
    expected = pd.Timedelta("1h")
    gaps   = diffs[diffs != expected]
    if len(gaps) == 0:
        print("  Temporal continuity: ✅ perfectly continuous (hourly)")
    else:
        print(f"  ⚠️  {len(gaps)} non-1h gaps detected:")
        print(gaps.value_counts().head(5))


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    null_counts  = df[numeric_cols].isnull().sum()
    total_nulls  = null_counts.sum()

    if total_nulls == 0:
        print("  Missing values: ✅ none")
        return df

    print(f"  Missing values: {total_nulls} total — applying linear interpolation")
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    # Final forward/backward fill for any remaining edge NaNs
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def range_check(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        violations = ((df[col] < lo) | (df[col] > hi)).sum()
        if violations > 0:
            print(f"  ⚠️  {col}: {violations} out-of-range values → clipping to [{lo}, {hi}]")
            df[col] = df[col].clip(lower=lo, upper=hi)
    print("  Range check complete")
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "valid_time", "temperature", "surface_pressure",
        "total_cloud_cover", "low_cloud_cover", "medium_cloud_cover",
        "high_cloud_cover", "precipitation", "humidity", "wind_speed",
    ]
    # Keep only columns that exist
    final = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️  Columns not found (skipping): {missing}")
    return df[final]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  PREPROCESS.PY — ERA5 Data Cleaning")
    print("=" * 55)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    print("\n[1/8] Loading raw CSV...")
    df = load_and_rename(INPUT_PATH)

    print("\n[2/8] Parsing datetime...")
    df = parse_datetime(df)

    print("\n[3/8] Applying unit conversions...")
    df = unit_conversions(df)

    print("\n[4/8] Sorting and deduplicating...")
    df = sort_and_deduplicate(df)

    print("\n[5/8] Checking temporal continuity...")
    check_temporal_continuity(df)

    print("\n[6/8] Handling missing values...")
    df = handle_missing(df)

    print("\n[7/8] Checking value ranges...")
    df = range_check(df)

    print("\n[8/8] Selecting final columns...")
    df = select_final_columns(df)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 55}")
    print(f"  ✅ Saved → {OUTPUT_PATH}")
    print(f"     Rows    : {len(df):,}")
    print(f"     Columns : {list(df.columns)}")
    print(f"     Date    : {df['valid_time'].min()} → {df['valid_time'].max()}")
    print(f"{'=' * 55}")

    return df


if __name__ == "__main__":
    main()
