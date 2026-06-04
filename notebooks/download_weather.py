import os
from pathlib import Path
from dotenv import load_dotenv
import cdsapi

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

RAW_DATA_DIR = (
    BASE_DIR /
    "data" /
    "raw"
)

ARCHIVE_DIR = (
    BASE_DIR /
    "data" /
    "archives"
)

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ARCHIVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# CDS API CONFIGURATION
# ==========================================

CDS_URL = os.getenv("CDS_URL")
CDS_KEY = os.getenv("CDS_API_KEY")


# ==========================================
# DATA CONFIGURATION
# ==========================================

YEAR = "2025"

AREA = [
    26.32,  # North
    82.40,  # West
    25.38,  # South
    83.52   # East
]

VARIABLES = [

    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "surface_pressure",

    "10m_u_component_of_wind",
    "10m_v_component_of_wind",

    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover"
]

DAYS = [
    f"{day:02d}"
    for day in range(1, 32)
]

TIMES = [
    f"{hour:02d}:00"
    for hour in range(24)
]

MONTH_GROUPS = {

    "JAN": ["01"],
    "FEB": ["02"],
    "MAR": ["03"],
    "APR": ["04"],
    "MAY": ["05"],
    "JUN": ["06"],
    "JUL": ["07"],
    "AUG": ["08"],
    "SEP": ["09"],
    "OCT": ["10"],
    "NOV": ["11"],
    "DEC": ["12"],
}


# ==========================================
# CDS CLIENT
# ==========================================

client = cdsapi.Client(
    url=CDS_URL,
    key=CDS_KEY
)


# ==========================================
# DOWNLOAD FUNCTION
# ==========================================

def download_quarterly_data(
    month_name: str,
    month: list[str]
) -> None:

    output_file = (
        ARCHIVE_DIR /
        f"azamgarh_weather_{month_name}_{YEAR}.zip"
    )

    print("\n" + "=" * 50)
    print(f"Downloading {month_name}")
    print("=" * 50)

    request = {

        "product_type": "reanalysis",

        "variable": VARIABLES,

        "year": YEAR,

        "month": month,

        "day": DAYS,

        "time": TIMES,

        "area": AREA,

        "format": "netcdf"
    }

    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(output_file)
    )

    print(f"\n{month_name} download completed.")
    print(f"Saved to: {output_file}")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    for month_name, month in MONTH_GROUPS.items():

        try:

            download_quarterly_data(
                month_name=month_name,
                month=month
            )

        except Exception as error:

            print(
                f"\nError downloading {month_name}"
            )

            print(error)