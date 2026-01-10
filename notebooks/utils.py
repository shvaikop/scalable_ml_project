import requests
import pandas as pd
from typing import Iterable



def fetch_smhi_water_level(
    api_url
) -> pd.DataFrame:
    """
    Fetch historical water level data from SMHI and return
    a DataFrame with columns:
      - date (datetime64[ns], UTC)
      - water_level_cm (float)
    """

    url = api_url
    # url = (
    #     f"https://opendata-download-hydroobs.smhi.se/api/version/1.0/"
    #     f"parameter/{parameter_id}/station/{station_id}/period/{period}/data.json"
    # )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data = response.json()

    values = data["value"]

    df = pd.DataFrame(values)

    # Convert unix milliseconds to datetime
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)

    # Rename and keep only what we need
    df = df.rename(columns={"value": "water_level_cm"})

    df = df[["date", "water_level_cm"]]

    # Sort just to be safe
    df = df.sort_values("date").reset_index(drop=True)

    return df


def add_water_level_lags(
    df: pd.DataFrame,
    value_col: str = "water_level_cm",
    date_col: str = "date",
    lags: Iterable[int] = (1, 3, 7, 14),
) -> pd.DataFrame:
    """
    Add lagged water level features to a daily time-series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing date and water level columns.
    value_col : str
        Name of the water level column.
    date_col : str
        Name of the datetime column.
    lags : Iterable[int]
        Lags (in days) to compute.

    Returns
    -------
    pd.DataFrame
        DataFrame with added lagged features.
    """

    # --- Safety checks ---
    if date_col not in df.columns:
        raise ValueError(f"Missing required column: {date_col}")
    if value_col not in df.columns:
        raise ValueError(f"Missing required column: {value_col}")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"{date_col} must be datetime dtype")

    # --- Sort to ensure correct lagging ---
    df = df.sort_values(date_col).copy()

    # --- Add lag features ---
    for lag in lags:
        df[f"{value_col}_t_{lag}"] = df[value_col].shift(lag)

    return df


def normalize_date_to_utc_day(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """
    Convert a datetime column to UTC and floor to day ("D").
    Works for tz-aware datetimes. Returns the same DataFrame (mutates in place).
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    s = pd.to_datetime(df[col], errors="raise")

    if s.dt.tz is None:
        raise ValueError(
            f"Column '{col}' is timezone-naive. Localize first (e.g. s.dt.tz_localize('UTC') "
            "or your local timezone) before converting."
        )

    df[col] = s.dt.tz_convert("UTC").dt.floor("D")
    return df