import os
import requests
from pathlib import Path
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import date
from typing import Iterable, Sequence, List, Union



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


def add_weather_lagged_aggregated_features(
    df_weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Create lagged and aggregated features for water-level prediction.

    Returns a single DataFrame aligned by date.
    """

    # --- Safety checks ---
    if not isinstance(df_weather.index, pd.DatetimeIndex):
        raise ValueError("df_weather index must be DatetimeIndex")

    df = df_weather.sort_index()

    # ======================
    # Precipitation features
    # ======================
    df["precip_sum_3d"] = (
        df["precipitation_sum"]
        .rolling(window=3, min_periods=3)
        .sum()
    )

    df["precip_sum_7d"] = (
        df["precipitation_sum"]
        .rolling(window=7, min_periods=7)
        .sum()
    )

    df["precip_sum_14d"] = (
        df["precipitation_sum"]
        .rolling(window=14, min_periods=14)
        .sum()
    )

    # ======================
    # Snowfall features
    # ======================
    df["snow_sum_14d"] = (
        df["snowfall_sum"]
        .rolling(window=14, min_periods=14)
        .sum()
    )

    df["snow_sum_30d"] = (
        df["snowfall_sum"]
        .rolling(window=30, min_periods=30)
        .sum()
    )

    df["snow_sum_60d"] = (
        df["snowfall_sum"]
        .rolling(window=60, min_periods=60)
        .sum()
    )

    return df


def normalize_date_to_utc_day(
    df: pd.DataFrame,
    col: str = "date",
    assume_tz: str = "UTC",
) -> pd.DataFrame:
    """
    Normalize a datetime column to UTC and floor to day ('D').

    - If the column is timezone-aware, it is converted to UTC.
    - If the column is timezone-naive, it is localized to `assume_tz` first
      (defaults to UTC), then converted to UTC.
    - The resulting timestamps are floored to midnight (00:00:00) in UTC.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (modified in place).
    col : str, default "date"
        Name of the datetime column to normalize.
    assume_tz : str, default "UTC"
        Timezone to assume for timezone-naive timestamps (e.g. "Europe/Stockholm").

    Returns
    -------
    pd.DataFrame
        The same DataFrame with `df[col]` normalized to UTC day boundaries.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    s = pd.to_datetime(df[col], errors="raise")

    # tz-aware vs tz-naive handling
    if s.dt.tz is None:
        # Localize naive timestamps (do NOT convert; convert comes after)
        s = s.dt.tz_localize(assume_tz)
    else:
        # Already tz-aware; keep it as is
        pass

    df[col] = s.dt.tz_convert("UTC").dt.floor("D")
    return df


def fetch_daily_weather(
    latitude: float,
    longitude: float,
    features: Sequence[str],
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """
    Fetch daily historical weather data from the Open-Meteo Archive API.

    This function queries the Open-Meteo archive endpoint for the given latitude/longitude
    and returns a pandas DataFrame indexed by time (daily frequency). The requested
    weather variables are specified via the `features` argument.

    Parameters
    ----------
    latitude : float
        Latitude of the location.
    longitude : float
        Longitude of the location.
    features : Sequence[str]
        List of Open-Meteo daily variable names to request (e.g. "precipitation_sum",
        "temperature_2m_mean", "wind_speed_10m_mean"). The values must be valid
        daily variables supported by the Open-Meteo archive endpoint.
    start_date : str, default "2020-01-01"
        Start date (inclusive) in ISO format "YYYY-MM-DD".
    end_date : str | None, default None
        End date (inclusive) in ISO format "YYYY-MM-DD". If None, defaults to today's date.
    timezone : str, default "UTC"
        Timezone for the returned timestamps (Open-Meteo `timezone` parameter),
        e.g. "UTC" or "Europe/Stockholm".

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by a DatetimeIndex named "time", sorted ascending.
        Columns correspond to the requested `features` plus the time column converted
        to the index.

    Raises
    ------
    ValueError
        If `features` is empty.
    requests.HTTPError
        If the Open-Meteo API request fails (non-2xx response).
    KeyError
        If the API response does not contain the expected "daily" payload.

    Examples
    --------
    df = fetch_daily_weather(
    ...     59.3293, 18.0686,
    ...     features=["precipitation_sum", "temperature_2m_mean"],
    ...     start_date="2024-01-01",
    ...     end_date="2024-02-01",
    ...     timezone="Europe/Stockholm",
    ... )
    df.head()
    """
    if not features:
        raise ValueError("`features` must contain at least one daily variable name.")

    if end_date is None:
        end_date = date.today().isoformat()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(features),
        "timezone": timezone,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    daily = response.json()["daily"]
    df = pd.DataFrame(daily)

    # Open-Meteo uses "time" for daily timestamps
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # Keep a stable column order: only requested features, in the order provided
    cols = [f for f in features if f in df.columns]
    return df[cols]


def fetch_daily_weather_forecast(
    latitude: float,
    longitude: float,
    features: Sequence[str],
    forecast_days: int = 14,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """
    Fetch daily weather *forecast* data from the Open-Meteo Forecast API.

    This function queries the Open-Meteo forecast endpoint for the given latitude/longitude
    and returns a pandas DataFrame indexed by daily timestamps. The requested forecast
    variables are specified via the `features` argument.

    Parameters
    ----------
    latitude : float
        Latitude of the location.
    longitude : float
        Longitude of the location.
    features : Sequence[str]
        List of Open-Meteo daily variable names to request (e.g. "precipitation_sum",
        "temperature_2m_mean", "wind_speed_10m_mean"). The values must be valid
        daily variables supported by the Open-Meteo forecast endpoint.
    forecast_days : int, default 14
        Number of days ahead to fetch (Open-Meteo `forecast_days` parameter).
    timezone : str, default "UTC"
        Timezone for the returned timestamps (Open-Meteo `timezone` parameter),
        e.g. "UTC" or "Europe/Stockholm".

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by a DatetimeIndex named "time", sorted ascending.
        Columns correspond to the requested `features` (in the order provided).

    Raises
    ------
    ValueError
        If `features` is empty, or the API response does not contain the expected payload.
    requests.HTTPError
        If the Open-Meteo API request fails (non-2xx response).

    Examples
    --------
    df = fetch_daily_weather_forecast(
    ...     59.3293, 18.0686,
    ...     features=["precipitation_sum", "temperature_2m_mean"],
    ...     forecast_days=7,
    ...     timezone="Europe/Stockholm",
    ... )
    df.tail()
    """
    if not features:
        raise ValueError("`features` must contain at least one daily variable name.")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(features),
        "forecast_days": int(forecast_days),
        "timezone": timezone,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    payload = r.json()
    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise ValueError(
            f"Unexpected forecast response format: expected payload['daily']['time'] "
            f"but got keys={list(payload.keys())}"
        )

    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # Keep stable column order: only requested features, in the order provided
    cols = [f for f in features if f in df.columns]
    return df[cols]


def find_missing_dates(
    df: pd.DataFrame,
    freq: str = "D",
    verbose: bool = True
) -> pd.DatetimeIndex:
    """
    Return the timestamps missing from a time-indexed DataFrame.

    This utility assumes the DataFrame index should form a complete, regular
    `pandas.date_range` between the minimum and maximum timestamp in the data.
    It constructs that full expected range at the given frequency and returns
    the timestamps that are absent from `df.index`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must have a `pd.DatetimeIndex`.
    freq : str, default "D"
        Expected frequency of the index. Common values include:
        - "D"  : daily
        - "H"  : hourly
        - "15min" : 15-minute intervals
        Any frequency string accepted by `pd.date_range` is valid.
    verbose : bool, default True
        If True, prints a short summary and (if any) the missing dates.

    Returns
    -------
    pd.DatetimeIndex
        A DatetimeIndex containing the missing timestamps (sorted ascending).
        If no timestamps are missing, an empty DatetimeIndex is returned.

    Raises
    ------
    ValueError
        If `df.index` is not a `pd.DatetimeIndex`.

    Notes
    -----
    - This checks for *missing* timestamps within the min/max range only.
      It does not detect duplicates, irregular spacing, or gaps outside
      the observed range.
    - If your index is timezone-aware, the returned timestamps will carry
      the same timezone.

    Examples
    --------
     missing = find_missing_dates(df, freq="D", verbose=False)
     if len(missing) > 0:
         print(missing[:5])
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )

    missing_dates = full_range.difference(df.index)

    if verbose:
        if len(missing_dates) == 0:
            print("✅ No missing dates found.")
        else:
            print(f"⚠️ Missing {len(missing_dates)} dates:")
            for d in missing_dates:
                print(d.date())

    return missing_dates


def offset_coordinates(lat: float, lon: float, distance_km: float):
    """
    Compute lat/lon offsets for N, S, E, W directions.
    """
    lat_rad = math.radians(lat)

    dlat = distance_km / 111.0
    dlon = distance_km / (111.0 * math.cos(lat_rad))

    return {
        "n": (lat + dlat, lon),
        "s": (lat - dlat, lon),
        "e": (lat, lon + dlon),
        "w": (lat, lon - dlon),
    }


def fetch_spatial_weather_75km(
    latitude: float,
    longitude: float,
    features: Sequence[str],
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """
    Fetch daily historical weather features for four spatial offsets (N/S/E/W) at ~75 km.

    This function computes four coordinates located approximately 75 km to the
    North, South, East, and West of the given (latitude, longitude) using
    `offset_coordinates(...)`. For each offset point it fetches daily historical
    weather data via `fetch_daily_weather(...)`, renames the columns to include
    the direction and distance suffix, and concatenates the results into a
    single DataFrame aligned on the time index.

    Parameters
    ----------
    latitude : float
        Latitude of the sensor / reference location.
    longitude : float
        Longitude of the sensor / reference location.
    features : Sequence[str]
        Daily variables to request from Open-Meteo (e.g. "precipitation_sum",
        "temperature_2m_mean"). These are passed through to `fetch_daily_weather`.
    start_date : str, default "2020-01-01"
        Start date (inclusive) for the historical query in ISO format "YYYY-MM-DD".
    end_date : str | None, default None
        End date (inclusive) in ISO format "YYYY-MM-DD". If None, `fetch_daily_weather`
        will default it to today's date.
    timezone : str, default "UTC"
        Timezone for returned timestamps passed through to `fetch_daily_weather`.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by time (DatetimeIndex), containing the requested weather
        features for all four offset directions. Columns are suffixed with
        `_{direction}_75km`, e.g. `precipitation_sum_n_75km`. The index is sorted
        ascending.

    Raises
    ------
    ValueError
        If `features` is empty.

    Notes
    -----
    - Assumes `offset_coordinates(latitude, longitude, distance_km=75)` returns a
      mapping like {"n": (lat, lon), "s": (...), "e": (...), "w": (...)}.
    - The final DataFrame is formed with `pd.concat(..., axis=1)` and aligned on the
      DatetimeIndex. Missing dates in any direction will result in NaNs.

    Examples
    --------
    >>> vars = ["precipitation_sum", "temperature_2m_mean"]
    >>> df_75 = fetch_spatial_weather_75km(59.3293, 18.0686, vars, start_date="2024-01-01")
    """
    if not features:
        raise ValueError("`features` must contain at least one daily variable name.")

    offsets = offset_coordinates(latitude, longitude, distance_km=75)
    dfs: list[pd.DataFrame] = []

    for direction, (lat, lon) in offsets.items():
        df = fetch_daily_weather(
            lat,
            lon,
            features=features,
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
        )

        # Rename each requested feature to include direction + distance suffix
        rename_map = {feat: f"{feat}_{direction}_75km" for feat in features}
        df = df.rename(columns=rename_map)

        dfs.append(df)

    return pd.concat(dfs, axis=1).sort_index()


def fetch_spatial_weather_forecast_75km(
    latitude: float,
    longitude: float,
    features: Sequence[str],
    forecast_days: int = 14,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """
    Fetch daily weather *forecast* features for four spatial offsets (N/S/E/W) at ~75 km.

    This function computes four coordinates located approximately 75 km to the
    North, South, East, and West of the given (latitude, longitude) using
    `offset_coordinates(...)`. For each offset point it fetches daily forecast
    weather variables via `fetch_daily_weather_forecast_spacial(...)`, renames
    the returned columns to include the direction and distance suffix, and
    concatenates the results into a single DataFrame aligned on the time index.

    Parameters
    ----------
    latitude : float
        Latitude of the sensor / reference location.
    longitude : float
        Longitude of the sensor / reference location.
    features : Sequence[str]
        Daily forecast variables to request from Open-Meteo (e.g. "precipitation_sum",
        "temperature_2m_mean", "wind_speed_10m_mean"). Values must be valid daily
        variables supported by the forecast endpoint.
    forecast_days : int, default 14
        Number of days ahead to fetch (Open-Meteo `forecast_days` parameter).
    timezone : str, default "UTC"
        Timezone for returned timestamps (Open-Meteo `timezone` parameter).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by time (DatetimeIndex), sorted ascending, containing
        the requested weather variables for each direction. Columns are suffixed
        with `_{direction}_75km`, for example:
        - precipitation_sum_n_75km
        - temperature_2m_mean_e_75km
        - wind_speed_10m_mean_w_75km
        etc.

    Raises
    ------
    ValueError
        If `features` is empty.
    Any exception raised by `offset_coordinates` or `fetch_daily_weather_forecast_spacial`.

    Notes
    -----
    - Assumes `offset_coordinates(latitude, longitude, distance_km=75)` returns a mapping
      like {"n": (lat, lon), "s": (...), "e": (...), "w": (...)}.
    - The final DataFrame is formed with `pd.concat(..., axis=1)` and aligned on the
      DatetimeIndex. Missing dates in any direction will result in NaNs.

    Examples
    --------
    >>> vars = ["precipitation_sum", "temperature_2m_mean"]
    >>> df_75 = fetch_spatial_weather_forecast_75km(59.3293, 18.0686, vars, forecast_days=7)
    >>> df_75.head()
    """
    if not features:
        raise ValueError("`features` must contain at least one daily variable name.")

    offsets = offset_coordinates(latitude, longitude, distance_km=75)
    dfs: list[pd.DataFrame] = []

    for direction, (lat, lon) in offsets.items():
        df = fetch_daily_weather_forecast(
            lat,
            lon,
            features=features,
            forecast_days=forecast_days,
            timezone=timezone,
        )

        # Rename each requested feature to include direction + distance suffix
        rename_map = {feat: f"{feat}_{direction}_75km" for feat in features}
        df = df.rename(columns=rename_map)

        dfs.append(df)

    return pd.concat(dfs, axis=1).sort_index()


def get_water_level_features() -> List[str]:
    """
    Return the list of water-level feature names used by the model/pipeline.
    """
    return [
        "water_level_cm",
        "water_level_cm_t_1",
        "water_level_cm_t_3",
        "water_level_cm_t_7",
        "water_level_cm_t_14",
    ]


def get_weather_features() -> List[str]:
    """
    Return the list of weather feature names used by the model/pipeline.
    """
    weather_features = [
        # Local
        "precipitation_sum",
        "snowfall_sum",
        "rain_sum",
        "temperature_2m_mean",
        "wind_speed_10m_mean",
        "surface_pressure_mean",

        # Aggregated
        "precip_sum_3d",
        "precip_sum_7d",
        "precip_sum_14d",
        "snow_sum_14d",
        "snow_sum_30d",
        "snow_sum_60d",
    ]

    # Spatial (75 km)
    for d in ["n", "s", "e", "w"]:
        weather_features.extend([
            f"precipitation_sum_{d}_75km",
            f"snowfall_sum_{d}_75km",
            f"rain_sum_{d}_75km",
            f"temperature_2m_mean_{d}_75km",
            f"wind_speed_10m_mean_{d}_75km",
            f"surface_pressure_mean_{d}_75km",
        ])
    return weather_features


def plot_actual_vs_predicted(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    output_filename: str,
    date_col: str = "date",
    actual_col: str = "water_level_cm",
    pred_col: str = "predicted_water_level_cm",
    title: str = "Actual vs Predicted Water Level",
    month_tick_interval: int = 2,
    show: bool = True,
) -> str:
    """
    Plot actual vs. predicted values over time and save the figure to disk.

    The function:
    - selects `date_col`, `actual_col`, and `pred_col` from `df`
    - converts the date column to timezone-aware UTC datetimes
    - sorts the data chronologically
    - plots actual as a solid line and predicted as a dashed line
    - formats the x-axis ticks as year-month at a configurable interval
    - saves the plot to `output_dir/output_filename`

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing date, actual values, and predicted values.
    output_dir : str | pathlib.Path
        Directory where the plot image will be saved. Created if it does not exist.
    output_filename : str
        Output filename (e.g., "actual_vs_pred.png"). Should include an image extension.
    date_col : str, default "date"
        Name of the datetime column.
    actual_col : str, default "water_level_cm"
        Name of the column containing ground-truth values.
    pred_col : str, default "predicted_water_level_cm"
        Name of the column containing model predictions.
    title : str, default "Actual vs Predicted Water Level"
        Plot title.
    month_tick_interval : int, default 2
        Interval for month ticks on the x-axis (e.g., 2 -> every 2 months).
    show : bool, default True
        If True, displays the plot. If False, only saves it.

    Returns
    -------
    str
        The full path to the saved plot file.

    Raises
    ------
    KeyError
        If any of `date_col`, `actual_col`, or `pred_col` is missing from `df`.
    """
    # Validate columns
    missing = [c for c in (date_col, actual_col, pred_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Prepare data
    df_plot = df[[date_col, actual_col, pred_col]].copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col], utc=True, errors="raise")
    df_plot = df_plot.sort_values(date_col)

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_plot[date_col], df_plot[actual_col], label="Actual", linewidth=2)
    ax.plot(df_plot[date_col], df_plot[pred_col], label="Predicted", linewidth=2, linestyle="--")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_tick_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)

    ax.set_xlabel("Date")
    ax.set_ylabel("Water Level (cm)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(output_path)


def get_model_name(sensor_name: str, sensor_id: Union[str, int]) -> str:
    return f"model_{sensor_name}_{sensor_id}"



