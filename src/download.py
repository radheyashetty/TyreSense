import os
import yaml
import pandas as pd
import fastf1


def load_config(config_path="config.yaml"):
    """Read project configuration from yaml file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def enable_f1_cache(cache_dir):
    """Enable FastF1 local disk cache to avoid redundant network calls."""
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def get_race_schedule(year):
    """Fetch the race schedule for a specific season year excluding testing."""
    schedule = fastf1.get_event_schedule(year)
    return schedule[schedule["EventFormat"] != "testing"]


def extract_race_data(session):
    """Extract raw laps and weather features from a loaded race session."""
    laps = session.laps
    if laps is None or len(laps) == 0:
        return None
    weather = laps.get_weather_data()
    dataframe = laps.copy()
    dataframe["TrackTemp"] = weather["TrackTemp"].values
    dataframe["AirTemp"] = weather["AirTemp"].values
    dataframe["Humidity"] = weather["Humidity"].values
    dataframe["Rainfall"] = weather["Rainfall"].values
    dataframe["EventName"] = session.event["EventName"]
    dataframe["Location"] = session.event["Location"]
    dataframe["RoundNumber"] = session.event["RoundNumber"]
    dataframe["Year"] = session.event.year
    return dataframe


def save_race_parquet(dataframe, raw_dir, year, round_number):
    """Save race dataframe to a parquet file in raw data directory."""
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, f"race_{year}_{round_number:02d}.parquet")
    dataframe.to_parquet(file_path, index=False)
    print(f"Saved {file_path} with {len(dataframe)} laps.")
    return file_path


def download_single_race(year, round_number, raw_dir):
    """Download and save a single race session to parquet."""
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=True, messages=False)
    dataframe = extract_race_data(session)
    if dataframe is not None:
        return save_race_parquet(dataframe, raw_dir, year, round_number)
    return None


def load_cached_race(raw_dir, year, round_number):
    """Load a saved race parquet file from disk without network access."""
    file_path = os.path.join(raw_dir, f"race_{year}_{round_number:02d}.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing cached race: {file_path}")
    return pd.read_parquet(file_path)


def download_season(year, raw_dir):
    """Download all races for a single season year."""
    schedule = get_race_schedule(year)
    for round_number in schedule["RoundNumber"]:
        try:
            download_single_race(year, int(round_number), raw_dir)
        except Exception as error:
            print(f"Failed {year} round {round_number}: {error}")
