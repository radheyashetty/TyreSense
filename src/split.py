import os
import pandas as pd
from src.download import load_config


def extract_season_from_race_id(race_id):
    """Extract integer season year from race_id formatted as YYYY_RR."""
    return int(str(race_id).split("_")[0])


def split_by_race(dataframe, config):
    """Split dataframe into train, validation, and test splits strictly by race season."""
    seasons = config["seasons"]
    train_years = seasons["train"]
    val_years = seasons["val"]
    test_years = seasons["test"]

    years = dataframe["race_id"].apply(extract_season_from_race_id)
    train_df = dataframe[years.isin(train_years)].copy()
    val_df = dataframe[years.isin(val_years)].copy()
    test_df = dataframe[years.isin(test_years)].copy()
    return train_df, val_df, test_df


def fit_train_scaler(train_df, feature_cols):
    """Compute mean and standard deviation strictly on training races."""
    scaler_params = {}
    for col in feature_cols:
        col_mean = float(train_df[col].mean())
        col_std = float(train_df[col].std())
        if col_std == 0.0 or pd.isna(col_std):
            col_std = 1.0
        scaler_params[col] = {"mean": col_mean, "std": col_std}
    return scaler_params


def apply_scaler(dataframe, scaler_params):
    """Standardise dataframe features using pre-fitted parameters."""
    df_scaled = dataframe.copy()
    for col, params in scaler_params.items():
        df_scaled[col] = (df_scaled[col] - params["mean"]) / params["std"]
    return df_scaled


def build_track_mapping(train_df):
    """Build track to index vocabulary from training races, reserving 0 for unk."""
    unique_tracks = sorted(train_df["track_id"].unique())
    mapping = {"unk": 0}
    for idx, track in enumerate(unique_tracks, start=1):
        mapping[track] = idx
    return mapping


def encode_tracks(dataframe, track_mapping):
    """Map track IDs to integer codes, replacing unseen tracks with unk index 0."""
    return pd.Series([track_mapping.get(t, 0) for t in dataframe["track_id"]], index=dataframe.index)


def save_splits(train_df, val_df, test_df, processed_dir):
    """Save split dataframes to parquet files in processed directory."""
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(processed_dir, "train_laps.parquet"), index=False)
    val_df.to_parquet(os.path.join(processed_dir, "val_laps.parquet"), index=False)
    test_df.to_parquet(os.path.join(processed_dir, "test_laps.parquet"), index=False)
