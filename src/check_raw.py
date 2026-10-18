import os
import argparse
import pandas as pd
from src.validate_raw import get_part1_columns, validate_parquet_file


def format_deleted_counts(series):
    """Format Deleted column value counts as a compact string."""
    if series.empty:
        return "empty"
    counts = series.value_counts(dropna=False).to_dict()
    items = [f"{str(k)}:{v}" for k, v in counts.items()]
    return ", ".join(items)


def calc_null_share(df, col):
    """Return share of null values for a column, or 1.0 if missing."""
    return float(df[col].isna().mean()) if col in df and len(df) > 0 else 1.0


def inspect_raw_parquet(file_path):
    """Inspect a single parquet file against quality and schema gates."""
    df = pd.read_parquet(file_path)
    race_id = os.path.splitext(os.path.basename(file_path))[0]
    schema_ok = set(get_part1_columns()).issubset(set(df.columns))
    is_valid, reason = validate_parquet_file(file_path)

    w_cols = ["TrackTemp", "AirTemp", "Humidity", "Rainfall"]
    weather_ok = all(c in df.columns and not df[c].isna().all() for c in w_cols)
    del_col = df["Deleted"] if "Deleted" in df else pd.Series(dtype=object)
    del_share = float((del_col == True).mean()) if len(del_col) > 0 else 0.0
    if del_col.isna().all():
        is_valid = False
        reason = "Deleted column entirely null" if reason == "valid" else f"{reason}, Deleted column entirely null"

    return {
        "race_id": race_id, "laps": len(df), "drivers": int(df["Driver"].nunique()) if "Driver" in df else 0,
        "schema_match": "yes" if schema_ok else "no", "null_lap_time": round(calc_null_share(df, "LapTime"), 4),
        "null_compound": round(calc_null_share(df, "Compound"), 4), "null_tyre_life": round(calc_null_share(df, "TyreLife"), 4),
        "null_track_status": round(calc_null_share(df, "TrackStatus"), 4), "weather_present": "yes" if weather_ok else "no",
        "deleted_share": round(del_share, 4), "deleted_counts": format_deleted_counts(del_col),
        "valid": "yes" if is_valid else "no", "reason": reason
    }


def check_all_raw_parquets(raw_dir="data/raw", reports_dir="reports"):
    """Audit all parquets in raw_dir and save results to raw_check.csv."""
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(reports_dir, "raw_check.csv")
    cols = [
        "race_id", "laps", "drivers", "schema_match", "null_lap_time",
        "null_compound", "null_tyre_life", "null_track_status",
        "weather_present", "deleted_share", "deleted_counts", "valid", "reason"
    ]
    if not os.path.exists(raw_dir) or not [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]:
        df_empty = pd.DataFrame(columns=cols)
        df_empty.to_csv(csv_path, index=False)
        return df_empty
    records = []
    for f in sorted([f for f in os.listdir(raw_dir) if f.endswith(".parquet")]):
        rec = inspect_raw_parquet(os.path.join(raw_dir, f))
        records.append(rec)
        print(f"{rec['race_id']}: laps={rec['laps']}, valid={rec['valid']}, deleted=[{rec['deleted_counts']}]")
    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_path, index=False)
    return df_out


def main():
    """CLI entry point for checking raw parquet files."""
    parser = argparse.ArgumentParser(description="Audit raw F1 parquet datasets")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw parquet directory")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Path to reports directory")
    args = parser.parse_args()
    check_all_raw_parquets(args.raw_dir, args.reports_dir)


if __name__ == "__main__":
    main()
