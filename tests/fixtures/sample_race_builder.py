import pandas as pd
import numpy as np


def build_raw_row(season, round_num, event_name, drv, drv_idx, lnum, is_pit_in, is_pit_out, curr_tl, curr_comp, stint):
    """Build single dictionary record matching canonical 39-column schema."""
    lt_sec = 90.0 + (lnum * 0.05) + (drv_idx * 0.1)
    pin = pd.Timedelta(seconds=lnum * 90.0 + 40.0) if is_pit_in else pd.NaT
    pout = pd.Timedelta(seconds=lnum * 90.0 - 50.0) if is_pit_out else pd.NaT
    return {
        "Time": pd.Timedelta(seconds=lnum * 90.0), "Driver": drv, "DriverNumber": str(drv_idx + 1),
        "LapTime": pd.Timedelta(seconds=lt_sec), "LapNumber": float(lnum), "Stint": stint,
        "PitOutTime": pout, "PitInTime": pin, "Sector1Time": pd.Timedelta(seconds=30.0),
        "Sector2Time": pd.Timedelta(seconds=30.0), "Sector3Time": pd.Timedelta(seconds=30.0),
        "Sector1SessionTime": pd.Timedelta(seconds=lnum * 90.0 - 60.0),
        "Sector2SessionTime": pd.Timedelta(seconds=lnum * 90.0 - 30.0),
        "Sector3SessionTime": pd.Timedelta(seconds=lnum * 90.0),
        "SpeedI1": 290.0, "SpeedI2": 280.0, "SpeedFL": 295.0, "SpeedST": 310.0,
        "IsPersonalBest": False, "Compound": curr_comp, "TyreLife": curr_tl,
        "FreshTyre": True, "Team": f"Team_{drv_idx % 10}", "LapStartTime": pd.Timedelta(seconds=(lnum - 1) * 90.0),
        "LapStartDate": pd.Timestamp(f"{season}-03-05 15:00:00") + pd.Timedelta(seconds=(lnum - 1) * 90.0),
        "TrackStatus": "1", "Position": float(drv_idx + 1), "Deleted": False, "DeletedReason": "",
        "FastF1Generated": False, "IsAccurate": np.nan, "TrackTemp": 32.5, "AirTemp": 24.0,
        "Humidity": 45.0, "Rainfall": False, "EventName": event_name, "Location": event_name,
        "RoundNumber": float(round_num), "Year": float(season)
    }


def build_mock_raw_parquet_df(season=2023, round_num=1, event_name="Sakhir", n_drivers=20, n_laps=50):
    """Generate a fully valid 39-column raw DataFrame fixture."""
    drivers = [f"D{i+1:02d}" for i in range(n_drivers)]
    rows = []
    for drv_idx, drv in enumerate(drivers):
        curr_tl, curr_comp, stint = 1.0, "MEDIUM", 1.0
        for lnum in range(1, n_laps + 1):
            is_pit_in, is_pit_out = (lnum == 25), (lnum == 26)
            if is_pit_out:
                curr_tl, curr_comp, stint = 1.0, "HARD", 2.0
            rows.append(build_raw_row(season, round_num, event_name, drv, drv_idx, lnum, is_pit_in, is_pit_out, curr_tl, curr_comp, stint))
            curr_tl += 1.0
    return pd.DataFrame(rows)


def build_mock_openf1_laps(base_time):
    """Generate mock lap records for drivers 1 and 11."""
    laps = []
    for d in [1, 11]:
        cum_sec = 0.0
        for lnum in range(1, 51):
            dur = 90.0 + (lnum * 0.05)
            cum_sec += dur
            is_out = (lnum in [1, 21 if d == 1 else 23])
            laps.append({
                "driver_number": d, "lap_number": lnum, "lap_duration": dur,
                "date_start": (base_time + pd.Timedelta(seconds=cum_sec - dur)).isoformat(),
                "duration_sector_1": 30.0, "duration_sector_2": 30.0, "duration_sector_3": dur - 60.0,
                "i1_speed": 290, "i2_speed": 280, "st_speed": 310, "is_pit_out_lap": is_out
            })
    return laps


def build_mock_openf1_data(season=2023, round_num=1, session_key=9100):
    """Generate mock OpenF1 JSON endpoint fixtures."""
    drivers = [
        {"driver_number": 1, "name_acronym": "VER", "session_key": session_key},
        {"driver_number": 11, "name_acronym": "PER", "session_key": session_key}
    ]
    stints = [
        {"driver_number": 1, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "lap_end": 20, "tyre_age_at_start": 0},
        {"driver_number": 1, "stint_number": 2, "compound": "HARD", "lap_start": 21, "lap_end": 50, "tyre_age_at_start": 0},
        {"driver_number": 11, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "lap_end": 22, "tyre_age_at_start": 0},
        {"driver_number": 11, "stint_number": 2, "compound": "HARD", "lap_start": 23, "lap_end": 50, "tyre_age_at_start": 0}
    ]
    base_time = pd.Timestamp(f"{season}-03-05 15:00:00+00:00")
    laps = build_mock_openf1_laps(base_time)
    weather = [
        {"date": (base_time + pd.Timedelta(minutes=m*10)).isoformat(), "track_temperature": 32.0, "air_temperature": 24.0, "humidity": 45.0, "rainfall": False}
        for m in range(10)
    ]
    rc = [
        {"date": (base_time + pd.Timedelta(minutes=5)).isoformat(), "category": "Flag", "flag": "GREEN", "message": "GREEN FLAG"},
        {"date": (base_time + pd.Timedelta(minutes=15)).isoformat(), "category": "LapDeleted", "message": "CAR 1 (VER) - LAP 10 DELETED - TRACK LIMITS"}
    ]
    return {"drivers": drivers, "stints": stints, "laps": laps, "weather": weather, "race_control": rc}
