import os
import glob
import difflib
import argparse
from datetime import date
import yaml
import pandas as pd

TRACKS_YAML_PATH = "config/tracks.yaml"


def load_tracks_config(path=TRACKS_YAML_PATH):
    """Load and return the tracks configuration dictionary."""
    if not os.path.exists(path):
        return {"tracks": {}}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"tracks": {}}


def save_tracks_config(cfg, path=TRACKS_YAML_PATH):
    """Save tracks configuration with sorted keys."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sorted_cfg = {"tracks": dict(sorted(cfg.get("tracks", {}).items()))}
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(sorted_cfg, f, default_flow_style=False, sort_keys=False)


def canonical_track_id(name, tracks_cfg=None):
    """Resolve circuit alias or name to canonical lowercase track_id."""
    if not name or not isinstance(name, str):
        raise ValueError(f"Invalid circuit name: {name}")
    norm = name.strip().lower()
    cfg = tracks_cfg or load_tracks_config()
    all_names = []
    for tid, info in cfg.get("tracks", {}).items():
        if norm == tid.lower():
            return tid
        all_names.append(tid)
        for alias in info.get("aliases", []):
            if norm == alias.strip().lower():
                return tid
            all_names.append(alias)
    closest = difflib.get_close_matches(name, all_names, n=1)
    sug = f" (did you mean '{closest[0]}'?) " if closest else ""
    raise ValueError(f"Unknown circuit '{name}'{sug}; not found in tracks.yaml")


def track_length_km(track_id, season=None, tracks_cfg=None):
    """Return circuit length in km, checking season overrides first."""
    cfg = tracks_cfg or load_tracks_config()
    info = cfg.get("tracks", {}).get(track_id)
    if not info:
        return None
    if season is not None:
        overrides = info.get("length_km_by_season", {})
        if overrides and int(season) in overrides:
            return float(overrides[int(season)])
    base_len = info.get("length_km")
    return float(base_len) if base_len is not None else None


def unknown_names_in_data(races_csv="reports/races.csv", tracks_cfg=None):
    """Return list of event names in races.csv that do not resolve to track_id."""
    if not os.path.exists(races_csv):
        return []
    df = pd.read_csv(races_csv)
    cfg = tracks_cfg or load_tracks_config()
    unknowns = []
    for ev in df["event_name"].dropna().unique():
        try:
            canonical_track_id(ev, cfg)
        except ValueError:
            unknowns.append(ev)
    return sorted(unknowns)


def coverage_matrix(races_csv="reports/races.csv", tracks_cfg=None):
    """Return DataFrame showing circuit appearance (1/0) across seasons."""
    if not os.path.exists(races_csv):
        return pd.DataFrame()
    df = pd.read_csv(races_csv)
    cfg = tracks_cfg or load_tracks_config()
    all_tracks = sorted(cfg.get("tracks", {}).keys())
    matrix = {s: {t: 0 for t in all_tracks} for s in sorted(df["season"].unique())}
    for _, r in df.iterrows():
        try:
            tid = canonical_track_id(r["event_name"], cfg)
            matrix[r["season"]][tid] = 1
        except ValueError:
            pass
    res_df = pd.DataFrame(matrix)
    res_df.index.name = "track_id"
    return res_df


def print_status(tracks_cfg=None, races_csv="reports/races.csv"):
    """Print circuit counts, validation/test-only tracks, and unverified lengths."""
    cfg = tracks_cfg or load_tracks_config()
    cov = coverage_matrix(races_csv, cfg)
    print("=== TRACK DATA SYSTEM STATUS ===")
    if not cov.empty:
        for s in cov.columns:
            print(f"  Season {s}: {cov[s].sum()} circuits")
        tr = set(cov[(cov.get(2022, 0) == 1) | (cov.get(2023, 0) == 1)].index)
        later = set(cov[(cov.get(2024, 0) == 1) | (cov.get(2025, 0) == 1)].index)
        print(f"Circuits seen ONLY in val/test (2024-2025): {sorted(later - tr)}")
    unverified = [t for t, i in cfg.get("tracks", {}).items() if not i.get("verified")]
    null_len = [t for t, i in cfg.get("tracks", {}).items() if i.get("length_km") is None]
    print(f"Circuits without verified length ({len(unverified)}): {unverified}")
    print(f"Circuits with null length ({len(null_len)}): {null_len}")
    unknowns = unknown_names_in_data(races_csv, cfg)
    print(f"Unknown names in races.csv ({len(unknowns)}): {unknowns}")


def scan_raw_names(raw_dir="data/raw", tracks_cfg=None):
    """Scan raw parquet EventNames and suggest closest matches for unknowns."""
    cfg = tracks_cfg or load_tracks_config()
    all_aliases = []
    for tid, info in cfg.get("tracks", {}).items():
        all_aliases.extend(info.get("aliases", []))
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.parquet"))):
        df = pd.read_parquet(p, columns=["EventName"])
        ev = df["EventName"].iloc[0]
        try:
            canonical_track_id(ev, cfg)
        except ValueError:
            closest = difflib.get_close_matches(ev, all_aliases, n=2)
            print(f"Unknown: '{ev}' in {os.path.basename(p)} -> closest: {closest}")


def add_alias(track_id, alias, path=TRACKS_YAML_PATH):
    """Add alias to track; reject if alias already exists on any track."""
    cfg = load_tracks_config(path)
    if track_id not in cfg.get("tracks", {}):
        raise ValueError(f"Track '{track_id}' not found in tracks config.")
    for tid, info in cfg["tracks"].items():
        for existing in info.get("aliases", []):
            if existing.strip().lower() == alias.strip().lower():
                raise ValueError(f"Alias '{alias}' already belongs to '{tid}'")
    cfg["tracks"][track_id].setdefault("aliases", []).append(alias)
    save_tracks_config(cfg, path)


def set_length(track_id, km, source_url, season=None, path=TRACKS_YAML_PATH):
    """Set length, update verified flag and date, and save."""
    cfg = load_tracks_config(path)
    if track_id not in cfg.get("tracks", {}):
        raise ValueError(f"Track '{track_id}' not found.")
    info = cfg["tracks"][track_id]
    if season:
        info.setdefault("length_km_by_season", {})[int(season)] = float(km)
    else:
        info["length_km"] = float(km)
    info["verified"], info["verified_on"] = True, str(date.today())
    if source_url:
        info["source"] = source_url
    save_tracks_config(cfg, path)


def add_track(track_id, name, country, path=TRACKS_YAML_PATH):
    """Add new track to configuration."""
    cfg = load_tracks_config(path)
    if track_id in cfg.get("tracks", {}):
        raise ValueError(f"Track '{track_id}' already exists.")
    cfg.setdefault("tracks", {})[track_id] = {
        "name": name, "country": country, "aliases": [name], "length_km": None,
        "openf1_circuit_key": None, "source": "", "verified": False, "verified_on": None, "notes": ""
    }
    save_tracks_config(cfg, path)


def main():
    """CLI router for src.tracks."""
    p = argparse.ArgumentParser(description="Track Data System")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("status")
    sub.add_parser("scan")
    pa = sub.add_parser("add-alias")
    pa.add_argument("track_id"); pa.add_argument("name")
    pl = sub.add_parser("set-length")
    pl.add_argument("track_id"); pl.add_argument("km", type=float)
    pl.add_argument("--source", required=True); pl.add_argument("--season", type=int, default=None)
    pt = sub.add_parser("add-track")
    pt.add_argument("track_id"); pt.add_argument("name"); pt.add_argument("--country", required=True)
    args = p.parse_args()
    if args.command == "status":
        print_status()
    elif args.command == "scan":
        scan_raw_names()
    elif args.command == "add-alias":
        add_alias(args.track_id, args.name)
    elif args.command == "set-length":
        set_length(args.track_id, args.km, args.source, args.season)
    elif args.command == "add-track":
        add_track(args.track_id, args.name, args.country)


if __name__ == "__main__":
    main()
