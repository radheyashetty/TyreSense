import os
import shutil
import pytest
import pandas as pd
from src.tracks import (
    load_tracks_config, canonical_track_id, track_length_km,
    add_alias, set_length, add_track
)
from src.download import process_single_race


def test_aliases_resolve_to_correct_track():
    """Verify every alias resolves back to its canonical track_id."""
    cfg = load_tracks_config()
    for tid, info in cfg.get("tracks", {}).items():
        assert canonical_track_id(tid, cfg) == tid
        for alias in info.get("aliases", []):
            assert canonical_track_id(alias, cfg) == tid


def test_no_shared_aliases():
    """Ensure no two distinct circuits share any alias."""
    cfg = load_tracks_config()
    seen = {}
    for tid, info in cfg.get("tracks", {}).items():
        for alias in info.get("aliases", []):
            norm = alias.strip().lower()
            assert norm not in seen, f"Duplicate alias '{norm}' on {tid} and {seen[norm]}"
            seen[norm] = tid


def test_all_91_downloaded_races_resolve():
    """Ensure every downloaded race in reports/races.csv resolves cleanly."""
    df = pd.read_csv("reports/races.csv")
    valid_races = df[df["status"].isin(["cached", "downloaded"])]
    assert len(valid_races) == 91
    for _, r in valid_races.iterrows():
        tid = canonical_track_id(r["event_name"])
        assert tid == r["track_id"]
        assert len(tid) > 0


def test_circuit_lengths_reasonable():
    """Check that all track lengths are physically reasonable (2.5 km to 8.0 km)."""
    cfg = load_tracks_config()
    for tid in cfg.get("tracks", {}).keys():
        length = track_length_km(tid, tracks_cfg=cfg)
        assert length is not None, f"Track {tid} has null length"
        assert 2.5 <= length <= 8.0, f"Track {tid} length {length} out of [2.5, 8.0] km"


def test_season_overrides_within_5_percent():
    """Check that season overrides differ by at most 5% from base length."""
    cfg = load_tracks_config()
    for tid, info in cfg.get("tracks", {}).items():
        base_len = info.get("length_km")
        overrides = info.get("length_km_by_season", {})
        for season, override_len in overrides.items():
            pct_diff = abs(override_len - base_len) / base_len
            assert pct_diff <= 0.05, f"Override for {tid} season {season} differs by {pct_diff:.2%}"


def test_cli_add_track_and_alias_on_temp_copy(tmp_path):
    """Test add_track and add_alias on an isolated copy of tracks.yaml."""
    temp_yaml = os.path.join(tmp_path, "tracks.yaml")
    shutil.copy("config/tracks.yaml", temp_yaml)
    add_track("portimao", "Algarve International Circuit", "Portugal", path=temp_yaml)
    cfg = load_tracks_config(temp_yaml)
    assert "portimao" in cfg["tracks"]
    add_alias("portimao", "Portuguese Grand Prix", path=temp_yaml)
    assert canonical_track_id("Portuguese Grand Prix", load_tracks_config(temp_yaml)) == "portimao"


def test_cli_set_length_on_temp_copy(tmp_path):
    """Test set_length updates length, verified flag and timestamp on temp config."""
    temp_yaml = os.path.join(tmp_path, "tracks.yaml")
    shutil.copy("config/tracks.yaml", temp_yaml)
    url = "https://example.com/circuit"
    set_length("monaco", 3.340, url, season=2026, path=temp_yaml)
    cfg = load_tracks_config(temp_yaml)
    assert cfg["tracks"]["monaco"]["length_km_by_season"][2026] == 3.340
    assert cfg["tracks"]["monaco"]["verified"] is True


def test_duplicate_alias_rejected(tmp_path):
    """Ensure adding an existing alias to another track is rejected with ValueError."""
    temp_yaml = os.path.join(tmp_path, "tracks.yaml")
    shutil.copy("config/tracks.yaml", temp_yaml)
    with pytest.raises(ValueError, match="already belongs to"):
        add_alias("silverstone", "Monaco Grand Prix", path=temp_yaml)


def test_unknown_circuit_suggests_match():
    """Ensure canonical_track_id raises ValueError with suggestion on typo."""
    with pytest.raises(ValueError, match="did you mean"):
        canonical_track_id("Silverston")


def test_download_handles_unknown_track_gracefully(tmp_path):
    """Ensure unknown circuit logs status 'unknown_track' and updates pending file."""
    fake_csv = os.path.join(tmp_path, "races.csv")
    event = {"Year": 2026, "RoundNumber": 1, "EventName": "Fictional GP", "EventDate": "2026-03-01"}
    state = {}
    rec = process_single_race(event, str(tmp_path), fake_csv, None, state)
    assert rec["status"] == "unknown_track"
    pending_csv = "reports/tracks_pending.csv"
    assert os.path.exists(pending_csv)
