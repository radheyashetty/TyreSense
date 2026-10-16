import os
import socket
from src.download import load_config, load_cached_race, download_single_race


def test_load_config():
    """Verify that config.yaml exists and contains required sections."""
    config = load_config("config.yaml")
    assert "paths" in config
    assert "seasons" in config
    assert "cleaning" in config
    assert "model" in config


def test_offline_cached_race_load(monkeypatch):
    """Verify that cached race parquet loads without network calls."""
    config = load_config("config.yaml")
    raw_dir = config["paths"]["raw_dir"]
    parquet_path = os.path.join(raw_dir, "race_2023_01.parquet")
    assert os.path.exists(parquet_path), "Test fixture race_2023_01.parquet missing"

    def block_connect(*args, **kwargs):
        raise RuntimeError("Network disabled in offline test")

    monkeypatch.setattr(socket.socket, "connect", block_connect)
    dataframe = load_cached_race(raw_dir, 2023, 1)
    assert not dataframe.empty
    assert "LapNumber" in dataframe.columns
    assert "TrackTemp" in dataframe.columns
    assert "Year" in dataframe.columns


def test_download_single_race_skips_when_cached(monkeypatch):
    """Verify that download_single_race returns cached file immediately without network."""
    config = load_config("config.yaml")
    raw_dir = config["paths"]["raw_dir"]

    def block_connect(*args, **kwargs):
        raise RuntimeError("Network called for cached file")

    monkeypatch.setattr(socket.socket, "connect", block_connect)
    cached_path = download_single_race(2023, 1, raw_dir)
    assert cached_path is not None
    assert os.path.exists(cached_path)
