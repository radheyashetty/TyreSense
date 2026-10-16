import warnings


def compute_fuel_coefficient(track_id, config):
    """Compute fuel burn coefficient: constant or scaled by circuit length."""
    fuel_cfg = config.get("fuel", {})
    mode = fuel_cfg.get("mode", "constant")
    base_c = fuel_cfg.get("base_c", config.get("cleaning", {}).get("fuel_c", 0.03))
    if mode != "track_scaled":
        return base_c

    ref_km = fuel_cfg.get("ref_km", 5.0)
    circuit_table = config.get("circuit_length_km", {})
    if track_id in circuit_table:
        length_km = circuit_table[track_id]
        return float(base_c * (length_km / ref_km))

    warnings.warn(f"Unknown track '{track_id}' for track_scaled fuel; fallback to {base_c}")
    return base_c


def fuel_correct(lap_time, lap_number, total_laps, c=0.03):
    """Remove the fuel effect: early laps carry more fuel so they are slower."""
    fuel_penalty = c * (total_laps - lap_number)
    return lap_time - fuel_penalty
