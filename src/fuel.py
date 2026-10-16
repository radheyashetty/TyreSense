def fuel_correct(lap_time, lap_number, total_laps, c=0.03):
    """Remove the fuel effect: early laps carry more fuel so they are slower."""
    fuel_penalty = c * (total_laps - lap_number)
    return lap_time - fuel_penalty
