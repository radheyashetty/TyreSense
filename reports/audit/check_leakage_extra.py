import numpy as np
import pandas as pd
from src.features import fit_baseline_curves, eval_curve
from src.claim_eval import forecast_stint_pace
from src.clean import flag_spikes_causal

laps = pd.read_parquet("data/processed/laps.parquet")
stints = pd.read_parquet("data/processed/stints.parquet")
curves, global_curve = fit_baseline_curves(laps)

# Pick 5 distinct stints with at least 8 laps
stint_ids = laps["stint_id"].unique()[:5]
print("Running extra leakage check on 5 stints:", stint_ids)

all_passed = True
for s_id in stint_ids:
    sub = laps[laps["stint_id"] == s_id].sort_values("stint_lap").copy().reset_index(drop=True)
    t = 4 # 0-indexed lap 4 (stint lap 5)
    
    # Baseline run at lap t
    obs_orig = sub.iloc[:t + 1].copy()
    spike_orig = flag_spikes_causal(obs_orig)
    deg_orig = forecast_stint_pace(sub, curves, global_curve, lap_idx=t)
    
    # Perturb lap t+1 (stint lap 6) by +5.0 seconds
    sub_perturbed = sub.copy()
    sub_perturbed.loc[t + 1, "lap_time_s"] += 5.0
    sub_perturbed.loc[t + 1, "lap_time_fc_s"] += 5.0
    
    obs_pert = sub_perturbed.iloc[:t + 1].copy()
    spike_pert = flag_spikes_causal(obs_pert)
    deg_pert = forecast_stint_pace(sub_perturbed, curves, global_curve, lap_idx=t)
    
    spike_match = (spike_orig == spike_pert)
    deg_match = np.allclose(deg_orig, deg_pert, atol=1e-7)
    max_deg_diff = float(np.max(np.abs(deg_orig - deg_pert)))
    
    print(f"Stint {s_id}: spike_match={spike_match}, deg_match={deg_match}, max_diff={max_deg_diff:.8e}")
    if not (spike_match and deg_match):
        all_passed = False

print("Extra Leakage Check All Passed:", all_passed)
