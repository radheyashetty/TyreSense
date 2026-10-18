# AGENT_RULES.md  (load this first in every Antigravity session)

Project: TyreSense, an F1 tyre-life forecaster (course project, deep learning).
The student must explain every line in a viva. Write the code a careful 4th-year
student would write: simple, readable, a bit plain. Not a framework.

## Code style
- Plain functions. The only classes allowed are torch nn.Module models.
- One job per function, under 30 lines, with a one-line docstring.
- One file per job, under 200 lines. Flat layout: src/, tests/, notes/, reports/.
- Plain names: tyre_age, lap_time, train_df. No abbreviations, no clever one-liners.
- Prefer simple for-loops and basic pandas. No nested comprehensions, lambdas,
  decorators, dataclasses, generics, async, or dynamic imports.
- Comments are short and say WHY. Do not comment every line.
- Use print() for progress. No logging framework. argparse only in train.py.
- Settings live in config.yaml, read by one function load_config(). No hidden defaults.
- Write the training loop by hand. No Lightning, Hydra, MLflow, W&B.
- Allowed libraries: numpy, pandas, scikit-learn, xgboost, torch, fastf1, streamlit,
  plotly, matplotlib, pytest, pyyaml. Ask before adding anything else.
- No code for future use. If it is not needed for the current task, do not write it.

Style example (this is the level to aim for):

    def fuel_correct(lap_time, lap_number, total_laps, c=0.03):
        """Remove the fuel effect: early laps carry more fuel so they are slower."""
        fuel_penalty = c * (total_laps - lap_number)
        return lap_time - fuel_penalty

## Token rules (keep sessions cheap)
- One task per session, named by ID (for example T1.2). Read only AGENT_RULES.md and
  that task's section. Do not read the whole repo or other tasks.
- Never print dataframes or logs in full. Use head(5), shape, and a few summary numbers.
- Tests and quick checks use 2 cached races, not the full dataset.
- Do not run full training or CV unless the task says so. Use a tiny run first.
- Edit with small diffs. Do not re-print whole files after a change.
- Final reply is at most 8 lines: files changed, test result, one risk.

## Process (every task)
1. Plan in 3 bullets. Wait for my OK.
2. Write the code and one simple pytest. Run it. Report failures honestly.
3. Give a 3-line plain-English note for notes/explain.md: what it does, why, what breaks.
4. Touch only files named in the task. Do not refactor working code.
5. Never invent numbers, results or citations. Say "unknown" instead.
6. Never open or evaluate the test split (2025) until I say "open test".

## Hand-written by the student (agent only reviews, does not write)
numpy MLP + gradient check (T1.6), live-correction loop (T2.2), fuel function (T1.2).
For these tasks: read my file, list bugs and unclear parts in 5 bullets, nothing else.

## Results and reporting
1. A number may appear in the report, README or app text only if it is read from a file in
   reports/ or runs/ by code, or copied from such a file with the file name given next to it.
2. Never write a result sentence before the experiment has run. Hypotheses are written first
   and are labelled "hypothesis".
3. Every experiment table shows ALL conditions tested (all laps, all seeds, all horizons),
   with mean and standard deviation, and the number of stints used (N).
4. Negative or surprising results are stated plainly in the same section as positive ones.
5. If a metric is dropped for some stints (for example NaN), report how many and why.
6. The test season (2025) is read exactly once, by src/final_eval.py, at the very end.
   Any other code that reads the test split is a bug.
7. No synthetic or random data may be used to produce a reported result. Synthetic data is
   allowed only inside tests, and test outputs are never quoted as results.
8. Words not allowed unless literally true: Bayesian, Kalman, temperature scaling,
   guarantee, proves, causal, measured wear.
