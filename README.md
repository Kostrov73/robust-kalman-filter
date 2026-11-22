# Robust KF vs. classical KF: PMSM motor temperature

A reproducible Python project for comparing a classical linear Kalman filter (KF) with a robust innovation-weighted Kalman filter (Huber-RKF) on the Electric Motor Temperature CSV dataset.

## Research question

How much does Huber-based innovation down-weighting improve temperature estimation when sensor observations contain impulsive outliers or heavy-tailed noise, compared with a classical KF using the same state-space model?

## Important interpretation

The CSV's `pm` column is treated as the **clean reference series** for the experiment. A corrupted copy is generated and supplied to both filters as the sensor observation. This is a controlled robustness experiment, not proof that the original `pm` measurements are ground-truth physical temperatures. Results should be described as error relative to the uncorrupted recorded `pm` series.

The project does not claim the original CSV contains outliers. It injects reproducible synthetic disturbances into the test observations and reports clean-data performance separately.

## Dataset

Kaggle: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature

Download the CSV and place it in `data/`. The dataset commonly contains columns such as `pm`, `profile_id`, `ambient`, `coolant`, `motor_speed`, `torque`, and stator temperatures. The program detects the CSV automatically if there is exactly one CSV in `data/`.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py --csv data/measurements.csv
```

Or, if there is exactly one CSV in `data/`:

```bash
python run_experiment.py
```

Useful options:

```bash
python run_experiment.py --csv data/my_data.csv --target pm --group profile_id
python run_experiment.py --csv data/my_data.csv --target pm --sample-every 1
python run_experiment.py --csv data/my_data.csv --outlier-rates 0,0.01,0.05,0.10,0.20
```

`--sample-every N` keeps every Nth row within each profile to reduce runtime. Keep it at 1 for the full data.

## Experimental protocol

1. Sort each profile by its original row order (the CSV is expected to be time-ordered within each profile).
2. Split each profile chronologically: first 70% is calibration/training, final 30% is test.
3. Estimate a two-state local-linear-trend model:
   - state: `[temperature, temperature_rate]`
   - observation: measured target temperature
4. Estimate process/measurement noise parameters using only the training segment.
5. On the test segment, compare:
   - Classical KF: standard Gaussian innovation update.
   - Huber-RKF: same prediction and state model, but inflates measurement variance when the normalized innovation is large. This reduces the influence of isolated extreme observations.
6. Repeat with sparse Gaussian contamination, Student-t heavy-tailed contamination, and impulsive outliers at several rates.
7. Report RMSE, MAE, median absolute error, 95th-percentile absolute error, maximum absolute error, and runtime.

The clean test target is the original uncorrupted `pm` value. Both algorithms receive exactly the same corrupted observations.

## Outputs

The run creates `results/` containing:
- `metrics.csv`: one row per profile, corruption type/rate, and filter.
- `metrics_summary.csv`: metrics aggregated across profiles.
- `example_profile.png`: example clean target, corrupted observations, KF and Huber-RKF estimates.
- `error_by_outlier_rate.png`: RMSE and MAE versus outlier rate.
- `run_metadata.json`: settings and selected columns.

## Method and equations

State transition (constant-rate model):

`x[k] = F x[k-1] + w[k]`, with `x = [T, dT/dt]^T` and
`F = [[1, dt], [0, 1]]`.

Observation:

`z[k] = H x[k] + v[k]`, with `H = [1, 0]`.

For the robust update, calculate normalized innovation `u = innovation / sqrt(S)`.
Huber weight is `w = min(1, c / abs(u))`. The effective measurement variance is
`R_eff = R / w^2`. The standard KF update is then performed using `R_eff`.
This is an innovation-based robustification, not the only possible definition of a robust KF.

## Scientific cautions / extensions

- `pm` is itself a sensor-derived dataset column. Without an independent calibrated reference sensor, the study evaluates recovery of the recorded series, not absolute physical truth.
- A random row split would leak highly correlated adjacent time samples. This project uses chronological splitting.
- If the CSV has multiple profiles, each profile is split independently and metrics are aggregated.
- The baseline and robust filter use the same model and parameters; only the measurement update differs.
- Noise parameters are estimated with robust heuristics from the training segment; include a parameter-sensitivity study before drawing strong conclusions. The current model is deliberately univariate. A future extension can use coolant, ambient, torque, speed and currents in a thermal state-space model, but that requires a physically justified observation/process model.
- Report sensitivity to Huber threshold `c` (e.g. 1.5, 2.0, 2.5, 3.0) and random seeds in a paper.
