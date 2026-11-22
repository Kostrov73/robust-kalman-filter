"""Core algorithms and experiment utilities."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd

def choose_csv(path):
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = list(Path("data").glob("*.csv"))
    if len(candidates) != 1:
        raise SystemExit("Put exactly one CSV in data/ or pass --csv path/to/file.csv")
    return candidates[0]

def detect_profiles(df, group_col):
    if group_col and group_col in df.columns:
        return [(key, part.reset_index(drop=True)) for key, part in df.groupby(group_col, sort=False)]
    if group_col and group_col not in df.columns:
        print(f"Warning: group column '{group_col}' not found; treating CSV as one time series.")
    return [("all_data", df.reset_index(drop=True))]

def estimate_noise(train, dt=0.5):
    """Robustly estimate observation noise and local-trend process noise."""
    d = np.diff(train)
    if len(d) < 5:
        raise ValueError("At least six training samples are required.")
    # First-difference MAD estimates short-term observation variation.
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    diff_sigma = max(1.4826 * mad / np.sqrt(2), 1e-6)
    # Measurement noise estimate: conservative fraction of short-term variation.
    # A floor avoids a numerically singular update on very smooth series.
    r = max((0.35 * diff_sigma) ** 2, 1e-4)
    # Continuous white-acceleration model; tune q from observed curvature.
    dd = np.diff(train, n=2)
    dd_mad = np.median(np.abs(dd - np.median(dd))) if len(dd) else diff_sigma
    accel_sigma = max(1.4826 * dd_mad / max(dt * dt, 1e-9), 1e-5)
    q = max(accel_sigma ** 2 * 0.02, 1e-7)
    return {"q": q, "r": r, "measurement_scale": max(np.sqrt(r), diff_sigma, 1e-3)}

def transition_matrices(dt, q):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    # Discrete white-acceleration process noise covariance.
    Q = q * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
    H = np.array([[1.0, 0.0]])
    return F, Q, H

def run_kf(observations, dt, q, r, method="KF", huber_c=2.0):
    z = np.asarray(observations, dtype=float)
    F, Q, H = transition_matrices(dt, q)
    x = np.array([z[0], 0.0], dtype=float)
    P = np.diag([max(r, 1e-6), 1.0])
    estimates = np.empty(len(z))
    start = time.perf_counter()
    I = np.eye(2)
    for k, measurement in enumerate(z):
        if k:
            x = F @ x
            P = F @ P @ F.T + Q
        innovation = float(measurement - (H @ x)[0])
        S = float((H @ P @ H.T)[0, 0] + r)
        r_eff = r
        if method == "Huber-RKF":
            u = abs(innovation) / np.sqrt(max(S, 1e-12))
            weight = min(1.0, huber_c / max(u, 1e-12))
            r_eff = r / max(weight * weight, 1e-8)
            S = float((H @ P @ H.T)[0, 0] + r_eff)
        elif method != "KF":
            raise ValueError("method must be 'KF' or 'Huber-RKF'")
        K = (P @ H.T) / S
        x = x + K[:, 0] * innovation
        # Joseph form improves numerical stability.
        A = I - K @ H
        P = A @ P @ A.T + (K * r_eff) @ K.T
        estimates[k] = x[0]
    return {"estimate": estimates, "runtime": time.perf_counter() - start}

def add_corruption(clean, rate, kind, rng, scale):
    clean = np.asarray(clean, dtype=float)
    obs = clean.copy()
    mask = np.zeros(len(clean), dtype=bool)
    if kind == "clean" or rate <= 0:
        return obs, mask
    n = max(1, int(round(len(clean) * rate)))
    indices = rng.choice(len(clean), size=min(n, len(clean)), replace=False)
    mask[indices] = True
    if kind == "impulsive":
        # Sparse, large signed impulses.
        signs = rng.choice([-1.0, 1.0], size=len(indices))
        amplitudes = rng.uniform(6.0, 12.0, size=len(indices)) * scale
        obs[indices] += signs * amplitudes
    elif kind == "gaussian":
        obs[indices] += rng.normal(0.0, 3.0 * scale, size=len(indices))
    elif kind == "student_t":
        obs[indices] += rng.standard_t(df=3, size=len(indices)) * 3.0 * scale
    else:
        raise ValueError(f"Unknown corruption: {kind}")
    return obs, mask

def calculate_metrics(reference, estimate, runtime):
    e = np.asarray(reference) - np.asarray(estimate)
    ae = np.abs(e)
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(ae)),
        "MedianAE": float(np.median(ae)),
        "P95AE": float(np.percentile(ae, 95)),
        "MaxAE": float(np.max(ae)),
        "runtime_sec": float(runtime),
    }

def summarize_metrics(df):
    return (df.groupby(["scenario", "outlier_rate", "filter"], as_index=False)
              [["RMSE", "MAE", "MedianAE", "P95AE", "MaxAE", "runtime_sec"]]
              .mean())
