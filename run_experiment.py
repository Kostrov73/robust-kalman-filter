#!/usr/bin/env python3
"""Run classical KF vs Huber robust KF on a motor-temperature CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from motor_kf import (
    add_corruption, estimate_noise, run_kf, calculate_metrics,
    summarize_metrics, choose_csv, detect_profiles
)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=None, help="Input CSV path")
    p.add_argument("--target", default="pm", help="Temperature column (default: pm)")
    p.add_argument("--group", default="profile_id", help="Profile/group column; use '' for one series")
    p.add_argument("--sample-every", type=int, default=1, help="Keep every Nth row per profile")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--outlier-rates", default="0,0.01,0.05,0.10,0.20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--huber-c", type=float, default=2.0)
    p.add_argument("--results", type=Path, default=Path("results"))
    return p.parse_args()

def main():
    a = parse_args()
    csv_path = choose_csv(a.csv)
    df = pd.read_csv(csv_path)
    if a.target not in df.columns:
        raise SystemExit(f"Target column '{a.target}' not found. Available columns: {list(df.columns)}")
    group_col = a.group.strip() or None
    profiles = detect_profiles(df, group_col)
    rates = [float(x) for x in a.outlier_rates.split(",")]
    if not (0 < a.train_fraction < 1):
        raise SystemExit("--train-fraction must be between 0 and 1")
    if a.sample_every < 1:
        raise SystemExit("--sample-every must be >= 1")
    a.results.mkdir(parents=True, exist_ok=True)

    rows, example = [], None
    rng = np.random.default_rng(a.seed)
    for profile_name, part in profiles:
        values = pd.to_numeric(part[a.target], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if a.sample_every > 1:
            values = values[::a.sample_every]
        n = len(values)
        split = int(n * a.train_fraction)
        if split < 30 or n - split < 20:
            print(f"Skipping {profile_name}: not enough valid rows (n={n})")
            continue
        train, test = values[:split], values[split:]
        params = estimate_noise(train, dt=0.5 * a.sample_every)
        dt = 0.5 * a.sample_every

        for rate in rates:
            scenarios = ["clean"] if rate == 0 else ["impulsive", "gaussian", "student_t"]
            for corruption in scenarios:
                obs, mask = add_corruption(
                    test, rate=rate, kind=corruption, rng=rng,
                    scale=params["measurement_scale"]
                )
                for method in ("KF", "Huber-RKF"):
                    result = run_kf(
                        obs, dt=dt, q=params["q"], r=params["r"],
                        method=method, huber_c=a.huber_c
                    )
                    metrics = calculate_metrics(test, result["estimate"], result["runtime"])
                    rows.append({
                        "profile": str(profile_name), "scenario": corruption,
                        "outlier_rate": rate, "filter": method,
                        "n_test": len(test), **metrics
                    })
                    if example is None and (rate == max(rates) or len(rates) == 1):
                        example = (profile_name, test, obs, mask, result, rate)

    if not rows:
        raise SystemExit("No profiles had enough valid rows. Check target/group columns or data length.")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(a.results / "metrics.csv", index=False)
    summary = summarize_metrics(metrics)
    summary.to_csv(a.results / "metrics_summary.csv", index=False)

    if example is not None:
        name, clean, obs, mask, _, rate = example
        # Re-run both methods for a common illustrative profile/scenario.
        params = estimate_noise(clean[:max(30, int(len(clean)*0.5))], dt=dt)
        kf = run_kf(obs, dt=dt, q=params["q"], r=params["r"], method="KF")
        rkf = run_kf(obs, dt=dt, q=params["q"], r=params["r"], method="Huber-RKF", huber_c=a.huber_c)
        plt.figure(figsize=(13, 5))
        idx = np.arange(len(clean))
        plt.plot(idx, clean, label="Uncorrupted recorded pm", linewidth=1.8)
        plt.plot(idx, obs, ".", alpha=0.35, markersize=2, label="Corrupted observations")
        plt.plot(idx, kf["estimate"], label="Classical KF", linewidth=1.1)
        plt.plot(idx, rkf["estimate"], label="Huber-RKF", linewidth=1.1)
        if mask.any():
            plt.scatter(idx[mask], obs[mask], s=18, marker="x", label="Injected outliers")
        plt.xlabel("Test sample")
        plt.ylabel(a.target)
        plt.title(f"Profile {name}: KF vs Huber-RKF (outlier rate={rate:.0%})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(a.results / "example_profile.png", dpi=160)
        plt.close()

    plot_error_curves(metrics, a.results / "error_by_outlier_rate.png")
    metadata = {
        "csv": str(csv_path), "target": a.target, "group": group_col,
        "profiles_processed": int(metrics["profile"].nunique()),
        "sample_every": a.sample_every, "train_fraction": a.train_fraction,
        "outlier_rates": rates, "seed": a.seed, "huber_c": a.huber_c,
        "note": "Original target column is used as clean reference; synthetic corruption is applied to test observations."
    }
    (a.results / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed. Results saved to: {a.results.resolve()}")
    print(summary.to_string(index=False))

def plot_error_curves(df, path):
    agg = df.groupby(["scenario", "outlier_rate", "filter"], as_index=False)[["RMSE", "MAE"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, g in agg.groupby("filter"):
        axes[0].plot(g["outlier_rate"] * 100, g["RMSE"], marker="o", label=method)
        axes[1].plot(g["outlier_rate"] * 100, g["MAE"], marker="o", label=method)
    axes[0].set(title="RMSE vs outlier rate", xlabel="Injected outliers (%)", ylabel="RMSE")
    axes[1].set(title="MAE vs outlier rate", xlabel="Injected outliers (%)", ylabel="MAE")
    for ax in axes:
        ax.grid(True, alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

if __name__ == "__main__":
    main()
