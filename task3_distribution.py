import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_load import load_data_chunked
import os

OUTPUT_DIR = "output"
EXTREME_THRESHOLD = 200.0


def run_distribution_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cols = ["station_id", "zone", "pm25"]
    df = load_data_chunked(usecols=cols)

    industrial_stations = df.loc[df["zone"] == "Industrial", "station_id"].unique()
    target_station = industrial_stations[0]
    pm25_data = df.loc[df["station_id"] == target_station, "pm25"].values

    p99 = np.percentile(pm25_data, 99)
    extreme_count = np.sum(pm25_data > EXTREME_THRESHOLD)
    extreme_prob = extreme_count / len(pm25_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1 = axes[0]
    n_bins_linear = 80
    ax1.hist(pm25_data, bins=n_bins_linear, color="#4c72b0", edgecolor="none", alpha=0.9)
    ax1.axvline(p99, color="#d62728", linestyle="--", linewidth=1.5, label=f"99th pctl = {p99:.1f}")
    ax1.axvline(EXTREME_THRESHOLD, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"Extreme = {EXTREME_THRESHOLD}")
    ax1.set_xlabel("PM2.5 (ug/m3)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Linear Histogram  --  {target_station} (peaks)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(False)

    ax2 = axes[1]
    bins_log = np.logspace(np.log10(max(pm25_data.min(), 0.1)), np.log10(pm25_data.max()), 60)
    ax2.hist(pm25_data, bins=bins_log, color="#4c72b0", edgecolor="none", alpha=0.9)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.axvline(p99, color="#d62728", linestyle="--", linewidth=1.5, label=f"99th pctl = {p99:.1f}")
    ax2.axvline(EXTREME_THRESHOLD, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"Extreme = {EXTREME_THRESHOLD}")
    ax2.set_xlabel("PM2.5 (ug/m3)  [log scale]")
    ax2.set_ylabel("Frequency  [log scale]")
    ax2.set_title(f"Log-Log Histogram  --  {target_station} (tails)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "task3_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "station": target_station,
        "p99": p99,
        "extreme_count": int(extreme_count),
        "extreme_probability": extreme_prob,
        "total_points": len(pm25_data),
        "figure": os.path.join(OUTPUT_DIR, "task3_distribution.png"),
    }


if __name__ == "__main__":
    results = run_distribution_analysis()
    print(f"Station: {results['station']}")
    print(f"99th percentile: {results['p99']:.2f}")
    print(f"Extreme events (>{EXTREME_THRESHOLD}): {results['extreme_count']}")
    print(f"Extreme probability: {results['extreme_probability']:.6f}")
