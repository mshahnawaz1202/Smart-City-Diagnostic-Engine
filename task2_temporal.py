import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_load import load_data_chunked, CHUNK_SIZE
import os

OUTPUT_DIR = "output"
PM25_THRESHOLD = 35.0


def run_temporal_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cols = ["timestamp", "station_id", "zone", "pm25"]
    df = load_data_chunked(usecols=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    df["violation"] = (df["pm25"] > PM25_THRESHOLD).astype(int)

    hourly_viol = (
        df.groupby(["station_id", "hour"])["violation"]
        .mean()
        .unstack(level="hour")
    )
    hourly_viol = hourly_viol.sort_index()

    monthly_viol = (
        df.groupby(["station_id", "month"])["violation"]
        .mean()
        .unstack(level="month")
    )
    monthly_viol = monthly_viol.sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax1 = axes[0]
    im1 = ax1.imshow(
        hourly_viol.values,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Station")
    ax1.set_title("PM2.5 Violation Rate by Hour")
    ax1.set_xticks(np.arange(0, 24, 3))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=7)
    ytick_step = max(1, len(hourly_viol) // 15)
    ax1.set_yticks(np.arange(0, len(hourly_viol), ytick_step))
    ax1.set_yticklabels(hourly_viol.index[::ytick_step], fontsize=6)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cb1.set_label("Violation Rate")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(False)

    ax2 = axes[1]
    im2 = ax2.imshow(
        monthly_viol.values,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Station")
    ax2.set_title("PM2.5 Violation Rate by Month")
    ax2.set_xticks(np.arange(12))
    ax2.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        fontsize=7,
    )
    ax2.set_yticks(np.arange(0, len(monthly_viol), ytick_step))
    ax2.set_yticklabels(monthly_viol.index[::ytick_step], fontsize=6)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label("Violation Rate")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "task2_temporal.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    hourly_mean = hourly_viol.mean(axis=0)
    monthly_mean = monthly_viol.mean(axis=0)
    hourly_range = hourly_mean.max() - hourly_mean.min()
    monthly_range = monthly_mean.max() - monthly_mean.min()

    if hourly_range > monthly_range:
        dominant = "daily (24-hour traffic cycle)"
    else:
        dominant = "monthly (30-day seasonal shift)"

    return {
        "hourly_heatmap": hourly_viol,
        "monthly_heatmap": monthly_viol,
        "dominant_pattern": dominant,
        "hourly_range": hourly_range,
        "monthly_range": monthly_range,
        "figure": os.path.join(OUTPUT_DIR, "task2_temporal.png"),
    }


if __name__ == "__main__":
    results = run_temporal_analysis()
    print("Dominant pattern:", results["dominant_pattern"])
    print(f"Hourly range: {results['hourly_range']:.4f}")
    print(f"Monthly range: {results['monthly_range']:.4f}")
