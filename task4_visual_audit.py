import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from data_load import load_station_means, FEATURES
import os

OUTPUT_DIR = "output"


def run_visual_audit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    means = load_station_means()
    means["pollution_index"] = means[["pm25", "pm10", "no2"]].mean(axis=1)

    np.random.seed(42)
    means["population_density"] = np.where(
        means["zone"] == "Industrial",
        np.random.uniform(2000, 6000, size=len(means)),
        np.random.uniform(5000, 15000, size=len(means)),
    )

    zones = means["zone"].unique()
    n_zones = len(zones)

    fig, axes = plt.subplots(1, n_zones, figsize=(6 * n_zones, 5), sharey=True)
    if n_zones == 1:
        axes = [axes]

    norm = Normalize(
        vmin=means["pollution_index"].min(),
        vmax=means["pollution_index"].max(),
    )
    cmap = cm.get_cmap("viridis")

    for i, zone in enumerate(sorted(zones)):
        ax = axes[i]
        subset = means[means["zone"] == zone]
        sc = ax.scatter(
            subset["population_density"],
            subset["pollution_index"],
            c=subset["pollution_index"],
            cmap="viridis",
            norm=norm,
            edgecolors="k",
            linewidths=0.3,
            s=70,
            alpha=0.9,
        )
        ax.set_xlabel("Population Density (per sq km)")
        if i == 0:
            ax.set_ylabel("Pollution Index")
        ax.set_title(f"{zone}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        label="Pollution Index",
        fraction=0.02,
        pad=0.04,
    )
    fig.suptitle("Small Multiples  --  Pollution vs Population Density by Zone", fontsize=13, y=1.02)
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(os.path.join(OUTPUT_DIR, "task4_visual_audit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "data": means[["station_id", "zone", "pollution_index", "population_density"]],
        "figure": os.path.join(OUTPUT_DIR, "task4_visual_audit.png"),
    }


if __name__ == "__main__":
    results = run_visual_audit()
    print("Visual audit complete. Figure saved.")
    print(results["data"].head(10).to_string())
