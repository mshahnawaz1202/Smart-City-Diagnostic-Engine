import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_load import load_station_means, FEATURES

OUTPUT_DIR = "output"


def run_pca_analysis():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    means = load_station_means()
    X = means[FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    means["PC1"] = X_pca[:, 0]
    means["PC2"] = X_pca[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=FEATURES,
    )

    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    colors = {"Industrial": "#d62728", "Residential": "#1f77b4"}
    for zone in ["Industrial", "Residential"]:
        mask = means["zone"] == zone
        ax.scatter(
            means.loc[mask, "PC1"],
            means.loc[mask, "PC2"],
            c=colors[zone],
            label=zone,
            edgecolors="k",
            linewidths=0.4,
            s=60,
            alpha=0.85,
        )
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    ax.set_title("PCA  --  Industrial vs Residential Zones")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax2 = axes[1]
    x_pos = np.arange(len(FEATURES))
    width = 0.35
    ax2.barh(x_pos - width / 2, loadings["PC1"], height=width, label="PC1", color="#2ca02c")
    ax2.barh(x_pos + width / 2, loadings["PC2"], height=width, label="PC2", color="#ff7f0e")
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(FEATURES)
    ax2.set_xlabel("Loading")
    ax2.set_title("PCA Loadings")
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "task1_pca.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "means": means,
        "loadings": loadings,
        "explained_variance": explained,
        "figure": os.path.join(OUTPUT_DIR, "task1_pca.png"),
    }


if __name__ == "__main__":
    results = run_pca_analysis()
    print("Explained variance:", results["explained_variance"])
    print("\nLoadings:\n", results["loadings"])
