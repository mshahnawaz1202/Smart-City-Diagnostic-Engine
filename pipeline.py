import os
import sys

from task_01_pca import run_pca_analysis
from task2_temporal import run_temporal_analysis
from task3_distribution import run_distribution_analysis
from task4_visual_audit import run_visual_audit


def main():
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("SMART CITY DIAGNOSTIC ENGINE")
    print("=" * 60)

    print("\n[1/4] Running PCA Dimensionality Reduction ...")
    pca_results = run_pca_analysis()
    print(f"  Explained variance: PC1={pca_results['explained_variance'][0]*100:.1f}%  PC2={pca_results['explained_variance'][1]*100:.1f}%")
    print(f"  Saved: {pca_results['figure']}")

    print("\n[2/4] Running Temporal Analysis ...")
    temporal_results = run_temporal_analysis()
    print(f"  Dominant pattern: {temporal_results['dominant_pattern']}")
    print(f"  Saved: {temporal_results['figure']}")

    print("\n[3/4] Running Distribution Analysis ...")
    dist_results = run_distribution_analysis()
    print(f"  99th percentile PM2.5: {dist_results['p99']:.2f} ug/m3")
    print(f"  Extreme events: {dist_results['extreme_count']} ({dist_results['extreme_probability']*100:.4f}%)")
    print(f"  Saved: {dist_results['figure']}")

    print("\n[4/4] Running Visual Integrity Audit ...")
    audit_results = run_visual_audit()
    print(f"  Saved: {audit_results['figure']}")

    print("\n" + "=" * 60)
    print("Pipeline complete. All outputs saved to ./output/")
    print("=" * 60)

    return {
        "pca": pca_results,
        "temporal": temporal_results,
        "distribution": dist_results,
        "audit": audit_results,
    }


if __name__ == "__main__":
    main()
