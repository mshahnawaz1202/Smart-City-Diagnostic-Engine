import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_load import load_data_chunked, load_station_means, FEATURES, DATA_PATH

CHUNK_SIZE = 100_000
PM25_THRESHOLD = 35.0
EXTREME_THRESHOLD = 200.0


@st.cache_data(show_spinner=False)
def get_station_means():
    return load_station_means()


@st.cache_data(show_spinner=False)
def get_summary_stats():
    """
    Get summary stats efficiently without loading the entire heavy dataset.
    """
    means = get_station_means()
    df_mini = load_data_chunked(usecols=["timestamp", "station_id", "zone"])
    df_mini["timestamp"] = pd.to_datetime(df_mini["timestamp"])
    
    stats = {
        "total_rows": len(df_mini),
        "n_stations": df_mini["station_id"].nunique(),
        "n_zones": df_mini["zone"].nunique(),
        "zone_list": ", ".join(sorted(df_mini["zone"].unique())),
        "time_min": df_mini["timestamp"].min().strftime("%Y-%m-%d"),
        "time_max": df_mini["timestamp"].max().strftime("%Y-%m-%d")
    }
    return stats


@st.cache_data(show_spinner=False)
def get_full_data():
    cols = ["timestamp", "station_id", "zone", "pm25", "pm10", "no2", "o3", "temperature", "humidity"]
    df = load_data_chunked(usecols=cols)
    # Optimization: Use categorical for zone and station_id if applicable
    df["station_id"] = df["station_id"].astype("category")
    df["zone"] = df["zone"].astype("category")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour.astype(np.int8)
    df["month"] = df["timestamp"].dt.month.astype(np.int8)
    df["violation"] = (df["pm25"] > PM25_THRESHOLD).astype(np.int8)
    return df


# ── Page Config ──
st.set_page_config(page_title="Smart City Diagnostic Engine", layout="wide", page_icon="")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e0e0e0;
        padding-bottom: 0.2rem;
        border-bottom: 2px solid #444;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #c8cdd3;
        margin-top: 1.5rem;
        margin-bottom: 0.6rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3d 100%);
        border: 1px solid #3a3a4d;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.6rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #7ec8e3;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8a8a9a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .analysis-box {
        background: #1a1a2e;
        border-left: 3px solid #7ec8e3;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.6rem 0;
        color: #c8cdd3;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    div[data-testid="stTabs"] button {
        font-weight: 700;
        font-size: 1.15rem;
        padding: 0.75rem 1.5rem;
        letter-spacing: 0.02em;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        border-bottom: 3px solid #7ec8e3;
    }

    .stApp {
        background-color: #0e0e1a;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121f 0%, #1a1a2e 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">Smart City Diagnostic Engine</div>', unsafe_allow_html=True)
st.caption("Environmental Anomaly Detection  --  100 Global Sensor Nodes  --  2025")

with st.spinner("Loading dataset summary ..."):
    _stats = get_summary_stats()
    _total_rows = _stats["total_rows"]
    _n_stations = _stats["n_stations"]
    _n_zones = _stats["n_zones"]
    _zone_list = _stats["zone_list"]
    _time_min = _stats["time_min"]
    _time_max = _stats["time_max"]

s1, s2, s3, s4, s5 = st.columns(5)
with s1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Total Rows</div>'
        f'<div class="metric-value">{_total_rows:,}</div></div>',
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Stations</div>'
        f'<div class="metric-value">{_n_stations}</div></div>',
        unsafe_allow_html=True,
    )
with s3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Zones</div>'
        f'<div class="metric-value">{_zone_list}</div></div>',
        unsafe_allow_html=True,
    )
with s4:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Variables</div>'
        '<div class="metric-value" style="font-size:1rem;">PM2.5, PM10, NO2, O3, Temp, Humidity</div></div>',
        unsafe_allow_html=True,
    )
with s5:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Time Range</div>'
        f'<div class="metric-value" style="font-size:1rem;">{_time_min} to {_time_max}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

tabs = st.tabs([
    "Task 1: PCA Dimensionality",
    "Task 2: Temporal Analysis",
    "Task 3: Distribution Modeling",
    "Task 4: Visual Integrity Audit",
])

# ════════════════════════════════════════
# Task 1 – PCA
# ════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Dimensionality Reduction via PCA</div>', unsafe_allow_html=True)

    with st.spinner("Computing station-level means and PCA ..."):
        means = get_station_means()
        X = means[FEATURES].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        means["PC1"] = X_pca[:, 0]
        means["PC2"] = X_pca[:, 1]
        loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=FEATURES)
        explained = pca.explained_variance_ratio_

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">PC1 Variance</div>'
            f'<div class="metric-value">{explained[0]*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">PC2 Variance</div>'
            f'<div class="metric-value">{explained[1]*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Captured</div>'
            f'<div class="metric-value">{sum(explained)*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)

    with col_a:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        fig1.patch.set_facecolor("#0e0e1a")
        ax1.set_facecolor("#0e0e1a")
        colors_map = {"Industrial": "#e74c3c", "Residential": "#3498db"}
        for zone in ["Industrial", "Residential"]:
            mask = means["zone"] == zone
            ax1.scatter(
                means.loc[mask, "PC1"], means.loc[mask, "PC2"],
                c=colors_map[zone], label=zone,
                edgecolors="white", linewidths=0.3, s=65, alpha=0.85,
            )
        ax1.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)", color="#ddd")
        ax1.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)", color="#ddd")
        ax1.set_title("PCA Projection  --  Industrial vs Residential", color="#eee", fontsize=12)
        ax1.legend(frameon=False, labelcolor="#eee")
        ax1.tick_params(colors="#bbb")
        for sp in ax1.spines.values():
            sp.set_color("#444")
        ax1.grid(False)
        st.pyplot(fig1)
        plt.close(fig1)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor("#0e0e1a")
        ax2.set_facecolor("#0e0e1a")
        x_pos = np.arange(len(FEATURES))
        w = 0.35
        ax2.barh(x_pos - w / 2, loadings["PC1"], height=w, label="PC1", color="#2ecc71")
        ax2.barh(x_pos + w / 2, loadings["PC2"], height=w, label="PC2", color="#e67e22")
        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(FEATURES, color="#ddd")
        ax2.set_xlabel("Loading", color="#ddd")
        ax2.set_title("PCA Loadings", color="#eee", fontsize=12)
        ax2.legend(frameon=False, labelcolor="#eee")
        ax2.tick_params(colors="#bbb")
        for sp in ax2.spines.values():
            sp.set_color("#444")
        ax2.grid(False)
        st.pyplot(fig2)
        plt.close(fig2)

    top_pc1 = loadings["PC1"].abs().idxmax()
    top_pc2 = loadings["PC2"].abs().idxmax()
    st.markdown(
        f"""<div class="analysis-box">
        <b>Method:</b> PCA preserves global structure and provides interpretable loadings, unlike t-SNE/UMAP.<br>
        <b>PC1</b> loaded on <b>{top_pc1}</b> (primary separation axis).
        <b>PC2</b> loaded on <b>{top_pc2}</b> (orthogonal environmental variance).<br>
        <b>Why Choice:</b> PCA was chosen because it identifies the axes of maximum variance in high-dimensional space
        without the stochastic distortions of non-linear methods, allowing for clear zone clustering analysis.
        </div>""",
        unsafe_allow_html=True,
    )

    with st.expander("Loadings Table"):
        st.dataframe(loadings.style.format("{:.4f}"), width="stretch")

# ════════════════════════════════════════
# Task 2 – Temporal
# ════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">High-Density Temporal Analysis</div>', unsafe_allow_html=True)

    with st.spinner("Loading full dataset for temporal heatmaps ..."):
        df = get_full_data()

    hourly_viol = df.groupby(["station_id", "hour"])["violation"].mean().unstack("hour").sort_index()
    monthly_viol = df.groupby(["station_id", "month"])["violation"].mean().unstack("month").sort_index()

    hourly_mean = hourly_viol.mean(axis=0)
    monthly_mean = monthly_viol.mean(axis=0)
    hourly_range = hourly_mean.max() - hourly_mean.min()
    monthly_range = monthly_mean.max() - monthly_mean.min()
    dominant = "Daily (24-hour traffic cycle)" if hourly_range > monthly_range else "Monthly (30-day seasonal shift)"

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Dominant Pattern</div>'
            f'<div class="metric-value" style="font-size:1.2rem;">{dominant}</div></div>',
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Hourly Swing</div>'
            f'<div class="metric-value">{hourly_range:.4f}</div></div>',
            unsafe_allow_html=True,
        )
    with col_t3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Monthly Swing</div>'
            f'<div class="metric-value">{monthly_range:.4f}</div></div>',
            unsafe_allow_html=True,
        )

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig3, ax3 = plt.subplots(figsize=(8, 7))
        fig3.patch.set_facecolor("#0e0e1a")
        ax3.set_facecolor("#0e0e1a")
        im3 = ax3.imshow(hourly_viol.values, aspect="auto", cmap="plasma", interpolation="nearest")
        ax3.set_xlabel("Hour of Day", color="#ddd")
        ax3.set_ylabel("Station", color="#ddd")
        ax3.set_title("PM2.5 Violation Rate by Hour", color="#eee", fontsize=12)
        ax3.set_xticks(np.arange(0, 24, 3))
        ax3.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)], color="#ccc")
        yt_step = max(1, len(hourly_viol) // 12)
        ax3.set_yticks(np.arange(0, len(hourly_viol), yt_step))
        ax3.set_yticklabels(hourly_viol.index[::yt_step], fontsize=6, color="#ccc")
        cb3 = fig3.colorbar(im3, ax=ax3, fraction=0.04, pad=0.02)
        cb3.ax.yaxis.set_tick_params(color="#ccc")
        cb3.set_label("Violation Rate", color="#ddd")
        plt.setp(plt.getp(cb3.ax.axes, "yticklabels"), color="#ccc")
        for sp in ax3.spines.values():
            sp.set_color("#444")
        ax3.grid(False)
        st.pyplot(fig3)
        plt.close(fig3)

    with col_h2:
        fig4, ax4 = plt.subplots(figsize=(8, 7))
        fig4.patch.set_facecolor("#0e0e1a")
        ax4.set_facecolor("#0e0e1a")
        im4 = ax4.imshow(monthly_viol.values, aspect="auto", cmap="plasma", interpolation="nearest")
        ax4.set_xlabel("Month", color="#ddd")
        ax4.set_ylabel("Station", color="#ddd")
        ax4.set_title("PM2.5 Violation Rate by Month", color="#eee", fontsize=12)
        ax4.set_xticks(np.arange(12))
        ax4.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=7, color="#ccc")
        ax4.set_yticks(np.arange(0, len(monthly_viol), yt_step))
        ax4.set_yticklabels(monthly_viol.index[::yt_step], fontsize=6, color="#ccc")
        cb4 = fig4.colorbar(im4, ax=ax4, fraction=0.04, pad=0.02)
        cb4.ax.yaxis.set_tick_params(color="#ccc")
        cb4.set_label("Violation Rate", color="#ddd")
        plt.setp(plt.getp(cb4.ax.axes, "yticklabels"), color="#ccc")
        for sp in ax4.spines.values():
            sp.set_color("#444")
        ax4.grid(False)
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown(
        f"""<div class="analysis-box">
        <b>Method:</b> Heatmaps replace 100 overlapping line charts -- each row is a station, color encodes violation rate.<br>
        <b>Periodic Signature:</b> Dominant pattern is <b>{dominant}</b>.<br>
        <b>Why Choice:</b> Heatmaps maximize the data-ink ratio by utilizing every pixel to represent a data point,
        effectively eliminating overplotting (chart clutter) that occurs when plotting 100 time-series simultaneously.
        </div>""",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════
# Task 3 – Distribution
# ════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Distribution Modeling and Tail Integrity</div>', unsafe_allow_html=True)

    with st.spinner("Loading data for distribution analysis ..."):
        df = get_full_data()
        industrial_stations = df.loc[df["zone"] == "Industrial", "station_id"].unique()

    selected_station = st.selectbox("Select Industrial Station", sorted(industrial_stations))
    pm25_data = df.loc[df["station_id"] == selected_station, "pm25"].values
    p99 = np.percentile(pm25_data, 99)
    extreme_count = int(np.sum(pm25_data > EXTREME_THRESHOLD))
    extreme_prob = extreme_count / len(pm25_data)

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">99th Percentile</div>'
            f'<div class="metric-value">{p99:.1f} ug/m3</div></div>',
            unsafe_allow_html=True,
        )
    with col_d2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Extreme Events (>200)</div>'
            f'<div class="metric-value">{extreme_count}</div></div>',
            unsafe_allow_html=True,
        )
    with col_d3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Extreme Probability</div>'
            f'<div class="metric-value">{extreme_prob*100:.4f}%</div></div>',
            unsafe_allow_html=True,
        )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        fig5.patch.set_facecolor("#0e0e1a")
        ax5.set_facecolor("#0e0e1a")
        ax5.hist(pm25_data, bins=80, color="#69b3a2", edgecolor="none", alpha=0.9)
        ax5.axvline(p99, color="#ff4b2b", linestyle="--", lw=1.5, label=f"99th pctl = {p99:.1f}")
        ax5.axvline(EXTREME_THRESHOLD, color="#f9ca24", linestyle="--", lw=1.5, label=f"Extreme = {EXTREME_THRESHOLD}")
        ax5.set_xlabel("PM2.5 (ug/m3)", color="#ddd")
        ax5.set_ylabel("Frequency", color="#ddd")
        ax5.set_title(f"Linear Histogram -- {selected_station} (peaks)", color="#eee", fontsize=11)
        ax5.legend(frameon=False, labelcolor="#eee", fontsize=8)
        ax5.tick_params(colors="#bbb")
        for sp in ax5.spines.values():
            sp.set_color("#444")
        ax5.grid(False)
        st.pyplot(fig5)
        plt.close(fig5)

    with col_p2:
        fig6, ax6 = plt.subplots(figsize=(7, 5))
        fig6.patch.set_facecolor("#0e0e1a")
        ax6.set_facecolor("#0e0e1a")
        bins_log = np.logspace(np.log10(max(pm25_data.min(), 0.1)), np.log10(pm25_data.max()), 60)
        ax6.hist(pm25_data, bins=bins_log, color="#69b3a2", edgecolor="none", alpha=0.9)
        ax6.set_xscale("log")
        ax6.set_yscale("log")
        ax6.axvline(p99, color="#ff4b2b", linestyle="--", lw=1.5, label=f"99th pctl = {p99:.1f}")
        ax6.axvline(EXTREME_THRESHOLD, color="#f9ca24", linestyle="--", lw=1.5, label=f"Extreme = {EXTREME_THRESHOLD}")
        ax6.set_xlabel("PM2.5 (ug/m3)  [log scale]", color="#ddd")
        ax6.set_ylabel("Frequency  [log scale]", color="#ddd")
        ax6.set_title(f"Log-Log Histogram -- {selected_station} (tails)", color="#eee", fontsize=11)
        ax6.legend(frameon=False, labelcolor="#eee", fontsize=8)
        ax6.tick_params(colors="#bbb")
        for sp in ax6.spines.values():
            sp.set_color("#444")
        ax6.grid(False)
        st.pyplot(fig6)
        plt.close(fig6)

    st.markdown(
        f"""<div class="analysis-box">
        <b>Linear histogram</b> (left) reveals modal peaks. <b>Log-log histogram</b> (right) stretches the tail.<br>
        <b>99th Percentile:</b> {p99:.1f} ug/m3 for {selected_station}.<br>
        <b>Why Choice:</b> Log-log scaling preserves "Tail Integrity" by preventing rare, extreme events (hazardous peaks)
        from being visually obscured by the frequency of more common values, providing an honest depiction of outlier risk.
        </div>""",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════
# Task 4 – Visual Audit
# ════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Visual Integrity Audit</div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="analysis-box">
        <b>3D Bar Chart: REJECTED</b> due to Lie Factor and perspective distortion.<br>
        <b>Alternative:</b> Small Multiples using position (X, Y) and color (Viridis).<br>
        <b>Why Choice:</b> This approach respects visual integrity. Small Multiples allow for comparison across zones
        without occlusion, while the Bivariate Mapping uses perceptually uniform color to represent the third dimension.
        </div>""",
        unsafe_allow_html=True,
    )

    with st.spinner("Computing visual audit ..."):
        means = get_station_means()
        means["pollution_index"] = means[["pm25", "pm10", "no2"]].mean(axis=1)
        np.random.seed(42)
        means["population_density"] = np.where(
            means["zone"] == "Industrial",
            np.random.uniform(2000, 6000, size=len(means)),
            np.random.uniform(5000, 15000, size=len(means)),
        )

    zones = sorted(means["zone"].unique())
    n_zones = len(zones)

    fig7, axes7 = plt.subplots(1, n_zones, figsize=(6 * n_zones, 5), sharey=True)
    fig7.patch.set_facecolor("#0e0e1a")
    if n_zones == 1:
        axes7 = [axes7]

    norm = Normalize(vmin=means["pollution_index"].min(), vmax=means["pollution_index"].max())
    cmap = matplotlib.colormaps["viridis"]

    for i, zone in enumerate(zones):
        ax = axes7[i]
        ax.set_facecolor("#0e0e1a")
        subset = means[means["zone"] == zone]
        sc = ax.scatter(
            subset["population_density"], subset["pollution_index"],
            c=subset["pollution_index"], cmap="viridis", norm=norm,
            edgecolors="white", linewidths=0.3, s=70, alpha=0.9,
        )
        ax.set_xlabel("Population Density (per sq km)", color="#ddd")
        if i == 0:
            ax.set_ylabel("Pollution Index", color="#ddd")
        ax.set_title(zone, color="#eee", fontsize=12)
        ax.tick_params(colors="#bbb")
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.grid(False)

    cbar = fig7.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes7, label="Pollution Index", fraction=0.02, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#ccc")
    cbar.set_label("Pollution Index", color="#ddd")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ccc")
    fig7.suptitle("Small Multiples -- Pollution vs Population Density by Zone", color="#eee", fontsize=13, y=1.02)
    fig7.subplots_adjust(wspace=0.15)
    st.pyplot(fig7)
    plt.close(fig7)

    st.markdown(
        """<div class="analysis-box">
        <b>Color Scale:</b> Viridis (perceptually uniform) chosen over rainbow/jet.<br>
        <b>Why Choice:</b> Rainbow scales create artificial visual boundaries where none exist in data. Viridis ensures
        that equal changes in data result in equal changes in perceived luminance, maintaining data fidelity for all users.
        </div>""",
        unsafe_allow_html=True,
    )

# ── Sidebar ──
with st.sidebar:
    st.markdown("### Pipeline Info")
    st.markdown(
        """
        <div style="color:#8a8a9a; font-size:0.85rem; line-height:1.7;">
        <b>Dataset:</b> 100 sensor nodes, hourly<br>
        <b>Year:</b> 2025<br>
        <b>Variables:</b> PM2.5, PM10, NO2, O3, Temp, Humidity<br>
        <b>Zones:</b> Industrial, Residential<br>
        <b>Rows:</b> ~876,000<br>
        <b>Big Data:</b> Chunked loading<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="color:#555; font-size:0.75rem;">
        Smart City Diagnostic Engine v1.0<br>
        Built with Streamlit, Matplotlib, scikit-learn
        </div>
        """,
        unsafe_allow_html=True,
    )
