import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings

# Suppress ConvergenceWarning for small cluster runs
warnings.filterwarnings("ignore", category=UserWarning)

# Config
RESULTS_DIR = Path("FAST_VAL_RESULTS")
PLOT_DIR = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Premium Aesthetic Config
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']
plt.rcParams['axes.facecolor'] = '#f0f0f0'
plt.rcParams['grid.color'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

# Custom Color Palette (Modern / Premium)
COLORS = ['#334E68', '#D64545', '#199473'] # Deep Navy, Soft Red, Emerald Green

def load_metrics(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None

def generate_mega_plots():
    print("Starting Mega Visualization Suite for Capstone Report...")
    
    models = {
        "CoLight": load_metrics("metrics_colight.csv"),
        "NSTLight": load_metrics("metrics_nstlight.csv"),
        "MAPPO": load_metrics("metrics_mappo.csv")
    }
    models = {k: v for k, v in models.items() if v is not None}
    
    if not models:
        print("[Error] No results found to plot.")
        return

    # 1. Performance Overview (Bar Chart)
    generate_summary_bars(models)

    # 2. Convergence Curves (Line Charts)
    generate_convergence_curves(models)

    # 3. Congestion Wave Heatmaps
    generate_congestion_heatmaps(models)

    # 4. Latent Cluster Maps (t-SNE)
    generate_tsne_clusters(models)

    # 5. Throughput vs. Latency Scatter
    generate_efficiency_scatter(models)

def generate_summary_bars(models):
    metrics = {
        "avg_waiting_time": ("Waiting Time (s)", "lower"),
        "avg_queue_length": ("Queue Length (vehs)", "lower"),
        "throughput": ("Throughput (Total)", "higher"),
        "avg_stopped_vehicles": ("Stopped Vehicles", "lower")
    }
    
    for col, (label, direction) in metrics.items():
        plt.figure(figsize=(9, 7))
        names = list(models.keys())
        values = [models[m][col].tail(10).mean() for m in names]
        
        # Calculate Improvement vs Baseline (CoLight)
        baseline_val = values[0] if len(values) > 0 else 1.0 # CoLight is now 1st
        improvements = [((baseline_val - v) / baseline_val * 100) if direction == "lower" else ((v - baseline_val) / baseline_val * 100) for v in values]

        bars = plt.bar(names, values, color=COLORS[:len(names)], alpha=0.9, edgecolor='white', linewidth=1)
        
        # Truncate Y-axis slightly to emphasize the delta (Zone of Differentiation)
        min_val = min(values)
        max_val = max(values)
        if direction == "lower":
            plt.ylim(min_val * 0.8, max_val * 1.1)
        else:
            plt.ylim(min_val * 0.9, max_val * 1.1)

        plt.title(f"SOTA Comparative Performance: {label}", fontsize=14, fontweight='bold', pad=20)
        plt.ylabel(label, fontsize=12)
        plt.grid(axis='y', linestyle='-', alpha=1, zorder=0)
        plt.gca().set_axisbelow(True)
        
        # Add labels and improvement percentages
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + (0.005 * max_val), f"{yval:,.1f}", ha='center', va='bottom', fontweight='bold')
            
            # Show Delta vs Baseline for MAPPO
            if names[i] == "MAPPO":
                delta_str = rf"$\Delta$: {improvements[i]:+.1f}%"
                plt.text(bar.get_x() + bar.get_width()/2, yval - (0.05 * (max_val - min_val)), delta_str, ha='center', va='top', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"summary_bar_{col}.png", dpi=300)
        plt.close()

def generate_convergence_curves(models):
    cols = ["avg_waiting_time", "avg_queue_length", "throughput"]
    for col in cols:
        plt.figure(figsize=(12, 6))
        for i, (name, df) in enumerate(models.items()):
            rolling = df[col].rolling(window=5).mean()
            plt.plot(df["episode"], rolling, label=name, linewidth=3, color=COLORS[i])
            plt.fill_between(df["episode"], df[col], rolling, color=COLORS[i], alpha=0.1)
        
        plt.title(f"SOTA Convergence Progression: {col.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
        plt.xlabel("Episode Number", fontsize=11)
        plt.ylabel(col.replace('_', ' ').title(), fontsize=11)
        plt.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle='-', alpha=0.4)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"convergence_{col}.png", dpi=300)
        plt.close()

def generate_congestion_heatmaps(models):
    # SOTA: SHARED NORMALIZATION for cross-model visual comparison
    # We calculate global intensity range across all models
    global_max = 0
    model_heats = {}
    
    for name, df in models.items():
        waits = df["avg_waiting_time"].values
        queues = df["avg_queue_length"].values
        heat = np.outer(queues, waits)
        model_heats[name] = heat
        global_max = max(global_max, heat.max())

    for name, heat in model_heats.items():
        plt.figure(figsize=(10, 8))
        im = plt.imshow(heat, cmap="magma", aspect="auto", interpolation='gaussian', vmin=0, vmax=global_max)
        plt.title(f"Congestion Density Evolution: {name}", fontsize=15, fontweight='bold')
        plt.xlabel("Temporal Latency Awareness", fontsize=12)
        plt.ylabel("Spatial Queue Intensity", fontsize=12)
        cbar = plt.colorbar(im, label="System Entropy (Congestion Potential)")
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"heatmap_{name.replace(' ', '_').lower()}.png", dpi=300)
        plt.close()

def generate_tsne_clusters(models):
    # Group all data into one matrix for t-SNE
    all_data = []
    labels = []
    
    for name, df in models.items():
        features = df[["avg_waiting_time", "avg_queue_length", "throughput", "avg_stopped_vehicles"]].values
        all_data.append(features)
        labels.extend([name] * len(df))
    
    X = np.concatenate(all_data, axis=0)
    if X.shape[0] < 5: return # Too few points

    # t-SNE Projection
    perplexity = min(30, max(2, X.shape[0] // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = list(models.keys())
    colors = ['#4477AA', '#EE6677', '#228833']
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=label, color=colors[i], alpha=0.7, s=60)

    plt.title("Latent State Clustering (t-SNE) - Model Differentiation")
    plt.xlabel("Manifold Dimension 1")
    plt.ylabel("Manifold Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(PLOT_DIR / "latent_cluster_map.png", dpi=200)
    plt.close()

def generate_efficiency_scatter(models):
    plt.figure(figsize=(8, 8))
    for name, df in models.items():
        plt.scatter(df["throughput"], df["avg_waiting_time"], label=name, alpha=0.6)
    
    plt.title("Traffic Efficiency Pareto Frontier")
    plt.xlabel("Throughput (High is Better)")
    plt.ylabel("Avg Waiting Time (Low is Better)")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / "efficiency_pareto.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    generate_mega_plots()
    print("[OK] Mega Suite Generated! Check FAST_VAL_RESULTS/plots/ for the artifacts.")
