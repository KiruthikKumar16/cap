import json
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('outputs/plots', exist_ok=True)

# --- 1. Phase 1 Benchmark Comparison Plots ---
try:
    with open('outputs/benchmark_results.json', 'r') as f:
        bench_data = json.load(f)

    labels = ['MAPPO (Ours)', 'PressLight', 'CoLight', 'Fixed-Time']
    # If the JSON doesn't contain fixed_time, we add dummy or filter
    models_to_keys = {
        'MAPPO (Ours)': 'our_model',
        'PressLight': 'presslight',
        'CoLight': 'colight',
        'Fixed-Time': 'fixed_time' # Might not be in JSON
    }

    throughput = []
    waiting_time = []
    valid_labels = []

    for label, key in models_to_keys.items():
        if key in bench_data:
            valid_labels.append(label)
            throughput.append(bench_data[key].get('mean_throughput', 0))
            waiting_time.append(bench_data[key].get('mean_waiting_time', 0))

    x = np.arange(len(valid_labels))
    width = 0.35

    # Plot 1: Throughput
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, throughput, width, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'][:len(valid_labels)])
    ax.set_ylabel('Mean Vehicle Throughput')
    ax.set_title('Phase 1: Multi-Agent Model Throughput Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    plt.savefig('outputs/plots/phase1_throughput_comparison.png')
    plt.close()

    # Plot 2: Waiting Time
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, waiting_time, width, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'][:len(valid_labels)])
    ax.set_ylabel('Mean Waiting Time (seconds)')
    ax.set_title('Phase 1: Intersection Waiting Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    plt.savefig('outputs/plots/phase1_waiting_time_comparison.png')
    plt.close()

    print("[OK] Generated Phase 1 Benchmark Bar Charts in outputs/plots/")
except Exception as e:
    print(f"[Warn] Could not generate Phase 1 charts: {e}")


# --- 2. Phase 2 Anomaly Detection Plots ---
try:
    with open('outputs/phase2/anomaly_eval_summary.json', 'r') as f:
        anom_data = json.load(f)

    methods_data = anom_data.get("methods", {})
    method_labels = []
    f1_scores = []
    precisions = []

    for k, v in methods_data.items():
        method_labels.append(v.get("label", k))
        f1_scores.append(v.get("metrics", {}).get("f1", 0))
        precisions.append(v.get("metrics", {}).get("precision", 0))

    x = np.arange(len(method_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#9467bd')
    rects2 = ax.bar(x + width/2, precisions, width, label='Precision', color='#8c564b')

    ax.set_ylabel('Scores (0 - 1.0)')
    ax.set_title('Phase 2: SpatialTemporalAutoencoder Detection Accuracy vs Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('outputs/plots/phase2_anomaly_metrics.png')
    plt.close()

    print("[OK] Generated Phase 2 Anomaly Score Charts in outputs/plots/")
except Exception as e:
    print(f"[Warn] Could not generate Phase 2 charts: {e}")

print("All plotting operations finished.")
