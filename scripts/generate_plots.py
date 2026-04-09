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
    models_to_keys = {
        'MAPPO (Ours)': 'our_model',
        'PressLight': 'presslight',
        'CoLight': 'colight',
        'Fixed-Time': 'fixed_time'
    }

    throughput, waiting_time, queue_length, travel_time = [], [], [], []
    valid_labels = []

    for label, key in models_to_keys.items():
        if key in bench_data:
            valid_labels.append(label)
            throughput.append(bench_data[key].get('mean_throughput', 0))
            waiting_time.append(bench_data[key].get('mean_waiting_time', 0))
            queue_length.append(bench_data[key].get('mean_queue_length', 0))
            travel_time.append(bench_data[key].get('mean_travel_time', 0))

    x = np.arange(len(valid_labels))
    width = 0.4

    # 1. Throughput
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, throughput, width, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'][:len(valid_labels)])
    ax.set_ylabel('Mean Vehicle Throughput')
    ax.set_title('Phase 1: Multi-Agent Model Throughput Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    plt.savefig('outputs/plots/phase1_throughput_comparison.png')
    plt.close()

    # 2. Waiting Time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, waiting_time, width, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'][:len(valid_labels)])
    ax.set_ylabel('Mean Waiting Time (seconds)')
    ax.set_title('Phase 1: Intersection Waiting Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    plt.savefig('outputs/plots/phase1_waiting_time_comparison.png')
    plt.close()

    # 3. Queue Length
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, queue_length, width, color=['#9467bd', '#17becf', '#e377c2', '#bcbd22'][:len(valid_labels)])
    ax.set_ylabel('Mean Queue Length (Vehicles)')
    ax.set_title('Phase 1: Congestion Queue Length Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    plt.savefig('outputs/plots/phase1_queue_length_comparison.png')
    plt.close()

    # 4. Travel Time (Stopped Vehicles)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, travel_time, width, color=['#ff9896', '#98df8a', '#c5b0d5', '#c49c94'][:len(valid_labels)])
    ax.set_ylabel('Total Stopped Vehicles')
    ax.set_title('Phase 1: Total Stopped Vehicles Matrix')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    plt.savefig('outputs/plots/phase1_travel_time_comparison.png')
    plt.close()

except Exception as e:
    print(f"[Warn] Could not generate Phase 1 charts: {e}")


# --- 2. Phase 2 Anomaly Detection Plots ---
try:
    with open('outputs/phase2/anomaly_eval_summary.json', 'r') as f:
        anom_data = json.load(f)

    methods = anom_data.get("methods", {})
    method_labels = []
    f1_scores = []
    precisions = []
    recalls = []

    for k, v in methods.items():
        if k == "z_score":
            method_labels.append("Z-Score (Baseline)")
        else:
            method_labels.append(v.get("label", k))
        f1_scores.append(v.get("metrics", {}).get("f1", 0))
        precisions.append(v.get("metrics", {}).get("precision", 0))
        recalls.append(v.get("metrics", {}).get("recall", 0))

    x = np.arange(len(method_labels))
    width = 0.25

    # 5. Anomaly Precision vs F1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#9467bd')
    ax.bar(x + width/2, precisions, width, label='Precision', color='#8c564b')
    ax.set_ylabel('Scores (0 - 1.0)')
    ax.set_title('Phase 2: SpatialTemporalAutoencoder Accuracy vs Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('outputs/plots/phase2_anomaly_metrics.png')
    plt.close()
    
    # 6. Anomaly Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, recalls, width*1.5, label='Recall (Crash Identification Rate)', color='#d62728')
    ax.set_ylabel('Recall Rate (0 - 1.0)')
    ax.set_title('Phase 2: True Crash Detection Recall Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    for i, v in enumerate(recalls):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/plots/phase2_anomaly_recall.png')
    plt.close()

    print("[OK] Generated 6 Comprehensive PNG Charts in outputs/plots/")
except Exception as e:
    print(f"[Warn] Could not generate Phase 2 charts: {e}")

