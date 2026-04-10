import json
import numpy as np
import matplotlib.pyplot as plt

def generate_heatmap(evaluation_file: str, output_file: str):
    """Generates a congestion heatmap from an evaluation file."""
    
    with open(evaluation_file, 'r') as f:
        results = json.load(f)
        
    # Assuming the evaluation file contains per-intersection metrics
    # This is a placeholder for the actual data structure
    # You will need to adapt this to your actual data structure
    metrics = results.get("per_intersection_metrics", {})
    
    grid_size = int(np.sqrt(len(metrics)))
    heatmap_data = np.zeros((grid_size, grid_size))
    
    for intersection_id, intersection_metrics in metrics.items():
        # Assuming intersection_id is in the format "J_x_y"
        parts = intersection_id.split("_")
        x, y = int(parts[1]), int(parts[2])
        heatmap_data[x, y] = intersection_metrics.get("avg_waiting_time", 0)
        
    plt.imshow(heatmap_data, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Average Waiting Time")
    plt.title("Congestion Heatmap")
    plt.savefig(output_file)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate congestion heatmap")
    parser.add_argument(
        "--evaluation-file",
        type=str,
        default="outputs/phase1/real_evaluation_results.json",
        help="Path to evaluation JSON file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/plots/congestion_heatmap.png",
        help="Path to output heatmap image",
    )
    args = parser.parse_args()
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    generate_heatmap(args.evaluation_file, args.output_file)
