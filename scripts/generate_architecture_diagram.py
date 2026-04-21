import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_architecture():
    print("Generating Framework Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors
    color_sumo = '#334E68' # Navy
    color_gnn = '#199473'  # Emerald
    color_marl = '#D64545' # Red/Coral
    color_text = '#333333'
    
    # Background
    ax.set_facecolor('#f9f9f9')
    
    # 1. SUMO Environment (Left)
    sumo_box = patches.FancyBboxPatch((0.05, 0.35), 0.2, 0.3, boxstyle="round,pad=0.02", facecolor=color_sumo, alpha=0.9, edgecolor='white')
    ax.add_patch(sumo_box)
    ax.text(0.15, 0.5, "SUMO\nEnvironment\n(TraCI API)", color='white', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 2. ST-GNN Perception (Middle Top)
    gnn_box = patches.FancyBboxPatch((0.35, 0.6), 0.3, 0.25, boxstyle="round,pad=0.02", facecolor=color_gnn, alpha=0.9, edgecolor='white')
    ax.add_patch(gnn_box)
    ax.text(0.5, 0.725, "Spatial-Temporal GNN\n(GAT + GRU)\nTraffic Forecaster", color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 3. MAPPO Controller (Middle Bottom)
    marl_box = patches.FancyBboxPatch((0.35, 0.15), 0.3, 0.25, boxstyle="round,pad=0.02", facecolor=color_marl, alpha=0.9, edgecolor='white')
    ax.add_patch(marl_box)
    ax.text(0.5, 0.275, "Multi-Agent PPO\n(Policy & Value)\nCentralized Training", color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 4. Result/Action (Right)
    action_box = patches.FancyBboxPatch((0.75, 0.35), 0.2, 0.3, boxstyle="round,pad=0.02", facecolor=color_sumo, alpha=0.9, edgecolor='white')
    ax.add_patch(action_box)
    ax.text(0.85, 0.5, "Signal Phase\nOptimization\n(Actions)", color='white', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Arrows
    # Sumo -> GNN (State)
    ax.annotate("", xy=(0.35, 0.7), xytext=(0.25, 0.65), arrowprops=dict(arrowstyle="->", lw=2, color=color_text))
    ax.text(0.3, 0.68, "Telemery\n(Queue/Wait)", fontsize=10, ha='center')
    
    # GNN -> MAPPO (Latent Embedding)
    ax.annotate("", xy=(0.5, 0.4), xytext=(0.5, 0.6), arrowprops=dict(arrowstyle="->", lw=2, color=color_text))
    ax.text(0.52, 0.5, "Latent\nRepresentation", fontsize=10, ha='left')
    
    # MAPPO -> Action (Policy)
    ax.annotate("", xy=(0.75, 0.3), xytext=(0.65, 0.275), arrowprops=dict(arrowstyle="->", lw=2, color=color_text))
    ax.text(0.7, 0.28, "Phase\nSelection", fontsize=10, ha='center')
    
    # Action -> SUMO (Cycle)
    ax.annotate("", xy=(0.15, 0.35), xytext=(0.85, 0.35), arrowprops=dict(arrowstyle="->", lw=2, color=color_text, connectionstyle="arc3,rad=0.3"))
    ax.text(0.5, 0.1, "Simulation Feedback Loop", fontsize=12, style='italic', ha='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title("MAPPO-STGNN Framework Architecture", fontsize=18, fontweight='bold', pad=20)
    
    output_path = Path("FAST_VAL_RESULTS/plots/framework_architecture.png")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created {output_path}")

if __name__ == "__main__":
    generate_architecture()
