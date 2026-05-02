import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Ensure the plots directory exists
os.makedirs('outputs/plots', exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 4.5))

def draw_box(ax, x, y, width, height, text, facecolor='#eeeeee'):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor='#333333', facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, horizontalalignment='center', 
            verticalalignment='center', fontsize=11, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='#333333'))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha='center', fontsize=10, fontstyle='italic')

# Create boxes
draw_box(ax, 0, 0.2, 2.2, 0.6, "Traffic Environment\n(SUMO Network)", '#ffb3ba')

draw_box(ax, 3.5, 0.2, 2.2, 0.6, "ST-GNN\nGraph Attention & GRU", '#bae1ff')

draw_box(ax, 7, 0.6, 2.2, 0.4, "Centralized Critic\n(Value Function)", '#baffc9')
draw_box(ax, 7, -0.2, 2.2, 0.4, "Local Actor\n(Policy Generator)", '#ffdfba')

draw_box(ax, 10.5, -0.15, 1.5, 0.3, "Traffic Signal\nPhase Action", '#e6e6fa')

# Draw arrows
draw_arrow(ax, 2.2, 0.5, 3.5, 0.5, "Local Observation ($o_i$)") # Env to ST-GNN
draw_arrow(ax, 5.7, 0.6, 7, 0.8, r"Global State ($\mathbf{s}$)") # ST-GNN to Critic
draw_arrow(ax, 5.7, 0.3, 7, 0.0, "State Features") # ST-GNN to Actor

# Action Output
draw_arrow(ax, 9.2, 0.0, 10.5, 0.0, "Action ($a_i$)")

# Feedback loop
ax.annotate("", xy=(1.1, 0.8), xytext=(11.25, 0.15),
            arrowprops=dict(connectionstyle="bar,fraction=0.1,angle=90", arrowstyle="->", lw=2, color='#333333', linestyle='--'))
ax.text(6, 1.25, "Environment Transition & Reward Distribution", ha='center', fontsize=10, fontstyle='italic', color='#555555')


ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.5, 1.5)
ax.axis('off')
plt.title("Decentralized Execution Flow with ST-GNN Mapping", fontsize=16, fontweight='bold', pad=10)
plt.tight_layout()

output_path = os.path.join('outputs', 'plots', 'architecture_flowchart.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Success! Flowchart saved natively to {output_path}")
