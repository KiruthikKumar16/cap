import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.phase1.graph_builder import TrafficGraphBuilder
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx

def generate_topology():
    print("Generating Network Topology Plot...")
    net_file = "data/raw/grid_6x6.net.xml"
    if not Path(net_file).exists():
        print(f"Error: Network file {net_file} not found.")
        return
        
    builder = TrafficGraphBuilder(net_file)
    
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(builder.graph, seed=42)
    
    nx.draw(builder.graph, pos, 
            with_labels=True, 
            node_color='#334E68', 
            node_size=800, 
            font_size=8, 
            font_color='white',
            edge_color='#cccccc',
            arrows=True,
            width=1.5)
            
    plt.title("6x6 Grid Network Topology (Graph Representation)", fontsize=14, fontweight='bold')
    
    output_path = Path("FAST_VAL_RESULTS/plots/grid_topology.png")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created {output_path}")

if __name__ == "__main__":
    generate_topology()
