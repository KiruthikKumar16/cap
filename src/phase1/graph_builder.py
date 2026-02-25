"""
Graph Construction Module for Traffic Network

This module builds graph representations of traffic networks from SUMO files,
where intersections are nodes and road segments are edges.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

try:
    import sumolib
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    print("Warning: sumolib not available. Install SUMO for full functionality.")


class TrafficGraphBuilder:
    """
    Builds graph representation of traffic network from SUMO network file.
    
    Attributes:
        net_file: Path to SUMO network file (.net.xml)
        intersections: List of intersection IDs
        graph: NetworkX graph representation
        node_to_idx: Mapping from SUMO junction ID to node index
        idx_to_node: Reverse mapping
    """
    
    def __init__(self, net_file: str):
        """
        Initialize graph builder.
        
        Args:
            net_file: Path to SUMO network file
        """
        self.net_file = net_file
        self.intersections: List[str] = []
        self.graph: Optional[nx.DiGraph] = None
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        
        if SUMO_AVAILABLE:
            self._load_network()
        else:
            print("Warning: SUMO not available. Using placeholder graph.")
            self._create_placeholder_graph()
    
    def _load_network(self) -> None:
        """Load SUMO network and extract intersections."""
        if not SUMO_AVAILABLE:
            return
            
        try:
            # Resolve to absolute path so sumolib/urllib does not treat it as a URL (e.g. "data/raw/..." -> scheme "data")
            net_path = Path(self.net_file).resolve()
            if not net_path.exists():
                raise FileNotFoundError(f"Network file not found: {net_path}")
            net = sumolib.net.readNet(str(net_path))
            
            # Get all nodes (junctions) — sumolib Net uses getNodes(), not getJunctions()
            nodes = net.getNodes()
            
            # Filter for signalized intersections (have traffic lights)
            self.intersections = []
            for node in nodes:
                if node.getType() == "traffic_light":
                    self.intersections.append(node.getID())
            
            # If no signalized intersections found, use all nodes (exclude internal junction IDs like :0_0)
            if not self.intersections:
                self.intersections = [n.getID() for n in nodes if not n.getID().startswith(":")]
            
            # Create node index mapping
            self.node_to_idx = {node_id: idx for idx, node_id in enumerate(self.intersections)}
            self.idx_to_node = {idx: node_id for node_id, idx in self.node_to_idx.items()}
            
            # Build graph
            self._build_graph(net)
            
        except Exception as e:
            print(f"Error loading SUMO network: {e}")
            print("Creating placeholder graph instead.")
            self._create_placeholder_graph()
    
    def _build_graph(self, net) -> None:
        """Build NetworkX graph from SUMO network."""
        self.graph = nx.DiGraph()
        
        # Add nodes (intersections)
        for node_id in self.intersections:
            self.graph.add_node(node_id)
        
        # Add edges (road segments connecting intersections); exclude internal edges
        edges = net.getEdges(withInternal=False)
        for edge in edges:
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            
            # Only add edge if both nodes are in our intersection list
            if from_node in self.intersections and to_node in self.intersections:
                if not self.graph.has_edge(from_node, to_node):
                    self.graph.add_edge(from_node, to_node)
    
    def _create_placeholder_graph(self) -> None:
        """Create placeholder graph for testing without SUMO."""
        # Create a simple 2x2 grid
        self.intersections = ["J0", "J1", "J2", "J3"]
        self.node_to_idx = {node_id: idx for idx, node_id in enumerate(self.intersections)}
        self.idx_to_node = {idx: node_id for node_id, idx in self.node_to_idx.items()}
        
        self.graph = nx.DiGraph()
        for node_id in self.intersections:
            self.graph.add_node(node_id)
        
        # 2x2 grid connections
        self.graph.add_edge("J0", "J1")
        self.graph.add_edge("J0", "J2")
        self.graph.add_edge("J1", "J0")
        self.graph.add_edge("J1", "J3")
        self.graph.add_edge("J2", "J0")
        self.graph.add_edge("J2", "J3")
        self.graph.add_edge("J3", "J1")
        self.graph.add_edge("J3", "J2")
    
    def get_edge_index(self) -> torch.Tensor:
        """
        Get edge index tensor for PyTorch Geometric.
        
        Returns:
            Edge index tensor of shape [2, num_edges]
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call _load_network() first.")
        
        edge_list = []
        for u, v in self.graph.edges():
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            edge_list.append([u_idx, v_idx])
        
        if not edge_list:
            # Self-loops if no edges
            edge_list = [[i, i] for i in range(len(self.intersections))]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index
    
    def get_num_nodes(self) -> int:
        """Get number of nodes (intersections) in the graph."""
        return len(self.intersections)
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get adjacency matrix representation.
        
        Returns:
            Adjacency matrix of shape [num_nodes, num_nodes]
        """
        if self.graph is None:
            raise ValueError("Graph not built.")
        
        adj = nx.adjacency_matrix(self.graph, nodelist=self.intersections, dtype=np.float32)
        return adj.toarray()
    
    def get_node_info(self) -> Dict[str, Dict]:
        """
        Get information about each node.
        
        Returns:
            Dictionary mapping node_id to node information
        """
        info = {}
        for node_id in self.intersections:
            info[node_id] = {
                "index": self.node_to_idx[node_id],
                "neighbors": list(self.graph.neighbors(node_id)) if self.graph else []
            }
        return info
    
    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the graph (requires matplotlib).
        
        Args:
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.graph is None:
                print("Graph not built.")
                return
            
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(self.graph)
            nx.draw(self.graph, pos, with_labels=True, node_color='lightblue',
                   node_size=1000, font_size=10, arrows=True)
            plt.title("Traffic Network Graph")
            
            if save_path:
                plt.savefig(save_path)
                print(f"Graph saved to {save_path}")
            else:
                plt.show()
        except ImportError:
            print("matplotlib not available for visualization")


def build_traffic_graph(net_file: str) -> Tuple[TrafficGraphBuilder, torch.Tensor]:
    """
    Convenience function to build traffic graph and get edge index.
    
    Args:
        net_file: Path to SUMO network file
        
    Returns:
        Tuple of (graph_builder, edge_index)
    """
    builder = TrafficGraphBuilder(net_file)
    edge_index = builder.get_edge_index()
    return builder, edge_index


