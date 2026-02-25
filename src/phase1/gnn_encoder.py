"""
GNN Encoder Module for Traffic Network

Encodes spatial dependencies between intersections using Graph Neural Networks.
Supports both GCN (Graph Convolutional Network) and GAT (Graph Attention Network).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class MLPEncoder(nn.Module):
    """
    MLP-based state encoder for ablation (no graph structure).

    Same interface as TrafficGNNEncoder: forward(x, edge_index) -> [N, out_dim].
    edge_index is ignored. Used for ablation study: train DQN without GNN.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        mods = []
        last = in_dim
        for _ in range(max(0, num_layers - 1)):
            mods.append(nn.Linear(last, hidden_dim))
            mods.append(nn.ReLU())
            mods.append(nn.Dropout(dropout))
            last = hidden_dim
        mods.append(nn.Linear(last, out_dim))
        self.mlp = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [num_nodes, in_dim]; edge_index ignored
        return self.mlp(x)


class TrafficGNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for traffic networks.
    
    Encodes node features (intersection states) into embeddings that capture
    spatial dependencies between intersections.
    
    Supports:
    - GCN (Graph Convolutional Network)
    - GAT (Graph Attention Network)
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        gnn_type: str = "gat",
        gat_heads: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize GNN encoder.
        
        Args:
            in_dim: Input feature dimension (12 for traffic features)
            hidden_dim: Hidden layer dimension (64)
            out_dim: Output embedding dimension (32)
            num_layers: Number of GNN layers (2)
            gnn_type: Type of GNN ("gcn" or "gat")
            gat_heads: Number of attention heads for GAT (2)
            dropout: Dropout rate (0.1)
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.gat_heads = gat_heads
        self.dropout = dropout
        
        if self.gnn_type not in ["gcn", "gat"]:
            raise ValueError(f"gnn_type must be 'gcn' or 'gat', got {gnn_type}")
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # First layer
        if self.gnn_type == "gat":
            self.layers.append(GATConv(in_dim, hidden_dim, heads=gat_heads, dropout=dropout))
            current_dim = hidden_dim * gat_heads
        else:  # gcn
            self.layers.append(GCNConv(in_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if self.gnn_type == "gat":
                self.layers.append(GATConv(current_dim, hidden_dim, heads=gat_heads, dropout=dropout))
                current_dim = hidden_dim * gat_heads
            else:  # gcn
                self.layers.append(GCNConv(current_dim, hidden_dim))
                current_dim = hidden_dim
        
        # Output layer
        if num_layers > 1:
            if self.gnn_type == "gat":
                # Final layer: reduce from multi-head to single output
                self.layers.append(GATConv(current_dim, out_dim, heads=1, dropout=dropout, concat=False))
            else:  # gcn
                self.layers.append(GCNConv(current_dim, out_dim))
        else:
            # Single layer case
            if self.gnn_type == "gat":
                self.layers.append(GATConv(in_dim, out_dim, heads=1, dropout=dropout, concat=False))
            else:
                self.layers.append(GCNConv(in_dim, out_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN encoder.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge index [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_dim]
        """
        # Apply GNN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x
    
    def get_output_dim(self) -> int:
        """Get output dimension of embeddings."""
        return self.out_dim


class FlattenGNNWrapper(nn.Module):
    """
    Wrapper to flatten GNN embeddings for RL agent.
    
    RL agents typically expect flat observation vectors.
    This wrapper flattens node embeddings into a single vector.
    """
    
    def __init__(self, gnn_encoder: TrafficGNNEncoder, num_nodes: int):
        """
        Initialize flatten wrapper.
        
        Args:
            gnn_encoder: GNN encoder to wrap
            num_nodes: Number of nodes in the graph
        """
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.num_nodes = num_nodes
        self.output_dim = gnn_encoder.out_dim * num_nodes
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode and flatten.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge index [2, num_edges]
            
        Returns:
            Flattened embeddings [num_nodes * out_dim]
        """
        embeddings = self.gnn_encoder(x, edge_index)
        flattened = embeddings.view(-1)  # Flatten to 1D
        return flattened


