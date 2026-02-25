from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset


def build_fully_connected_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    """
    Build a fully-connected directed edge_index for placeholder graph mode.

    Args:
        num_nodes: Number of nodes in the graph.
        device: Torch device.

    Returns:
        edge_index: LongTensor [2, E]
    """
    src, dst = torch.meshgrid(
        torch.arange(num_nodes, dtype=torch.long),
        torch.arange(num_nodes, dtype=torch.long),
        indexing="ij",
    )
    edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
    return edge_index.to(device)


class SyntheticTrafficSequenceDataset(Dataset):
    """
    Synthetic traffic sequence dataset with optional anomaly injection.

    Each sample is a sequence of length H+1:
        x_plus: [H+1, N, F]
    Optionally returns per-node labels indicating anomaly presence at the last step.
    """

    def __init__(
        self,
        num_samples: int,
        horizon: int,
        num_nodes: int,
        num_features: int,
        anomaly_prob: float = 0.0,
        anomaly_scale: float = 0.6,
        anomaly_span: int = 1,
        seed: int = 42,
        return_labels: bool = False,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.anomaly_prob = anomaly_prob
        self.anomaly_scale = anomaly_scale
        self.anomaly_span = max(1, anomaly_span)
        self.seed = seed
        self.return_labels = return_labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        g = torch.Generator()
        g.manual_seed(self.seed + idx)

        # Base normalized features in [0, 1]
        x_plus = torch.rand(self.horizon + 1, self.num_nodes, self.num_features, generator=g)
        labels = torch.zeros(self.num_nodes, dtype=torch.long)

        if self.anomaly_prob > 0:
            mask = torch.rand(self.num_nodes, generator=g) < self.anomaly_prob
            if torch.any(mask):
                labels[mask] = 1
                noise = torch.randn(self.num_nodes, self.num_features, generator=g).abs() * self.anomaly_scale
                # Inject anomalies at the last step (and optionally a short span)
                for t in range(self.anomaly_span):
                    step_idx = -1 - t
                    if abs(step_idx) <= x_plus.shape[0]:
                        x_plus[step_idx, mask] = torch.clamp(
                            x_plus[step_idx, mask] + noise[mask], 0.0, 1.0
                        )

        if self.return_labels:
            return x_plus, labels
        return x_plus
