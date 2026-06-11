import torch
import torch.nn as nn
import numpy as np


class MILRankingModel(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_units: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def compute_ranking_loss(
        self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 1.0
    ) -> torch.Tensor:
        diff = pos_scores.mean() - neg_scores.mean()
        loss = torch.max(torch.tensor(0.0), margin - diff)
        return loss


class MILBagProcessor:
    def __init__(self, model: MILRankingModel, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_bag(self, features: np.ndarray) -> float:
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(tensor)
        return score.item()
