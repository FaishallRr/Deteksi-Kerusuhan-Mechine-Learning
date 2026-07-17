"""
Attention-based MIL model with video-level training.
Uses attention to aggregate segment scores into video-level prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMILModel(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_units: int = 512, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_segments, feat_dim = x.shape
        x_flat = x.view(-1, feat_dim)
        att_weights = self.attention(x_flat)
        att_weights = att_weights.view(batch_size, n_segments)
        att_weights = F.softmax(att_weights, dim=1)
        bag_feat = torch.sum(x * att_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(bag_feat)
        return logits.squeeze(-1)

    def forward_segments(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(-1, x.shape[-1]) if x.dim() == 3 else x
        logits = x_flat
        for layer in self.classifier:
            logits = layer(logits)
        return logits

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.forward_segments(x)
        return self.forward(x)

    def predict_bag(self, features: torch.Tensor) -> float:
        if features.dim() == 2:
            features = features.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(features)
            return torch.sigmoid(logits).item()


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, features_list, labels):
        self.features = features_list
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])
