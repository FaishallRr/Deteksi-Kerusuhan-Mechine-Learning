import torch
import numpy as np
from typing import List


class TemporalFeatureExtractor:
    def __init__(self, architecture: str = "x3d_s", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._build_model(architecture)

    def _build_model(self, architecture: str):
        if architecture == "x3d_s":
            import torchvision.models.video as video_models
            model = video_models.x3d_s(weights="DEFAULT")
            model = torch.nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        model = model.to(self.device)
        model.eval()
        return model

    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        tensor = self._preprocess(frames)
        with torch.no_grad():
            features = self.model(tensor)
        return features.cpu().numpy().flatten()

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        processed = []
        for frame in frames:
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            processed.append(frame)

        tensor = np.stack(processed, axis=1)
        tensor = torch.from_numpy(tensor).unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor
