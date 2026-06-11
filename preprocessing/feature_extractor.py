import torch
import torch.nn as nn
import numpy as np
from typing import List
from PIL import Image


class TemporalFeatureExtractor:
    def __init__(self, architecture: str = "s3d", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._build_model(architecture)

    def _build_model(self, architecture: str):
        import torchvision.models.video as video_models

        if architecture == "s3d":
            model = video_models.s3d(weights=video_models.S3D_Weights.KINETICS400_V1)
            model = nn.Sequential(*list(model.children())[:-1])
        elif architecture in ("r3d_18", "r2plus1d_18", "mc3_18"):
            weights_map = {
                "r3d_18": video_models.R3D_18_Weights.KINETICS400_V1,
                "r2plus1d_18": video_models.R2Plus1D_18_Weights.KINETICS400_V1,
                "mc3_18": video_models.MC3_18_Weeds.KINETICS400_V1,
            }
            model = getattr(video_models, architecture)(weights=weights_map[architecture])
            model = nn.Sequential(*list(model.children())[:-1])
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
        from torchvision.transforms import functional as F

        processed = []
        for frame in frames:
            img = Image.fromarray(frame)
            img = F.resize(img, [224, 224])
            img = F.to_tensor(img)
            img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            processed.append(img)

        tensor = torch.stack(processed, dim=1)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor
