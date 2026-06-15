import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from PIL import Image


def weather_augment(frame: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    aug_type = rng.integers(0, 5)
    img = frame.astype(np.float32)
    if aug_type == 0:
        factor = 0.5 + 0.6 * rng.random()
        img = np.clip(img * factor, 0, 255)
    elif aug_type == 1:
        factor = 0.5 + 0.6 * rng.random()
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = mean + (img - mean) * factor
        img = np.clip(img, 0, 255)
    elif aug_type == 2:
        noise = rng.normal(0, 10 + 20 * rng.random(), img.shape)
        img = np.clip(img + noise, 0, 255)
    elif aug_type == 3:
        k = int(1 + 4 * rng.random()) * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img.astype(np.uint8)


class TemporalFeatureExtractor:
    def __init__(self, architecture: str = "s3d", device: str = "cpu", augment: bool = False):
        self.device = torch.device(device)
        self.model = self._build_model(architecture)
        self.augment = augment

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

    def extract_batch(self, segment_list: List[List[np.ndarray]]) -> np.ndarray:
        tensors = []
        for frames in segment_list:
            t = self._preprocess_single(frames)
            tensors.append(t)
        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            features = self.model(batch)
        return features.cpu().numpy()

    def _preprocess_single(self, frames: List[np.ndarray]) -> torch.Tensor:
        from torchvision.transforms import functional as F
        processed = []
        for frame in frames:
            if self.augment:
                frame = weather_augment(frame)
            img = Image.fromarray(frame)
            img = F.resize(img, [224, 224])
            img = F.to_tensor(img)
            img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            processed.append(img)
        tensor = torch.stack(processed, dim=1)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        from torchvision.transforms import functional as F

        processed = []
        for frame in frames:
            if self.augment:
                frame = weather_augment(frame)
            img = Image.fromarray(frame)
            img = F.resize(img, [224, 224])
            img = F.to_tensor(img)
            img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            processed.append(img)

        tensor = torch.stack(processed, dim=1)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor
