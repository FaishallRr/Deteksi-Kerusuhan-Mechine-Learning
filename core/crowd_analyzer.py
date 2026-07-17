import cv2
import numpy as np
from collections import deque


class CrowdAnalyzer:
    def __init__(self, flow_window=5):
        self._prev_gray = None
        self._flow_history = deque(maxlen=flow_window)
        self._magnitude_history = deque(maxlen=flow_window)

    def compute_optical_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
        if self._prev_gray is None:
            self._prev_gray = gray
            h, w = gray.shape
            return np.zeros((h, w, 2), dtype=np.float32), 0.0, 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self._prev_gray = gray
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        self._flow_history.append(flow)
        self._magnitude_history.append(mag)
        return flow, mag, ang

    def motion_entropy(self, ang, mag, bins=16):
        mask = mag > 2.0
        if not np.any(mask):
            return 0.0
        ang_bins = np.floor(ang / (2 * np.pi / bins)).astype(np.int32)
        ang_bins[~mask] = -1
        ang_bins = ang_bins % bins
        hist = np.bincount(ang_bins[mask], minlength=bins).astype(np.float64)
        hist /= hist.sum() + 1e-10
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(bins)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def motion_magnitude_stats(self, mag):
        mask = mag > 2.0
        if not np.any(mask):
            return 0.0, 0.0
        return float(np.mean(mag[mask])), float(np.std(mag[mask]))

    def compute_chaos_index(self, frame, n_p, running_ratio, close_pair_ratio):
        flow, mag, ang = self.compute_optical_flow(frame)
        h, w = frame.shape[:2]
        entropy = self.motion_entropy(ang, mag, bins=16)
        avg_mag, std_mag = self.motion_magnitude_stats(mag)
        avg_mag_norm = min(avg_mag / 30.0, 1.0)
        std_mag_norm = min(std_mag / 20.0, 1.0)
        density = min(n_p * 1000 / (h * w), 1.0) if n_p >= 3 else 0.0
        chaos = (
            entropy * 0.30 + avg_mag_norm * 0.15 + std_mag_norm * 0.10
            + running_ratio * 0.30 + min(close_pair_ratio, 1.0) * 0.10
            + density * 0.05
        )
        return min(max(chaos, 0.0), 1.0), {
            "entropy": round(entropy, 3),
            "avg_mag": round(avg_mag, 2),
            "std_mag": round(std_mag, 2),
            "running_ratio": round(running_ratio, 3),
            "close_pair_ratio": round(close_pair_ratio, 3),
            "density": round(density, 3),
        }

    def classify_scene(self, n_p, chaos_index, running_ratio):
        if n_p < 3:
            return "normal"
        if chaos_index >= 0.55 and running_ratio > 0.15:
            return "demo_rusuh"
        if chaos_index >= 0.65:
            return "demo_rusuh"
        if chaos_index >= 0.40:
            return "demo_damai"
        if chaos_index >= 0.20:
            return "demo_damai"
        return "normal"

    def reset(self):
        self._prev_gray = None
        self._flow_history.clear()
        self._magnitude_history.clear()
