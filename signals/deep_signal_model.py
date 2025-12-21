"""Optional deep-learning classifier that consumes *signals* (not pixels).

This keeps the project’s main concept (signals) while allowing a neural model
(typically exported to ONNX) to provide a probabilistic "danger" score.

Design goals:
- Zero behavior change when no model file is present
- No heavy dependencies: uses OpenCV DNN (cv2.dnn) if available
- Graceful failure: if model can't load/infer, returns None

Expected model contract (recommended):
- Input: float32 tensor shape (1, 4) with [ear, perclos, pitch, mar]
- Output: either
  - shape (1, 1): danger probability/logit
  - shape (1, 2): [p_safe, p_danger] or [p_danger, p_safe] (configurable)

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SignalModelConfig:
    model_path: str = "models/signal_danger.onnx"
    # If output is 2-class vector, which index corresponds to danger.
    danger_index: int = 1
    # Threshold for treating model as "confident danger".
    danger_threshold: float = 0.80


class SignalDeepClassifier:
    """Loads an ONNX model (if present) and produces a danger probability."""

    def __init__(self, config: Optional[SignalModelConfig] = None):
        self.config = config or SignalModelConfig()
        self._net = None
        self._session = None
        self._enabled = False
        self._backend = None  # 'cv2' or 'ort'

        model_file = Path(self.config.model_path)
        if not model_file.exists():
            return

        # Try onnxruntime first (best compatibility + speed on CPU)
        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1

            self._session = ort.InferenceSession(
                str(model_file),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._backend = 'ort'
            self._enabled = True
            return
        except Exception:
            pass

        # Fallback to OpenCV
        try:
            import cv2  # type: ignore
            self._net = cv2.dnn.readNetFromONNX(str(model_file))
            self._backend = 'cv2'
            self._enabled = True
        except Exception:
            self._net = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and (self._net is not None or self._session is not None))

    def predict_danger_prob(
        self,
        ear: float,
        perclos: float,
        pitch: float,
        mar: float,
    ) -> Optional[float]:
        """Return p(danger) in [0,1], or None if disabled/unavailable."""

        if not self.enabled:
            return None

        # Feature normalization: the app computes PERCLOS as a percentage (0..100),
        # but the (default) signal model is trained on ratio (0..1).
        perclos_in = float(perclos)
        if perclos_in > 1.0:
            perclos_in = perclos_in / 100.0
        perclos_in = max(0.0, min(1.0, perclos_in))

        x = np.array([[float(ear), perclos_in, float(pitch), float(mar)]], dtype=np.float32)

        try:
            out = None
            if self._backend == 'ort':
                out = self._session.run(None, {self._input_name: x})[0]
            elif self._backend == 'cv2':
                self._net.setInput(x)
                out = self._net.forward()

            if out is None:
                return None

            out = np.asarray(out).reshape(-1)
            if out.size == 1:
                y = float(out[0])
                # If it looks like a logit, squash; if already prob, keep.
                if y < 0.0 or y > 1.0:
                    y = float(_sigmoid(np.array([y], dtype=np.float32))[0])
                return max(0.0, min(1.0, y))

            # 2-class style output
            danger_idx = int(self.config.danger_index)
            danger_idx = max(0, min(out.size - 1, danger_idx))
            vec = out.astype(np.float32)

            # If vector isn't normalized, apply softmax.
            s = float(np.sum(vec))
            if not (0.98 <= s <= 1.02 and np.all(vec >= 0.0) and np.all(vec <= 1.0)):
                ex = np.exp(vec - np.max(vec))
                vec = ex / np.sum(ex)

            return float(vec[danger_idx])

        except Exception:
            return None

    def is_confident_danger(self, danger_prob: Optional[float]) -> bool:
        if danger_prob is None:
            return False
        return danger_prob >= float(self.config.danger_threshold)
