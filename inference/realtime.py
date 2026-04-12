"""Real-time ASL recognition from webcam using a trained SLR checkpoint.

Uses a sliding window buffer of seq_len frames. MediaPipe Holistic runs
per-frame; keypoints are accumulated, normalized, and fed to the model
once the buffer is full. Prediction is displayed as an OpenCV overlay.

Usage:
    python inference/realtime.py \\
        --checkpoint checkpoints/lstm/best.pt \\
        --model_type lstm \\
        --label_json data/splits/label_to_idx.json

    python inference/realtime.py \\
        --checkpoint checkpoints/transformer/best.pt \\
        --model_type transformer \\
        --label_json data/splits/label_to_idx.json \\
        --camera_id 0 \\
        --confidence_threshold 0.5
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp_lib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.model_config import ModelConfig
from models.slr_model_lstm import SLRModelLSTM
from models.slr_model_transformer import SLRModelTransformer
from preprocessing.extract_keypoints import (
    extract_frame_landmarks,
    normalize_frame,
    FEATURE_DIM,
    _N_HAND,
    _N_POSE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("realtime")


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame: np.ndarray,
    prediction: Optional[str],
    confidence: float,
    fps: float,
    buffer_fill: int,
    seq_len: int,
    top3: List[Tuple[str, float]],
) -> np.ndarray:
    """Draw prediction overlay onto a BGR frame.

    Args:
        frame: BGR uint8 numpy array.
        prediction: Predicted gloss string or None if buffer not full yet.
        confidence: Softmax confidence for the top prediction.
        fps: Current inference + capture FPS.
        buffer_fill: Number of frames currently in the sliding window buffer.
        seq_len: Required buffer size before inference runs.
        top3: Top-3 predictions as (gloss, probability) tuples.

    Returns:
        Annotated BGR frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent black bar at top
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Main prediction
    if prediction is not None:
        color = (0, 255, 0) if confidence > 0.5 else (0, 200, 255)
        cv2.putText(
            frame,
            f"{prediction.upper()}  {confidence:.0%}",
            (12, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            2,
            cv2.LINE_AA,
        )
    else:
        fill_pct = buffer_fill / max(seq_len, 1)
        bar_w = int(w * fill_pct)
        cv2.rectangle(frame, (0, 60), (bar_w, 75), (0, 200, 255), -1)
        cv2.putText(
            frame,
            f"Buffering... {buffer_fill}/{seq_len}",
            (12, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Top-3 secondary predictions
    for i, (gloss, prob) in enumerate(top3[1:3], start=1):
        cv2.putText(
            frame,
            f"{i+1}. {gloss}  {prob:.0%}",
            (12, 90 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    # FPS counter
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 120, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return frame


# ---------------------------------------------------------------------------
# Real-time inference engine
# ---------------------------------------------------------------------------

class RealtimeInference:
    """Sliding-window real-time inference engine.

    Accumulates `seq_len` frames, runs the model, displays prediction.
    Slides the window by `stride` frames after each prediction so the
    recognition responds continuously without hard resets.

    Args:
        model: Loaded SLR model in eval mode.
        model_cfg: ModelConfig matching the checkpoint.
        idx_to_class: Dict mapping integer label index to gloss string.
        device: Inference device.
        confidence_threshold: Minimum confidence to display a prediction.
        stride: Number of frames to slide the window per inference step.
    """

    def __init__(
        self,
        model,
        model_cfg: ModelConfig,
        idx_to_class: Dict[int, str],
        device: torch.device,
        confidence_threshold: float = 0.3,
        stride: int = 4,
    ) -> None:
        self.model = model
        self.cfg = model_cfg
        self.idx_to_class = idx_to_class
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.stride = stride

        self.buffer: Deque[np.ndarray] = collections.deque(maxlen=model_cfg.seq_len)
        self.last_prediction: Optional[str] = None
        self.last_confidence: float = 0.0
        self.last_top3: List[Tuple[str, float]] = []
        self._frames_since_inference: int = 0

    def push_frame(self, keypoints: np.ndarray) -> None:
        """Add a new frame's keypoint vector to the sliding buffer.

        Args:
            keypoints: Float32 array of shape (225,).
        """
        self.buffer.append(keypoints)
        self._frames_since_inference += 1

        # Run inference when buffer full and stride interval reached
        if (
            len(self.buffer) == self.cfg.seq_len
            and self._frames_since_inference >= self.stride
        ):
            self._run_inference()
            self._frames_since_inference = 0

    @torch.no_grad()
    def _run_inference(self) -> None:
        """Run one forward pass over the current buffer contents."""
        seq = np.stack(list(self.buffer), axis=0)  # (T, 225)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, T, 225)

        logits = self.model(tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()  # (num_classes,)

        top_probs, top_indices = probs.topk(min(3, len(self.idx_to_class)))

        self.last_top3 = [
            (self.idx_to_class.get(idx.item(), str(idx.item())), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

        top_conf = self.last_top3[0][1]
        if top_conf >= self.confidence_threshold:
            self.last_prediction = self.last_top3[0][0]
            self.last_confidence = top_conf
        else:
            self.last_prediction = None
            self.last_confidence = top_conf


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    checkpoint_path: str,
    model_type: str,
    label_json: str,
    camera_id: int,
    confidence_threshold: float,
    stride: int,
) -> None:
    """Open webcam and run real-time recognition until 'q' is pressed.

    Args:
        checkpoint_path: Path to trained .pt checkpoint.
        model_type: 'lstm' or 'transformer'.
        label_json: Path to label_to_idx.json.
        camera_id: OpenCV camera device index.
        confidence_threshold: Minimum confidence to display a prediction.
        stride: Window slide interval in frames.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", device)

    # Load model
    if model_type == "lstm":
        model = SLRModelLSTM.load_checkpoint(checkpoint_path, device=str(device))
    else:
        model = SLRModelTransformer.load_checkpoint(checkpoint_path, device=str(device))
    model.to(device)
    model.eval()
    logger.info("Model loaded: %s", checkpoint_path)

    model_cfg: ModelConfig = model.cfg

    # Load vocab
    with open(label_json) as f:
        class_to_idx: Dict[str, int] = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # MediaPipe
    mp_holistic = mp_lib.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,   # tracking mode for video
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    engine = RealtimeInference(
        model=model,
        model_cfg=model_cfg,
        idx_to_class=idx_to_class,
        device=device,
        confidence_threshold=confidence_threshold,
        stride=stride,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", camera_id)
        sys.exit(1)

    logger.info("Webcam opened. Press 'q' to quit.")

    fps_counter: Deque[float] = collections.deque(maxlen=30)
    import time

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame capture failed — retrying.")
            continue

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Rebuild a fake holistic result object accepted by extract_frame_landmarks
        joints = extract_frame_landmarks(frame, holistic)  # (75, 3)
        joints = normalize_frame(joints)                   # (75, 3)
        keypoints = joints.reshape(FEATURE_DIM)            # (225,)

        engine.push_frame(keypoints)

        fps_counter.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        fps = float(np.mean(fps_counter))

        annotated = _draw_overlay(
            frame=frame,
            prediction=engine.last_prediction,
            confidence=engine.last_confidence,
            fps=fps,
            buffer_fill=len(engine.buffer),
            seq_len=model_cfg.seq_len,
            top3=engine.last_top3,
        )

        cv2.imshow("SLR — Real-time ASL Recognition", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    logger.info("Real-time inference stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time ASL recognition from webcam."
    )
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--model_type",  type=str, required=True, choices=["lstm", "transformer"])
    parser.add_argument("--label_json",  type=str, default="data/splits/label_to_idx.json")
    parser.add_argument("--camera_id",   type=int, default=0)
    parser.add_argument("--confidence_threshold", type=float, default=0.3,
                        help="Min softmax confidence to display a prediction.")
    parser.add_argument("--stride",      type=int, default=4,
                        help="Slide window every N frames (lower = more responsive, higher = faster).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        label_json=args.label_json,
        camera_id=args.camera_id,
        confidence_threshold=args.confidence_threshold,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
