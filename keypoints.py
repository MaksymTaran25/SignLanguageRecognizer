"""Hand keypoint extraction and normalization using MediaPipe Tasks API.

Per-frame feature: up to 2 hands × 21 landmarks × 3 coords = 126-D vector.
Hands are slotted by their image x-position (leftmost → slot 0). When only
one hand is visible, slot 1 is zero-filled. Normalization is done in a single
shared frame so inter-hand geometry survives — critical for two-handed signs
like 'more', 'help', 'water', 'please'.
"""

import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

import config

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")


def _ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading hand landmarker model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


class HandKeypointExtractor:
    """Extracts up to 2 hands and produces a 126-D normalized frame vector."""

    def __init__(self, static_image_mode=True, max_num_hands=config.MAX_HANDS,
                 min_detection_confidence=0.5):
        _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.static_image_mode = static_image_mode
        self._frame_ts = 0

    def extract_hands(self, frame_rgb):
        """Return a list of (21, 3) landmark arrays for every detected hand.

        Order is undefined; caller should sort if a stable slot assignment matters.
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.static_image_mode:
            results = self.landmarker.detect(image)
        else:
            self._frame_ts += 33  # ~30 fps in ms
            results = self.landmarker.detect_for_video(image, self._frame_ts)

        if not results.hand_landmarks:
            return []
        return [np.array([[lm.x, lm.y, lm.z] for lm in hand])
                for hand in results.hand_landmarks]

    def extract_frame_features(self, frame_rgb):
        """Run detection and return a 126-D normalized frame feature, or None
        if no hand is visible in the frame.
        """
        hands = self.extract_hands(frame_rgb)
        if not hands:
            return None
        return normalize_frame(hands)

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _sort_hands_by_x(hands):
    """Sort detected hands left-to-right by mean x of their landmarks."""
    return sorted(hands, key=lambda h: float(np.mean(h[:, 0])))


def normalize_frame(hands):
    """Build the 126-D feature for one frame.

    Args:
        hands: list of (21, 3) arrays, one per detected hand.

    Returns:
        np.ndarray of shape (126,). Slot 0 is the leftmost hand (or the only
        hand). Slot 1 holds the rightmost hand, or zeros if only one hand was
        seen. Both hands share the same origin (slot-0 wrist) and scale
        (slot-0 wrist→middle-MCP), so inter-hand position is preserved.
    """
    if not hands:
        return np.zeros(config.FRAME_FEATURE_DIM, dtype=np.float32)

    hands = _sort_hands_by_x(hands)[:config.MAX_HANDS]
    primary = hands[0]
    origin = primary[0]  # wrist of slot-0 hand
    scale = float(np.linalg.norm(primary[9] - origin))  # wrist → middle MCP
    if scale < 1e-6:
        scale = 1.0

    slots = []
    for i in range(config.MAX_HANDS):
        if i < len(hands):
            slots.append((hands[i] - origin) / scale)
        else:
            slots.append(np.zeros((config.LANDMARKS_PER_HAND,
                                   config.COORDS_PER_LANDMARK),
                                  dtype=np.float32))
    return np.concatenate(slots, axis=0).flatten().astype(np.float32)


def resample_sequence(frames, target_length=None):
    """Resample a list of frame features to a fixed length via linear index sampling.

    Pads with the last available frame if the input is shorter than target_length.

    Args:
        frames: list of (126,) np arrays. May be empty.
        target_length: defaults to config.SEQUENCE_LENGTH.

    Returns:
        np.ndarray of shape (target_length, 126), or None if frames is empty.
    """
    if target_length is None:
        target_length = config.SEQUENCE_LENGTH
    if not frames:
        return None

    arr = np.stack(frames, axis=0).astype(np.float32)
    n = arr.shape[0]
    if n == target_length:
        return arr

    idx = np.linspace(0, n - 1, num=target_length)
    idx = np.clip(np.round(idx).astype(int), 0, n - 1)
    return arr[idx]


def flatten_sequence(seq):
    """Flatten a (T, 126) sequence into a 1-D feature vector for the MLP."""
    return seq.reshape(-1).astype(np.float32)
