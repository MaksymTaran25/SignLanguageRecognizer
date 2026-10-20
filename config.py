"""Configuration for the Sign Language Recognizer."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Active vocabulary — only signs with sufficient personal training data.
# WLASL-only signs are dropped (they won't generalize to your webcam).
VOCABULARY = [
    "hello", "thank_you", "yes", "no", "please",
    "sorry", "help", "want", "eat", "drink", "I",
]

NUM_CLASSES = len(VOCABULARY)

# MediaPipe / hand topology
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3  # x, y, z
MAX_HANDS = 2  # support two-handed signs (more, help, water, please, ...)
FRAME_FEATURE_DIM = MAX_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 126

# Temporal pipeline:
#   - capture FRAMES_PER_SIGN raw frames (collection / video extraction / live buffer)
#   - resample to SEQUENCE_LENGTH evenly-spaced frames before feeding the model
# This preserves motion while keeping the input dim manageable.
FRAMES_PER_SIGN = 15
SEQUENCE_LENGTH = 10
NUM_FEATURES = SEQUENCE_LENGTH * FRAME_FEATURE_DIM  # 1260

SAMPLES_PER_SIGN = 30  # default target during personal collection

# MLP architecture
HIDDEN_LAYERS = (512, 256, 128)
LEARNING_RATE = 0.001
L2_ALPHA = 0.001
MAX_ITER = 1000
EARLY_STOPPING = True

# Evaluation
N_FOLDS = 5
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Live demo
DEMO_CONFIDENCE_THRESHOLD = 0.55
DEMO_SMOOTHING_WINDOW = 5
