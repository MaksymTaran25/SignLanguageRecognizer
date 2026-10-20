# Gesture-Based Sign Language Subset Recognizer

Real-time ASL sign language recognition using MediaPipe hand landmarks and an
MLP classifier over **temporal sequences**. Recognizes 11 common ASL signs
from webcam input.

## Vocabulary (11 signs)

`hello` · `thank_you` · `yes` · `no` · `please` · `sorry` · `help` · `want`
· `eat` · `drink` · `I`

The list lives in `config.py`, so adding a sign means recording samples for it
and adding it there.

## Results

325 recorded sequences, stratified 5-fold cross-validation with augmentation
applied to the training folds only: **94.2% mean accuracy** (folds 87.7% to
96.9%). The confusion matrix from the held-out test split is in
`models/confusion_matrix.png`; the main confusion is `no` vs `drink`.

## Pipeline

```
Webcam → MediaPipe Hands (up to 2 hands × 21 landmarks)
       → Per-frame normalization (shared origin/scale, 126-D)
       → Rolling buffer of 15 frames → resample to 10 frames
       → Flatten to 1260-D → StandardScaler → MLP (512 → 256 → 128)
       → 11-class softmax → majority-vote smoothing
```

The model sees the **trajectory** of the hands across the sign, not just an
averaged static pose. Two-handed signs (`help`, `please`, ...) work because both hands are normalized in a shared
reference frame, so their relative geometry is preserved.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and a webcam.

## Usage

### 1. Get training data

The most reliable model comes from **personal recordings**: webcam lighting,
hand size, and signing style match between training and demo. WLASL is good
for bootstrapping but not enough on its own.

#### Personal recordings (recommended)

```bash
python collect_data.py                              # all 11 signs, 15 each
python collect_data.py --signs hello yes --samples 30
```

Press SPACE to start a recording (short countdown, then ~15 frames captured).
Aim for at least 20–30 samples per sign, with mild variation in distance,
angle, and speed.

#### WLASL bootstrap (optional)

```bash
python download_wlasl.py                # download videos + extract sequences
python download_wlasl.py --extract-only # re-extract from cached videos
```

Each sample is saved as a `(SEQUENCE_LENGTH, 126)` float32 array under
`data/raw/{sign}/sample_XXXX.npy`.

### 2. Train

```bash
python train.py                          # train + evaluate on held-out test set
python train.py --cross-validate         # stratified 5-fold CV
python train.py --target-per-class 80    # more aggressive augmentation
```

Saves `models/sign_classifier.pkl` and a confusion matrix to
`models/confusion_matrix.png`.

### 3. Run the demo

```bash
python demo.py
python demo.py --threshold 0.7   # require higher confidence
```

Live feed shows landmark overlay, predicted sign, confidence bar, FPS, and
buffer fill. Press `q` to quit.

## Architecture

- **Per-frame feature** (126-D): up to 2 hands × 21 landmarks × 3 coords.
  Hands are slotted by image x-position (slot 0 = leftmost). Both share a
  single normalization frame (slot-0 wrist origin, slot-0 wrist→middle-MCP
  scale), so inter-hand geometry is preserved. Missing slots are zero-padded.
- **Sequence feature** (1260-D): 10 frames × 126.
- **Classifier**: `StandardScaler` → MLP (512 → 256 → 128, ReLU) → softmax.
- **Augmentation**: per-sample coherent scale + rotation + per-frame noise +
  small time-jitter resampling. Same transform across the trajectory so
  motion identity is preserved.
- **Smoothing at inference**: majority vote over the last 5 predictions.

## Project structure

```
├── config.py           # vocabulary, sequence/feature sizes, MLP hyperparams
├── keypoints.py        # MediaPipe extraction, 2-hand normalization, resampling
├── download_wlasl.py   # WLASL download + sequence extraction
├── collect_data.py     # webcam sequence collection
├── dataset.py          # loading + temporal augmentation + train/val/test split
├── train.py            # MLP training + cross-validation
├── evaluate.py         # accuracy, F1, confusion matrix, top confused pairs
├── demo.py             # real-time inference with rolling buffer
├── data/raw/           # per-sign sequence samples (T, 126), not committed
└── models/             # classifier + MediaPipe model (downloaded on first run)
```
