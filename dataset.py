"""Dataset loading and augmentation for temporal hand-keypoint sequences.

Each sample on disk is an (SEQUENCE_LENGTH, FRAME_FEATURE_DIM) array
saved as data/raw/{sign}/sample_XXXX.npy. Older 1-D averaged samples
(from the previous pipeline) are detected and skipped with a warning.
"""

import os

import numpy as np
from sklearn.model_selection import train_test_split

import config


EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FRAME_FEATURE_DIM)


def load_dataset(data_dir=None, flatten=True):
    """Load every sequence sample under data_dir/{sign}/*.npy.

    Args:
        data_dir: directory of per-sign subfolders. Defaults to config.RAW_DIR.
        flatten: if True, return X as (N, NUM_FEATURES). Otherwise (N, T, F).

    Returns:
        X, y, labels
    """
    if data_dir is None:
        data_dir = config.RAW_DIR

    X, y = [], []
    labels = []
    skipped = 0

    for label_idx, sign in enumerate(config.VOCABULARY):
        sign_dir = os.path.join(data_dir, sign)
        if not os.path.isdir(sign_dir):
            continue
        labels.append(sign)
        samples = sorted(f for f in os.listdir(sign_dir) if f.endswith(".npy"))
        for fname in samples:
            arr = np.load(os.path.join(sign_dir, fname))
            if arr.shape != EXPECTED_SHAPE:
                skipped += 1
                continue
            X.append(arr.astype(np.float32))
            y.append(label_idx)

    if not X:
        raise ValueError(
            f"No usable samples in {data_dir}. Expected shape {EXPECTED_SHAPE}.\n"
            "Run `python download_wlasl.py --extract-only` and/or "
            "`python collect_data.py` to (re)generate sequence samples."
        )

    if skipped:
        print(f"  Skipped {skipped} sample(s) with mismatched shape "
              f"(expected {EXPECTED_SHAPE}). Re-extract them with the new pipeline.")

    X = np.stack(X, axis=0)  # (N, T, F)
    y = np.array(y, dtype=np.int32)
    print(f"Loaded {len(X)} samples across {len(set(y))} classes from {data_dir}")

    if flatten:
        X = X.reshape(len(X), -1)
    return X, y, labels


def _augment_sequence(seq, rng, noise_std=0.015, scale_range=(0.92, 1.08),
                      rot_range_rad=0.18, time_jitter=0.10):
    """Apply a coherent augmentation across all frames of one sequence.

    The same scale and rotation are applied to every frame (so trajectory
    identity is preserved). Independent low-amplitude noise is added per frame
    to simulate landmark jitter. Time jitter resamples the frame indices
    slightly, mimicking small speed variations between recordings.

    seq: (T, F) array where F == MAX_HANDS * LANDMARKS_PER_HAND * 3.
    """
    T, F = seq.shape
    n_hands = config.MAX_HANDS
    pts = seq.reshape(T, n_hands, config.LANDMARKS_PER_HAND, 3).copy()

    # Coherent scale
    pts *= rng.uniform(*scale_range)

    # Coherent rotation around image z-axis (apply to non-zero hands only).
    angle = rng.uniform(-rot_range_rad, rot_range_rad)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    xy = pts[..., :2]
    rotated = xy @ R.T
    # Preserve zero-padding on absent hands (mean of slot-1 will be ~0).
    nonzero_mask = np.any(np.abs(pts) > 1e-6, axis=-1, keepdims=True)
    pts[..., :2] = np.where(nonzero_mask, rotated, xy)

    # Per-frame jitter
    pts += rng.normal(0, noise_std, size=pts.shape).astype(np.float32) * nonzero_mask

    aug = pts.reshape(T, F)

    # Time jitter: re-sample indices around the original timeline.
    if time_jitter > 0 and T > 2:
        offsets = rng.uniform(-time_jitter, time_jitter, size=T)
        idx = np.clip(np.arange(T) + offsets * (T - 1), 0, T - 1)
        idx_floor = np.floor(idx).astype(int)
        idx_ceil = np.clip(idx_floor + 1, 0, T - 1)
        frac = (idx - idx_floor)[:, None]
        aug = aug[idx_floor] * (1 - frac) + aug[idx_ceil] * frac

    return aug.astype(np.float32)


def augment_dataset(X, y, target_per_class=40):
    """Bring every class up to target_per_class samples via temporal augmentation.

    Args:
        X: (N, T, F) or (N, T*F) array.
        y: (N,) integer labels.

    Returns:
        X_aug, y_aug — same shape family as input.
    """
    flat_input = (X.ndim == 2)
    if flat_input:
        X3 = X.reshape(-1, config.SEQUENCE_LENGTH, config.FRAME_FEATURE_DIM)
    else:
        X3 = X

    rng = np.random.RandomState(config.RANDOM_STATE)
    aug_X = list(X3)
    aug_y = list(y)

    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts):
        if count >= target_per_class:
            continue
        cls_X = X3[y == cls]
        needed = target_per_class - count
        for _ in range(needed):
            base = cls_X[rng.randint(len(cls_X))]
            aug_X.append(_augment_sequence(base, rng))
            aug_y.append(cls)

    aug_X = np.stack(aug_X, axis=0).astype(np.float32)
    aug_y = np.array(aug_y, dtype=np.int32)
    print(f"Augmented: {len(X)} → {len(aug_X)} samples "
          f"({target_per_class} per class, {len(classes)} classes)")

    if flat_input:
        aug_X = aug_X.reshape(len(aug_X), -1)
    return aug_X, aug_y


def split_dataset(X, y):
    """Stratified 70/15/15 train/val/test split (falls back to non-stratified
    when any class has too few samples)."""
    _, counts = np.unique(y, return_counts=True)
    use_stratify = counts.min() >= 7
    if not use_stratify:
        print("  Note: some classes have <7 samples — using non-stratified split.")

    stratify_y = y if use_stratify else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=stratify_y,
    )

    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    stratify_tv = y_trainval if use_stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio,
        random_state=config.RANDOM_STATE, stratify=stratify_tv,
    )

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
