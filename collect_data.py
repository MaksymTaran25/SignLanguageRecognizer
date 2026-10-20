"""Webcam data collection for sign-language sequences.

Each press of SPACE records a short clip (FRAMES_PER_SIGN frames),
extracts the 126-D per-frame features, resamples to SEQUENCE_LENGTH,
and saves it as a (SEQUENCE_LENGTH, 126) array. This MUST match the
shape produced by download_wlasl.py and used by demo.py.

Usage:
    python collect_data.py                              # all 20 signs
    python collect_data.py --signs hello yes no         # specific signs
    python collect_data.py --samples 30                 # 30 samples per sign
"""

import argparse
import os
import time

import cv2
import numpy as np

import config
from keypoints import HandKeypointExtractor, normalize_frame, resample_sequence

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),
    (11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17),
]

COUNTDOWN_FRAMES = 12  # ~0.4s @ 30fps countdown before recording starts


def draw_landmarks_cv2(frame, hands):
    """Draw all detected hands."""
    h, w = frame.shape[:2]
    for landmarks in hands:
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def record_clip(extractor, cap, frames_per_sign):
    """Capture frames_per_sign frames of features after a short countdown.

    Returns:
        sequence: (SEQUENCE_LENGTH, 126) array, or None if hand was rarely seen.
    """
    # Countdown overlay
    for i in range(COUNTDOWN_FRAMES, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Recording in... {i // 4 + 1}",
                    (frame.shape[1] // 2 - 180, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(15)

    frame_features = []
    frames_seen = 0
    while frames_seen < frames_per_sign:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = extractor.extract_hands(frame_rgb)
        if hands:
            frame_features.append(normalize_frame(hands))

        # Recording indicator
        cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
        cv2.putText(frame, f"REC  {frames_seen + 1}/{frames_per_sign}",
                    (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if hands:
            draw_landmarks_cv2(frame, hands)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)
        frames_seen += 1

    if len(frame_features) < max(5, frames_per_sign // 3):
        return None
    return resample_sequence(frame_features, target_length=config.SEQUENCE_LENGTH)


def collect_samples_for_sign(sign_name, extractor, cap, num_samples,
                             frames_per_sign, save_dir):
    """Collect samples for one sign. Returns ('done'|'skip'|'quit', count)."""
    sign_dir = os.path.join(save_dir, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    existing = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
    saved = 0

    while saved < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = extractor.extract_hands(frame_rgb)
        hand_visible = bool(hands)
        if hand_visible:
            draw_landmarks_cv2(frame, hands)

        # UI overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        sign_display = sign_name.replace("_", " ")
        cv2.putText(frame, f"Sign: {sign_display}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame,
                    f"Saved: {saved}/{num_samples}  (total on disk: {existing + saved})",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if hand_visible:
            n = len(hands)
            cv2.putText(frame,
                        f"{n} hand{'s' if n > 1 else ''}  -  SPACE=record  S=skip  Q=quit",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Show your hand!  S=skip  Q=quit",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            return "quit", saved
        if key == ord("s"):
            return "skip", saved
        if key == ord(" ") and hand_visible:
            seq = record_clip(extractor, cap, frames_per_sign)
            if seq is None:
                print(f"  [{sign_display}] dropped — hand lost during recording")
                continue
            sample_num = existing + saved
            filepath = os.path.join(sign_dir, f"sample_{sample_num:04d}.npy")
            np.save(filepath, seq)
            saved += 1
            print(f"  [{sign_display}] {saved}/{num_samples} saved")

    return "done", saved


def main():
    parser = argparse.ArgumentParser(description="Collect sign language sequences")
    parser.add_argument("--signs", nargs="+", default=None,
                        help="Specific signs to collect (default: all vocabulary)")
    parser.add_argument("--samples", type=int, default=15,
                        help="Samples per sign (default: 15)")
    parser.add_argument("--frames", type=int, default=config.FRAMES_PER_SIGN,
                        help=f"Frames captured per recording (default: {config.FRAMES_PER_SIGN})")
    args = parser.parse_args()

    signs = args.signs or config.VOCABULARY
    os.makedirs(config.RAW_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Waiting for camera...")
    ret = False
    for _ in range(30):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.1)
    if not ret:
        print("Error: Camera not providing frames.")
        cap.release()
        return

    print("\nSign Language Data Collection")
    print(f"Signs: {signs}")
    print(f"Samples per sign: {args.samples}")
    print(f"Sequence length stored per sample: {config.SEQUENCE_LENGTH} frames "
          f"x {config.FRAME_FEATURE_DIM}-D = {config.NUM_FEATURES}")
    print("\nControls:")
    print("  SPACE  — start recording (short countdown, then capture)")
    print("  S      — skip to next sign")
    print("  Q      — quit\n")

    with HandKeypointExtractor(static_image_mode=False,
                               max_num_hands=config.MAX_HANDS) as extractor:
        for i, sign in enumerate(signs):
            print(f"\n[{i+1}/{len(signs)}] Collecting '{sign}'")
            status, count = collect_samples_for_sign(
                sign, extractor, cap, args.samples, args.frames, config.RAW_DIR
            )
            if status == "quit":
                print(f"\nStopped early. Collected {count} samples for '{sign}'.")
                break
            elif status == "skip":
                print(f"  Skipped '{sign}' ({count} samples collected)")
            else:
                print(f"  Done with '{sign}' ({count} samples)")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 40)
    print("Collection Summary")
    print("=" * 40)
    total = 0
    for sign in config.VOCABULARY:
        sign_dir = os.path.join(config.RAW_DIR, sign)
        if os.path.isdir(sign_dir):
            count = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
            total += count
            print(f"  {sign:15s}: {count} samples")
    print(f"\n  Total: {total} samples")
    print("\nNext: python train.py")


if __name__ == "__main__":
    main()
