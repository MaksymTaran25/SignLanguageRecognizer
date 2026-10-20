"""Real-time sign language recognition demo using webcam.

Mirrors the training pipeline exactly:
  - per frame: 126-D feature (2 hands, slot-0 = leftmost)
  - rolling buffer of FRAMES_PER_SIGN raw feature frames
  - resampled to SEQUENCE_LENGTH and flattened before predict_proba

Usage:
    python demo.py
    python demo.py --threshold 0.6
"""

import argparse
import collections
import os
import pickle
import time

import cv2
import numpy as np

import config
from keypoints import (
    HandKeypointExtractor,
    flatten_sequence,
    normalize_frame,
    resample_sequence,
)
from voice import Speaker

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),
    (11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17),
]


def draw_landmarks_cv2(frame, hands):
    h, w = frame.shape[:2]
    for landmarks in hands:
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, "sign_classifier.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}. Run train.py first."
        )
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Real-time sign language demo")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--threshold", type=float,
                        default=config.DEMO_CONFIDENCE_THRESHOLD,
                        help=f"Minimum confidence to display/speak prediction "
                             f"(default: {config.DEMO_CONFIDENCE_THRESHOLD})")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable text-to-speech output")
    parser.add_argument("--voice-name", default=None,
                        help="macOS voice (e.g. Samantha, Daniel). Default: system.")
    parser.add_argument("--voice-rate", type=int, default=190,
                        help="Speaking rate in words per minute (default: 190)")
    parser.add_argument("--hold-time", type=float, default=1.0,
                        help="Seconds the same sign must be held (stable + above "
                             "threshold) before it gets spoken (default: 1.0)")
    args = parser.parse_args()

    pipeline = load_model(args.model)

    speaker = None if args.no_voice else Speaker(voice=args.voice_name,
                                                 rate=args.voice_rate)
    if speaker and not speaker.enabled:
        print("Voice output: not available on this system (continuing silently).")
    elif speaker:
        print(f"Voice output: enabled ({speaker.backend})")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Rolling buffer of per-frame 126-D features. Holds the last
    # FRAMES_PER_SIGN frames where a hand was visible.
    frame_buffer = collections.deque(maxlen=config.FRAMES_PER_SIGN)
    pred_buffer = collections.deque(maxlen=config.DEMO_SMOOTHING_WINDOW)
    last_spoken_idx = None

    # Hold-time gate: only speak when the same sign has been the smoothed
    # prediction AND above the confidence threshold for at least HOLD_TIME
    # seconds in a row. This prevents flashes from the model triggering speech.
    hold_idx = None
    hold_started_at = 0.0

    print("\nReal-time Sign Language Recognizer")
    print(f"Vocabulary: {config.VOCABULARY}")
    print(f"Buffering {config.FRAMES_PER_SIGN} frames before predicting")
    print("Press 'q' to quit\n")

    print("Waiting for camera...")
    ret = False
    for _ in range(30):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.1)
    if not ret:
        print("Error: Camera not providing frames. Check camera permissions.")
        cap.release()
        return

    with HandKeypointExtractor(static_image_mode=False,
                               max_num_hands=config.MAX_HANDS) as extractor:
        prev_time = time.time()
        prediction_text = "Collecting frames..."
        confidence = 0.0
        sign_name = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands = extractor.extract_hands(frame_rgb)
            if hands:
                features = normalize_frame(hands)
                frame_buffer.append(features)
                draw_landmarks_cv2(frame, hands)
            else:
                # Lose the buffer slowly: drop one frame so brief occlusion
                # doesn't wipe progress, but sustained absence does.
                if frame_buffer:
                    frame_buffer.popleft()

            ready = len(frame_buffer) >= config.FRAMES_PER_SIGN
            if ready:
                seq = resample_sequence(list(frame_buffer),
                                        target_length=config.SEQUENCE_LENGTH)
                features_2d = flatten_sequence(seq).reshape(1, -1)
                probabilities = pipeline.predict_proba(features_2d)[0]
                pred_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_idx])

                pred_buffer.append(pred_idx)
                counts = np.bincount(list(pred_buffer), minlength=config.NUM_CLASSES)
                smoothed_idx = int(np.argmax(counts))
                sign_name = config.VOCABULARY[smoothed_idx]

                now = time.time()
                confident = confidence >= args.threshold

                # Maintain the hold-time gate.
                if confident and smoothed_idx == hold_idx:
                    held_for = now - hold_started_at
                else:
                    # Either the sign changed or confidence dropped — reset.
                    hold_idx = smoothed_idx if confident else None
                    hold_started_at = now if confident else 0.0
                    held_for = 0.0

                if confident:
                    if held_for >= args.hold_time:
                        prediction_text = f"{sign_name} ({confidence:.0%})"
                        if speaker and smoothed_idx != last_spoken_idx:
                            spoken = sign_name.replace("_", " ")
                            if speaker.speak(spoken):
                                last_spoken_idx = smoothed_idx
                    else:
                        # Show that we're waiting on the hold timer to fill.
                        progress = int(100 * held_for / args.hold_time)
                        prediction_text = (f"hold {sign_name}... "
                                           f"{progress}%  ({confidence:.0%})")
                else:
                    prediction_text = f"? ({sign_name} {confidence:.0%})"
            elif hands:
                prediction_text = (f"Collecting frames... "
                                   f"({len(frame_buffer)}/{config.FRAMES_PER_SIGN})")
            else:
                pred_buffer.clear()
                prediction_text = "No hand detected"
                confidence = 0.0
                hold_idx = None
                hold_started_at = 0.0
                # Hand left → next time we see a sign, allow speaking again
                if last_spoken_idx is not None and speaker:
                    speaker.reset()
                last_spoken_idx = None

            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            color = (0, 255, 0) if confidence >= args.threshold else (0, 165, 255)
            cv2.putText(frame, prediction_text,
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame,
                        f"FPS: {fps:.0f}  Buffer: {len(frame_buffer)}/{config.FRAMES_PER_SIGN}  "
                        f"Hands: {len(hands)}",
                        (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            bar_w = int(confidence * 200)
            cv2.rectangle(frame, (frame.shape[1] - 220, 30),
                          (frame.shape[1] - 220 + bar_w, 50), color, -1)
            cv2.rectangle(frame, (frame.shape[1] - 220, 30),
                          (frame.shape[1] - 20, 50), (255, 255, 255), 1)

            cv2.imshow("Sign Language Recognizer", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if speaker:
        speaker.close()
    print("Demo ended.")


if __name__ == "__main__":
    main()
