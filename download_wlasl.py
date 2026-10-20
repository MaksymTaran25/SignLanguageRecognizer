"""Download WLASL videos for the 20-word subset and extract keypoints.

Usage:
    python download_wlasl.py                # Download + extract keypoints
    python download_wlasl.py --download-only # Just download videos
    python download_wlasl.py --extract-only  # Just extract from already-downloaded videos

Requires: yt-dlp, mediapipe, opencv-python
    pip install yt-dlp
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

import config
from keypoints import HandKeypointExtractor, normalize_frame, resample_sequence

EXPECTED_SHAPE = (config.SEQUENCE_LENGTH, config.FRAME_FEATURE_DIM)

# Find yt-dlp: check PATH first, then common pip install location
YT_DLP = "yt-dlp"
_user_bin = os.path.join(os.path.expanduser("~"), "Library", "Python", "3.9", "bin", "yt-dlp")
if not shutil.which(YT_DLP) and os.path.exists(_user_bin):
    YT_DLP = _user_bin

WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
WLASL_JSON_PATH = os.path.join(config.DATA_DIR, "WLASL_v0.3.json")
VIDEO_DIR = os.path.join(config.DATA_DIR, "videos")

# Map our vocabulary to WLASL gloss names (WLASL uses lowercase, no underscores)
VOCAB_TO_GLOSS = {
    "hello": "hello",
    "thank_you": "thank you",
    "yes": "yes",
    "no": "no",
    "please": "please",
    "sorry": "sorry",
    "help": "help",
    "want": "want",
    "eat": "eat",
    "drink": "drink",
    "home": "home",
    "friend": "friend",
    "name": "name",
    "more": "more",
    "stop": "stop",
    "good": "good",
    "bad": "bad",
    "water": "water",
    "I": "I",
    "you": "you",
}


def download_json():
    """Download the WLASL annotation JSON if not present."""
    if os.path.exists(WLASL_JSON_PATH):
        print(f"WLASL JSON already exists at {WLASL_JSON_PATH}")
        return
    os.makedirs(config.DATA_DIR, exist_ok=True)
    print("Downloading WLASL annotation JSON...")
    subprocess.run(
        ["curl", "-L", "-o", WLASL_JSON_PATH, WLASL_JSON_URL],
        check=True,
    )
    print(f"Saved to {WLASL_JSON_PATH}")


def load_annotations():
    """Load WLASL JSON and return entries for our 20-word subset."""
    with open(WLASL_JSON_PATH) as f:
        data = json.load(f)

    # Build a gloss → entry mapping
    gloss_map = {}
    for entry in data:
        gloss_map[entry["gloss"].lower()] = entry

    subset = {}
    for vocab_word, wlasl_gloss in VOCAB_TO_GLOSS.items():
        if wlasl_gloss.lower() in gloss_map:
            subset[vocab_word] = gloss_map[wlasl_gloss.lower()]
            n = len(subset[vocab_word].get("instances", []))
            print(f"  Found '{wlasl_gloss}' → {n} instances")
        else:
            print(f"  WARNING: '{wlasl_gloss}' not found in WLASL")

    return subset


def download_videos(subset):
    """Download videos for each sign using yt-dlp."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    total_downloaded = 0

    for vocab_word, entry in subset.items():
        sign_dir = os.path.join(VIDEO_DIR, vocab_word)
        os.makedirs(sign_dir, exist_ok=True)

        instances = entry.get("instances", [])
        print(f"\n--- {vocab_word}: {len(instances)} instances ---")

        for inst in instances:
            video_id = inst.get("video_id", "")
            url = inst.get("url", "")
            instance_id = inst.get("instance_id", 0)

            if not url and not video_id:
                continue

            output_path = os.path.join(sign_dir, f"{instance_id}.mp4")
            if os.path.exists(output_path):
                continue

            source_url = url if url else f"https://www.youtube.com/watch?v={video_id}"

            try:
                result = subprocess.run(
                    [
                        YT_DLP,
                        "--quiet", "--no-warnings",
                        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                        "-o", output_path,
                        "--socket-timeout", "10",
                        "--retries", "2",
                        source_url,
                    ],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0 and os.path.exists(output_path):
                    total_downloaded += 1
                    print(f"  Downloaded instance {instance_id}")
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"  Failed instance {instance_id}: {e}")

    print(f"\nTotal videos downloaded: {total_downloaded}")


def extract_keypoints_from_videos(subset):
    """Extract per-frame keypoint sequences from downloaded videos.

    Each output .npy stores a (SEQUENCE_LENGTH, FRAME_FEATURE_DIM) array,
    matching what collect_data.py and demo.py produce at inference time.
    Frames are horizontally flipped before extraction so the handedness-slot
    convention (slot 0 = leftmost-in-image) matches webcam-mirrored capture.
    """
    os.makedirs(config.RAW_DIR, exist_ok=True)

    with HandKeypointExtractor(static_image_mode=True,
                               max_num_hands=config.MAX_HANDS) as extractor:
        for vocab_word in subset:
            sign_video_dir = os.path.join(VIDEO_DIR, vocab_word)
            sign_output_dir = os.path.join(config.RAW_DIR, vocab_word)
            os.makedirs(sign_output_dir, exist_ok=True)

            if not os.path.isdir(sign_video_dir):
                print(f"  No videos for '{vocab_word}', skipping")
                continue

            videos = sorted([f for f in os.listdir(sign_video_dir) if f.endswith(".mp4")])
            print(f"\n--- Extracting '{vocab_word}': {len(videos)} videos ---")

            instances = subset[vocab_word].get("instances", [])
            inst_map = {str(inst.get("instance_id", 0)): inst for inst in instances}

            for video_file in videos:
                instance_id = video_file.replace(".mp4", "")
                output_path = os.path.join(sign_output_dir, f"sample_{instance_id}.npy")

                # Re-extract if the existing file has the wrong shape
                # (e.g. left over from the old averaged-pose pipeline).
                if os.path.exists(output_path):
                    try:
                        if np.load(output_path).shape == EXPECTED_SHAPE:
                            continue
                    except Exception:
                        pass

                video_path = os.path.join(sign_video_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue

                inst = inst_map.get(instance_id, {})
                frame_start = inst.get("frame_start", 1)
                frame_end = inst.get("frame_end", -1)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_start < 1:
                    frame_start = 1
                if frame_end == -1 or frame_end > total_frames:
                    frame_end = total_frames

                num_range = frame_end - frame_start + 1
                if num_range < 3:
                    cap.release()
                    continue

                # Sample more frames densely; resample at the end.
                target_samples = max(config.FRAMES_PER_SIGN, config.SEQUENCE_LENGTH * 2)
                sample_indices = np.linspace(
                    frame_start - 1, frame_end - 1, target_samples, dtype=int
                )

                frame_features = []
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)  # match webcam-mirrored convention
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hands = extractor.extract_hands(frame_rgb)
                    if hands:
                        frame_features.append(normalize_frame(hands))

                cap.release()

                if len(frame_features) < max(5, config.SEQUENCE_LENGTH // 2):
                    continue

                seq = resample_sequence(frame_features,
                                        target_length=config.SEQUENCE_LENGTH)
                np.save(output_path, seq.astype(np.float32))

            saved = 0
            for fname in os.listdir(sign_output_dir):
                if fname.endswith(".npy"):
                    try:
                        if np.load(os.path.join(sign_output_dir, fname)).shape == EXPECTED_SHAPE:
                            saved += 1
                    except Exception:
                        pass
            print(f"  '{vocab_word}': {saved} valid sequence samples")


def print_summary():
    """Print dataset summary after extraction."""
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    total = 0
    for sign in config.VOCABULARY:
        sign_dir = os.path.join(config.RAW_DIR, sign)
        if os.path.isdir(sign_dir):
            count = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
            total += count
            print(f"  {sign:15s}: {count} samples")
        else:
            print(f"  {sign:15s}: 0 samples")
    print(f"\n  Total: {total} samples across {config.NUM_CLASSES} classes")


def main():
    parser = argparse.ArgumentParser(description="Download WLASL subset and extract keypoints")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download videos, skip keypoint extraction")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract keypoints from already-downloaded videos")
    args = parser.parse_args()

    if not args.extract_only:
        download_json()
        subset = load_annotations()
        download_videos(subset)
    else:
        download_json()
        subset = load_annotations()

    if not args.download_only:
        extract_keypoints_from_videos(subset)

    print_summary()


if __name__ == "__main__":
    main()
