"""
C Major Scale Wrist Deviation Rater
Phase 2 MVP - Video Track

Works with mediapipe 0.10.30+. Downloads model files automatically on first run.

Usage:
    python rate_cmajor.py --correct correct.mov --incorrect incorrect.mov
    python rate_cmajor.py --correct correct.mov --incorrect incorrect.mov --min_conf 0.3
    python rate_cmajor.py --correct correct.mov --incorrect incorrect.mov --left
    python rate_cmajor.py --correct correct.mov --incorrect incorrect.mov --hand_only

--min_conf   lower this (try 0.3) if DETECTION RATE IS LOW
--left       track LEFT hand instead of RIGHT
--hand_only  skip POSE model entirely; uses hand landmarks only for angle
             use this if camera doesn't show the arm/elbow (camera too close to the keys)
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import argparse
import urllib.request
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker_lite.task"


def download_models():
    for url, path in [(HAND_MODEL_URL, HAND_MODEL_PATH),
                      (POSE_MODEL_URL, POSE_MODEL_PATH)]:
        if not Path(path).exists():
            print(f"Downloading {path} (first run only)...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  Saved ({Path(path).stat().st_size // 1024} KB)")
            except Exception as e:
                print(f"[ERROR] Could not download {path}: {e}")
                print(f"[FIX] Download manually from:\n  {url}")
                return False
    return True


def create_landmarkers(conf):
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
        min_tracking_confidence=conf
    )
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=conf,
        min_pose_presence_confidence=conf,
        min_tracking_confidence=conf
    )
    return (mp_vision.HandLandmarker.create_from_options(hand_options),
            mp_vision.PoseLandmarker.create_from_options(pose_options))


def get_target_hand(hand_result, track_left):
    if not hand_result.hand_landmarks:
        return None
    # In a non-mirrored recording: player's right hand = "Left" in MediaPipe
    target_label = "Right" if track_left else "Left"
    for i, handedness_list in enumerate(hand_result.handedness):
        if handedness_list[0].category_name == target_label:
            return hand_result.hand_landmarks[i]
    if len(hand_result.hand_landmarks) == 1:
        return hand_result.hand_landmarks[0]
    return None


def wrist_deviation(elbow, wrist, index_tip):
    """
    Angle between forearm direction (elbow->wrist) and hand direction
    (wrist->index tip). Neutral wrist = 180 degrees. Deviation = |180 - angle|.
    Returns 0 for flat wrist, higher for a broken/bent wrist.
    """
    forearm = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
    hand    = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
    if np.linalg.norm(forearm) < 1e-6 or np.linalg.norm(hand) < 1e-6:
        return None
    cos_angle = np.clip(
        np.dot(forearm, hand) / (np.linalg.norm(forearm) * np.linalg.norm(hand)),
        -1.0, 1.0
    )
    return abs(180.0 - np.degrees(np.arccos(cos_angle)))


def wrist_deviation_hand_only(hand_lms):
    """
    Fallback when no elbow is visible (camera too tight on the keys).
    Uses the base of the index finger (landmark 5) as a proxy for the
    forearm direction instead of the actual elbow.
    Less accurate than using the elbow, but works on close-up clips.
    Landmarks: 0=wrist, 5=index base (MCP joint), 8=index tip
    """
    return wrist_deviation(hand_lms[5], hand_lms[0], hand_lms[8])


def smooth(values, window=7):
    """
    Moving average: replace each value with the average of itself
    and the (window-1)/2 frames on either side. This removes jitter
    caused by per-frame detection noise without shifting the signal
    in time. Window=7 means each point averages 3 frames before and after.
    """
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode='same')


def process_video(path, label, track_left, conf, hand_only):
    print(f"\nProcessing {label}: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open {path}")
        return [], 30.0

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {total} frames at {fps:.1f} fps  |  confidence={conf}  |  hand_only={hand_only}")

    hand_landmarker, pose_landmarker = create_landmarkers(conf)

    data        = []
    frame_idx   = 0
    detected    = 0
    pose_misses = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_lms    = get_target_hand(hand_result, track_left)

        dev = None
        if hand_lms:
            if hand_only:
                dev = wrist_deviation_hand_only(hand_lms)
            else:
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                if pose_result.pose_landmarks:
                    elbow = pose_result.pose_landmarks[0][14]
                    dev   = wrist_deviation(elbow, hand_lms[0], hand_lms[8])
                else:
                    pose_misses += 1

        if dev is not None:
            data.append((frame_idx, dev))
            detected += 1

        frame_idx += 1

    cap.release()
    hand_landmarker.close()
    pose_landmarker.close()

    pct = detected / frame_idx * 100 if frame_idx > 0 else 0
    print(f"  Detected in {detected}/{frame_idx} frames ({pct:.0f}%)")

    if not hand_only and pose_misses > frame_idx * 0.5:
        print(f"  [WARNING] Pose (elbow) not found in {pose_misses} frames.")
        print(f"  [TIP] Camera may be too close to the keys. Try --hand_only")
    if pct < 20:
        print(f"  [TIP] Low rate — also try --left if tracking the wrong hand")

    return data, fps


def plot_results(correct, correct_fps, incorrect, incorrect_fps, threshold, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("C Major Scale — Wrist Deviation", fontsize=13, fontweight='bold')

    clips = [
        (correct,   correct_fps,   "Correct technique",   axes[0], "#1976D2"),
        (incorrect, incorrect_fps, "Incorrect technique", axes[1], "#D32F2F"),
    ]

    for data, fps, label, ax, color in clips:
        if not data:
            ax.text(0.5, 0.5, f"No hand detected in {label}",
                    transform=ax.transAxes, ha='center')
            continue

        frames, devs = zip(*data)
        times        = [f / fps for f in frames]
        devs_smooth  = smooth(list(devs))

        mean_dev = np.mean(devs)
        pct_over = np.mean(np.array(devs) > threshold) * 100

        # Faint raw signal behind the smoothed line so you can see
        # how much jitter the smoothing removed
        ax.plot(times, devs, alpha=0.15, color=color, linewidth=0.8)
        ax.plot(times, devs_smooth, color=color, linewidth=2,
                label="Wrist deviation (smoothed)")
        ax.axhline(threshold, color='orange', linestyle='--', linewidth=1.5,
                   label=f"Threshold ({threshold}°)")
        ax.fill_between(times, devs_smooth, threshold,
                        where=[d > threshold for d in devs_smooth],
                        alpha=0.25, color='orange', label="Over threshold")

        ax.set_title(f"{label}   |   Mean: {mean_dev:.1f}°   Over threshold: {pct_over:.0f}% of frames")
        ax.set_ylabel("Deviation (degrees)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylim(0, max(np.max(devs) * 1.2, threshold * 2))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nReport saved to {out_path}")
    plt.show()

    if correct and incorrect:
        c_mean = np.mean([d for _, d in correct])
        i_mean = np.mean([d for _, d in incorrect])
        print(f"\nCorrect mean deviation:   {c_mean:.1f} degrees")
        print(f"Incorrect mean deviation: {i_mean:.1f} degrees")
        print(f"Difference:               {i_mean - c_mean:+.1f} degrees")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct",   required=True)
    parser.add_argument("--incorrect", required=True)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--output",    default="wrist_deviation_report.png")
    parser.add_argument("--left",      action="store_true",
                        help="Track left hand instead of right")
    parser.add_argument("--min_conf",  type=float, default=0.7,
                        help="Detection confidence (try 0.3 if detection rate is low)")
    parser.add_argument("--hand_only", action="store_true",
                        help="Skip pose/elbow; use hand landmarks only. "
                             "Use if camera is too close to see the arm.")
    args = parser.parse_args()

    if not download_models():
        return

    for p in [args.correct, args.incorrect]:
        if not Path(p).exists():
            print(f"[ERROR] File not found: {p}")
            return

    correct_data,   correct_fps   = process_video(
        args.correct,   "Correct",   args.left, args.min_conf, args.hand_only)
    incorrect_data, incorrect_fps = process_video(
        args.incorrect, "Incorrect", args.left, args.min_conf, args.hand_only)

    plot_results(
        correct_data,   correct_fps,
        incorrect_data, incorrect_fps,
        threshold=args.threshold,
        out_path=args.output
    )


if __name__ == "__main__":
    main()