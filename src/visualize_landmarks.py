"""
Landmark Visualizer
Draws hand and pose landmarks on a video so you can verify what
MediaPipe is detecting before running the rate file.

Outputs an annotated MP4 and prints per-frame detection stats.

Usage:
    python visualize_landmarks.py --video correct.mov
    python visualize_landmarks.py --video correct.mov --output annotated.mp4
    python visualize_landmarks.py --video correct.mov --hand_only
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import urllib.request
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker_lite.task"

# Colors (BGR)
COLOR_HAND  = (0, 220, 120)   # green
COLOR_POSE  = (255, 140, 0)   # blue-orange
COLOR_WRIST = (0, 100, 255)   # red - highlight the key wrist point
COLOR_ELBOW = (0, 200, 255)   # yellow - highlight elbow


def download_models():
    for url, path in [(HAND_MODEL_URL, HAND_MODEL_PATH),
                      (POSE_MODEL_URL, POSE_MODEL_PATH)]:
        if not Path(path).exists():
            print(f"Downloading {path}...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"[ERROR] {e}\n[FIX] Download from: {url}")
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


def lm_to_px(landmark, w, h):
    """Convert normalized landmark (0-1) to pixel coordinates."""
    return int(landmark.x * w), int(landmark.y * h)


def draw_hand(frame, hand_lms, w, h):
    """Draw all 21 hand landmarks and connections between them."""
    # MediaPipe hand connection pairs
    connections = [
        (0,1),(1,2),(2,3),(3,4),         # thumb
        (0,5),(5,6),(6,7),(7,8),         # index
        (5,9),(9,10),(10,11),(11,12),    # middle
        (9,13),(13,14),(14,15),(15,16),  # ring
        (13,17),(17,18),(18,19),(19,20), # pinky
        (0,17)                           # palm base
    ]
    pts = [lm_to_px(lm, w, h) for lm in hand_lms]

    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], COLOR_HAND, 1)

    for i, pt in enumerate(pts):
        color = COLOR_WRIST if i == 0 else COLOR_HAND
        radius = 6 if i == 0 else 3
        cv2.circle(frame, pt, radius, color, -1)


def draw_arm(frame, pose_lms, w, h):
    """Draw shoulder, elbow, wrist from pose model."""
    shoulder = lm_to_px(pose_lms[12], w, h)  # right shoulder
    elbow    = lm_to_px(pose_lms[14], w, h)  # right elbow
    wrist    = lm_to_px(pose_lms[16], w, h)  # right wrist (pose)

    cv2.line(frame, shoulder, elbow, COLOR_POSE, 2)
    cv2.line(frame, elbow, wrist, COLOR_POSE, 2)
    cv2.circle(frame, shoulder, 5, COLOR_POSE, -1)
    cv2.circle(frame, elbow, 8, COLOR_ELBOW, -1)   # elbow highlighted larger
    cv2.circle(frame, wrist, 5, COLOR_POSE, -1)

    # Label the elbow so it's clear (doesn't appear in hand-only mode)
    cv2.putText(frame, "elbow", (elbow[0]+8, elbow[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ELBOW, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",     required=True)
    parser.add_argument("--output",    default="annotated.mp4")
    parser.add_argument("--min_conf",  type=float, default=0.5,
                        help="Detection confidence (try 0.3 if little is drawn)")
    parser.add_argument("--hand_only", action="store_true",
                        help="Only draw hand landmarks, skip pose")
    args = parser.parse_args()

    if not download_models():
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {args.video}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write output as MP4
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    hand_landmarker, pose_landmarker = create_landmarkers(args.min_conf)

    frame_idx  = 0
    hand_found = 0
    pose_found = 0

    print(f"Annotating {total} frames...")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        for hand_lms in hand_result.hand_landmarks:
            draw_hand(frame, hand_lms, w, h)
            hand_found += 1

        if not args.hand_only:
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_result.pose_landmarks:
                draw_arm(frame, pose_result.pose_landmarks[0], w, h)
                pose_found += 1

        # Frame counter overlay
        cv2.putText(frame, f"frame {frame_idx}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 60 == 0:
            print(f"  {frame_idx}/{total}")

    cap.release()
    writer.release()
    hand_landmarker.close()
    pose_landmarker.close()

    print(f"\nSaved: {args.output}")
    print(f"Hand detected: {hand_found}/{frame_idx} frames ({hand_found/frame_idx*100:.0f}%)")
    if not args.hand_only:
        print(f"Pose detected: {pose_found}/{frame_idx} frames ({pose_found/frame_idx*100:.0f}%)")
    print(f"\nIf you see green dots but no orange arm lines, use --hand_only in rate_cmajor.py")


if __name__ == "__main__":
    main()