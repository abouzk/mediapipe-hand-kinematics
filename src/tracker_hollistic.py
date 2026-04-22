"""
Real-Time Biomechanical Telemetry Extractor (Phase 1.5 MVP - Holistic)
---------------------------------------------------------
Architecture Pipeline: 
1. Hardware Ingestion: OpenCV (cv2) captures physical light / 2D matrices.
2. AI Inference: Google MediaPipe Holistic processes RGB frames for full-body + hand spatial mapping.
3. Data Extraction: Isolate specific geometric nodes across different sub-models 
   (Pose: Elbow/Shoulder, Hand: Wrist/Index) for advanced angular calculations.
"""

import cv2
import mediapipe as mp

def check_mediapipe_compatibility():
    """Validate that the installed MediaPipe build has the required solutions API."""
    version = getattr(mp, "__version__", "unknown")
    if not hasattr(mp, "solutions"):
        print("[ERROR] Incompatible MediaPipe installation detected.")
        print(f"[ERROR] Installed version: {version}")
        print("[ERROR] This project requires a MediaPipe build that has `mp.solutions`.")
        print("[FIX] Run: pip install mediapipe==0.10.14")
        return False
    return True

def main():
    # --- SYSTEM INITIALIZATION ---
    print("[SYSTEM] Initializing Holistic AI Inference Engine...")
    if not check_mediapipe_compatibility():
        return

    mp_holistic = mp.solutions.holistic
    # Configure model: Holistic combines Pose, Face, and Hands. 
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    # --- HARDWARE INGESTION ---
    print("[SYSTEM] Bridging Hardware Webcam (Port 0)...")
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[ERROR] Hardware Capture failed. Check USB/Camera permissions.")
        return

    print("[STATUS] Pipeline Active. Streaming XYZ Telemetry to stdout. Press 'q' to terminate.")

    # --- MAIN CONTROL LOOP ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Pre-process matrix
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Execute Inference
        results = holistic.process(frame_rgb)

        # --- TELEMETRY EXTRACTION ---
        
        # 1. Render visual diagnostic feed for Pose (Arm) and Right Hand
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 2. Extract specific nodes for downstream math (Assuming Right Hand for piano MVP)
        if results.pose_landmarks and results.right_hand_landmarks:
            
            # Extract Macro Arm Vectors from Pose Model
            # 12 = Right Shoulder, 14 = Right Elbow
            r_shoulder = results.pose_landmarks.landmark[12]
            r_elbow = results.pose_landmarks.landmark[14]
            
            # Extract Micro Hand Vectors from Hand Model for maximum precision
            # 0 = Base of Wrist, 8 = Index Finger Tip
            r_wrist = results.right_hand_landmarks.landmark[0]
            r_index = results.right_hand_landmarks.landmark[8]
            
            # Output raw data stream
            print(f"Elbow[X:{r_elbow.x:.2f} Y:{r_elbow.y:.2f}] | Wrist[X:{r_wrist.x:.2f} Y:{r_wrist.y:.2f}] | Index[X:{r_index.x:.2f} Y:{r_index.y:.2f}]")

        cv2.imshow("MediaPipe Holistic Architecture - MVP Phase 1.5", frame)

        # Hardware interrupt
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[SYSTEM] Terminating data stream and releasing hardware...")
            break

    # --- HARDWARE CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()