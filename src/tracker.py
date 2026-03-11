"""
Real-Time Biomechanical Telemetry Extractor (Phase 1 MVP)
---------------------------------------------------------
Architecture Pipeline: 
1. Hardware Ingestion: OpenCV (cv2) captures physical light / 2D matrices.
2. AI Inference: Google MediaPipe processes RGB frames for 21-node spatial mapping.
3. Data Extraction: Isolate specific geometric nodes (Wrist, Index) and output 
   normalized (X, Y, Z) coordinates for downstream ergonomic threshold calculations.
"""

import cv2
import mediapipe as mp

def main():
    # --- SYSTEM INITIALIZATION ---
    print("[SYSTEM] Initializing AI Inference Engine...")
    mp_hands = mp.solutions.hands
    # Configure model: Max 1 hand for performance, 70% confidence thresholds for strict validation
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # --- HARDWARE INGESTION ---
    print("[SYSTEM] Bridging Hardware Webcam (Port 0)...")
    cap = cv2.VideoCapture(0)
    
    # Set resolution to 640x480 for optimal balance of detail and processing speed
    # Change to 1280x720 if higher fidelity is required and hardware allows
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
            continue # Drop empty frames to maintain loop frequency

        # Pre-process matrix: Flip for user ergonomics (mirror image), convert BGR to RGB for model
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Execute Inference
        results = hands.process(frame_rgb)

        # --- TELEMETRY EXTRACTION ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Visual verification: Draw skeletal wireframe on the 2D output feed
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract Target Nodes: 0 (Base of Wrist) and 8 (Index Finger Tip)
                wrist = hand_landmarks.landmark[0]
                index = hand_landmarks.landmark[8]
                
                # Output raw data stream for Phase 2 geometric processing
                print(f"Wrist[X:{wrist.x:.3f} Y:{wrist.y:.3f} Z:{wrist.z:.3f}] | Index[X:{index.x:.3f} Y:{index.y:.3f} Z:{index.z:.3f}]")

        # Render visual diagnostic feed
        cv2.imshow("MediaPipe Vision Architecture - MVP Phase 1", frame)

        # Hardware interrupt to safely terminate the control loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[SYSTEM] Terminating data stream and releasing hardware...")
            break

    # --- HARDWARE CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()