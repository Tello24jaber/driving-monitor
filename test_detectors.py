#!/usr/bin/env python3
"""
Test script to examine detector outputs and understand why danger warning triggers
"""
import cv2
import numpy as np
import mediapipe as mp
from detect.face import FaceDetector
from detect.pose import HeadPose
from signals import SignalBuffer, MovingAverageFilter, PERCLOSCalculator, DrowsinessDecisionEngine

# Initialize
cap = cv2.VideoCapture(0)
faceMesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
headPose = HeadPose(faceMesh)
faceDetector = FaceDetector(faceMesh, fps, 0.7, 0.15, 6, 0.28, 10)

signal_buffer = SignalBuffer(max_size=300)
ear_filter = MovingAverageFilter(window_size=7)
pitch_filter = MovingAverageFilter(window_size=15)
perclos_calc = PERCLOSCalculator(window_size=90, threshold=0.18)
decision_engine = DrowsinessDecisionEngine()

frame_count = 0

print("Starting detector test... Press 'q' to exit")
print("-" * 80)
print(f"{'Frame':<6} {'EAR':<8} {'Pitch':<8} {'PERCLOS':<10} {'State':<8} {'Reason':<40}")
print("-" * 80)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        frame, results = headPose.process_image(frame)
        frame = headPose.estimate_pose(frame, results, False)
        roll, pitch, yaw = headPose.calculate_angles()
        
        frame, sleepEyes, mar, gaze, yawning, baseR, baseP, baseY, baseG = faceDetector.evaluate_face(
            frame, results, roll, pitch, yaw, False
        )
        
        # Get EAR
        ear_value = faceDetector.ear if hasattr(faceDetector, 'ear') else 0.0
        
        # Add to buffer
        signal_buffer.add_sample(ear_value, ear_value, pitch, roll, yaw)
        
        # Check if we have enough samples
        if signal_buffer.is_ready(30):
            raw_ear = signal_buffer.get_array('ear_avg')
            raw_pitch = signal_buffer.get_array('pitch')
            
            smoothed_ear = ear_filter.apply(raw_ear)
            smoothed_pitch = pitch_filter.apply(raw_pitch)
            
            current_smoothed_ear = smoothed_ear[-1]
            current_smoothed_pitch = smoothed_pitch[-1]
            current_perclos = perclos_calc.compute(smoothed_ear)
            
            # Make decision
            state, reason = decision_engine.evaluate(
                current_smoothed_ear,
                current_perclos,
                current_smoothed_pitch
            )
            
            # Print debug info
            if frame_count % 10 == 0:
                print(f"{frame_count:<6} {current_smoothed_ear:<8.3f} {pitch:<8.2f} {current_perclos:<10.1f} {state:<8} {reason[:40]:<40}")
        
        # Display
        cv2.imshow('Debug View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest complete!")
