import time
import numpy as np
import cv2
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.deep_signal_model import SignalDeepClassifier

def main():
    print("Initializing SignalDeepClassifier...")
    classifier = SignalDeepClassifier()
    
    if not classifier.enabled:
        print("ERROR: Classifier failed to load model!")
        return

    print("Model loaded successfully.")
    
    # Test inputs
    test_inputs = [
        (0.35, 0.05, 0.0, 0.1),  # Safe
        (0.15, 0.40, 5.0, 0.2),  # Drowsy (eyes)
        (0.30, 0.10, 25.0, 0.1), # Drowsy (head)
        (0.30, 0.10, 0.0, 0.8),  # Drowsy (yawn)
    ]
    
    print("\nChecking predictions:")
    for ear, perclos, pitch, mar in test_inputs:
        prob = classifier.predict_danger_prob(ear, perclos, pitch, mar)
        print(f"Input (EAR={ear}, PERCLOS={perclos}, Pitch={pitch}, MAR={mar}) -> Danger Prob: {prob:.4f}")

    # Latency Test
    print("\nRunning latency test (1000 iterations)...")
    start_time = time.time()
    n_iters = 1000
    
    for _ in range(n_iters):
        # Random inputs to simulate varying data
        ear = np.random.uniform(0.1, 0.4)
        perclos = np.random.uniform(0.0, 0.5)
        pitch = np.random.uniform(-30, 30)
        mar = np.random.uniform(0.0, 1.0)
        
        classifier.predict_danger_prob(ear, perclos, pitch, mar)
        
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_ms = (total_time / n_iters) * 1000
    fps = 1.0 / (total_time / n_iters)
    
    print(f"Total time: {total_time:.4f}s")
    print(f"Average inference time: {avg_time_ms:.4f} ms")
    print(f"Theoretical Max FPS (model only): {fps:.1f}")
    
    if avg_time_ms < 5.0:
        print("\nRESULT: Model is extremely fast (Real-time ready).")
    else:
        print("\nRESULT: Model might be slow.")

if __name__ == "__main__":
    main()
