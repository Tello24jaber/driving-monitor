#!/usr/bin/env python3
"""
Test the decision engine logic with various input values
"""
import numpy as np
from signals import DrowsinessDecisionEngine

# Create decision engine
engine = DrowsinessDecisionEngine()

print("Testing DrowsinessDecisionEngine")
print("=" * 80)
print(f"EAR danger threshold: {engine.ear_danger_threshold}")
print(f"EAR warning threshold: {engine.ear_warning_threshold}")
print(f"PERCLOS danger threshold: {engine.perclos_danger_threshold:.1f}%")
print(f"Pitch danger threshold: {engine.pitch_danger_threshold:.1f}°")
print(f"Sustained frames required: EAR={engine.ear_sustained_frames}, Pitch={engine.pitch_sustained_frames}")
print("=" * 80)

# Test scenario 1: Normal values
print("\n[Test 1] Normal values:")
for i in range(50):
    state, reason = engine.evaluate(0.30, 10.0, 5.0)  # Normal: high EAR, low PERCLOS, low pitch
    if i % 10 == 0:
        print(f"  Frame {i}: State={state}, Reason={reason}")

# Test scenario 2: Eyes closing
print("\n[Test 2] Eyes slowly closing:")
engine.reset()
for i in range(100):
    ear = 0.30 - (i * 0.002)  # Gradually decrease EAR
    perclos = (i * 0.4)  # Gradually increase PERCLOS
    state, reason = engine.evaluate(ear, perclos, 5.0)
    if i % 20 == 0 or state == 'DANGER':
        print(f"  Frame {i}: EAR={ear:.3f}, PERCLOS={perclos:.1f}%, State={state}")

# Test scenario 3: Head tilting
print("\n[Test 3] Head tilting:")
engine.reset()
for i in range(100):
    pitch = (i * 0.3)  # Gradually increase pitch/tilt
    state, reason = engine.evaluate(0.30, 15.0, pitch)
    if i % 20 == 0 or state == 'DANGER':
        print(f"  Frame {i}: Pitch={pitch:.1f}°, State={state}")

print("\n" + "=" * 80)
print("Test complete!")
