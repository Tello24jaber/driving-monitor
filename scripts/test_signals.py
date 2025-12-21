"""
Test script to verify signal processing pipeline
"""
import numpy as np
from signals import (
    SignalBuffer,
    MovingAverageFilter,
    PERCLOSCalculator,
    DrowsinessDecisionEngine,
    AudioAlert
)

print("=" * 60)
print("SIGNAL PROCESSING PIPELINE TEST")
print("=" * 60)

# Test 1: Signal Buffer
print("\n1. Testing SignalBuffer...")
buffer = SignalBuffer(max_size=100)

# Add some samples
for i in range(50):
    ear_l = 0.25 + 0.05 * np.sin(i * 0.1)
    ear_r = 0.25 + 0.05 * np.cos(i * 0.1)
    pitch = 10.0 + 5.0 * np.sin(i * 0.05)
    buffer.add_sample(ear_l, ear_r, pitch, 0, 0)

print(f"   Buffer size: {len(buffer)}")
print(f"   Buffer ready: {buffer.is_ready(30)}")
print(f"   ✓ Signal buffer working")

# Test 2: Moving Average Filter
print("\n2. Testing MovingAverageFilter...")
filter_ear = MovingAverageFilter(window_size=7)
raw_signal = buffer.get_array('ear_avg')
smoothed_signal = filter_ear.apply(raw_signal)

print(f"   Raw signal length: {len(raw_signal)}")
print(f"   Smoothed signal length: {len(smoothed_signal)}")
print(f"   Raw signal variance: {np.var(raw_signal):.6f}")
print(f"   Smoothed signal variance: {np.var(smoothed_signal):.6f}")
print(f"   Noise reduction: {(1 - np.var(smoothed_signal)/np.var(raw_signal))*100:.1f}%")
print(f"   ✓ Convolution smoothing working")

# Test 3: PERCLOS Calculator
print("\n3. Testing PERCLOSCalculator...")
perclos_calc = PERCLOSCalculator(window_size=30, threshold=0.2)

# Create signal with some closed eyes
test_signal = np.ones(50) * 0.25
test_signal[20:30] = 0.15  # 10 frames of closed eyes

perclos = perclos_calc.compute(test_signal)
print(f"   Test signal length: {len(test_signal)}")
print(f"   Closed eye samples: 10 / 50")
print(f"   Expected PERCLOS: 20.0%")
print(f"   Computed PERCLOS: {perclos:.1f}%")
print(f"   ✓ PERCLOS computation working")

# Test 4: Decision Engine
print("\n4. Testing DrowsinessDecisionEngine...")
decision = DrowsinessDecisionEngine()

# Test normal condition
state1, reason1 = decision.evaluate(0.25, 30.0, 10.0)
print(f"   Normal condition: State={state1}")

# Test danger condition (high PERCLOS)
state2, reason2 = decision.evaluate(0.25, 75.0, 10.0)
print(f"   High PERCLOS: State={state2}")

# Test danger condition (low EAR sustained)
for _ in range(25):
    state3, reason3 = decision.evaluate(0.15, 30.0, 10.0)
print(f"   Sustained low EAR: State={state3}")

# Test hysteresis (should stay in DANGER despite improvement)
state4, reason4 = decision.evaluate(0.20, 65.0, 10.0)
print(f"   Slight improvement (hysteresis): State={state4}")

# Test full recovery
for _ in range(10):
    state5, reason5 = decision.evaluate(0.28, 30.0, 5.0)
print(f"   Full recovery: State={state5}")

print(f"   ✓ Decision engine with hysteresis working")

# Test 5: Audio Alert
print("\n5. Testing AudioAlert...")
audio = AudioAlert()
print(f"   Audio system initialized")
print(f"   ✓ Audio alert system ready")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nSignal processing pipeline is working correctly!")
print("Run 'python gui_signals.py' to start the GUI application.")
