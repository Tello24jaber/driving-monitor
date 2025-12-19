# Driver Drowsiness Detection - Signals & Systems Implementation

## 🎯 Overview

This project has been transformed from a basic frame-by-frame threshold detector into a **Signals & Systems-based drowsiness detection pipeline** using discrete-time signal processing.

## 🔬 Signal Processing Pipeline

### 1. **Signal Acquisition**
- EAR (Eye Aspect Ratio) and head pitch are treated as **discrete-time signals**: `x[n]`
- Stored in circular buffers (300 samples ≈ 10 seconds at 30 FPS)

### 2. **Signal Smoothing via Convolution**
- **Moving Average Filter**: `y[n] = (1/M) * Σ x[n-k]` for k = 0 to M-1
- Implemented using `numpy.convolve()` with rectangular kernel
- EAR smoothing: 7-sample window (reduces blink noise)
- Head pitch smoothing: 15-sample window (reduces jitter)

### 3. **PERCLOS Computation**
- **PERCLOS** = **PER**centage of eye **CLO**sure over time
- Computed over 90-sample sliding window (3 seconds)
- Formula: `PERCLOS = (closed_frames / total_frames) × 100`
- Threshold: EAR < 0.2 = "closed"

### 4. **Decision Engine with Hysteresis**
Triggers **DANGER** state if ANY condition holds:
- **Sustained low EAR**: Smoothed EAR < 0.18 for 20+ frames
- **High PERCLOS**: PERCLOS > 70%
- **Head nodding**: Smoothed pitch > 20° for 30+ frames

**Hysteresis** prevents alert flickering:
- Alert ON at threshold X
- Alert OFF at threshold X + margin

## 🚨 Danger Feedback System

### Visual Warnings
- **Normal state**: Green indicator, normal video feed
- **DANGER state**: 
  - RED overlay on video feed
  - Large "DROWSINESS DETECTED!" warning text
  - RED alert banner at top
  - Metrics display reason for alert

### Audio Warnings
- **Continuous beep** plays when DANGER state is active
- Frequency: 800 Hz
- Pattern: Beep (300ms) → Pause (100ms) → Repeat
- Automatically stops when state returns to OK

## 📊 Live Visualization

### Real-Time Plots
1. **EAR Signal Plot**:
   - Blue line: Raw EAR signal (noisy)
   - Cyan line: Smoothed EAR via convolution
   - Red dashed line: Danger threshold (0.2)

2. **Head Pitch Plot**:
   - Orange line: Raw pitch signal
   - Yellow line: Smoothed pitch signal
   - Red dashed line: Danger threshold (20°)

### Metrics Display
- Smoothed EAR value
- PERCLOS percentage
- Head pitch angle
- Frame count
- Alert state and reason

## 🚀 How to Run

### Launch the Signals & Systems GUI:
```bash
python gui_signals.py
```

### Original GUI (still available):
```bash
python gui_app.py
```

## 📁 New Project Structure

```
driving-monitor/
├── signals/                    # NEW: Signal processing modules
│   ├── __init__.py
│   ├── signal_buffer.py       # Circular buffer for time-series
│   ├── filters.py             # Moving average convolution
│   ├── perclos.py             # PERCLOS computation
│   ├── decision.py            # Drowsiness decision engine
│   └── audio_alert.py         # Audio warning system
├── gui_signals.py             # NEW: Signals & Systems GUI
├── gui_app.py                 # Original GUI (unchanged)
├── detect/
│   ├── face.py                # Face detection (unchanged)
│   └── pose.py                # Head pose (unchanged)
└── ...
```

## 🎓 Signals & Systems Concepts Applied

### Discrete-Time Signals
- `EAR[n]`: Eye aspect ratio at frame n
- `pitch[n]`: Head pitch angle at frame n
- Sampling rate: ~30 FPS (Nyquist: 15 Hz)

### Convolution
- **Time-domain convolution**: `y[n] = x[n] * h[n]`
- Rectangular impulse response: `h[n] = 1/M` for n = 0...M-1
- Implements low-pass filter (removes high-frequency noise)

### Sliding Window Analysis
- PERCLOS uses non-overlapping windows
- Window size: 90 samples (3 seconds)
- Real-time computation at each frame

### Hysteresis (State Machine)
- Prevents rapid state transitions
- Schmitt trigger-like behavior
- Separate thresholds for entering/exiting DANGER

## ⚙️ Tunable Parameters

### Signal Buffer
- `max_size`: 300 samples (10 seconds)

### Filters
- `ear_window_size`: 7 samples
- `pitch_window_size`: 15 samples

### PERCLOS
- `window_size`: 90 samples (3 seconds)
- `threshold`: 0.2 (eye closure)

### Decision Engine
- `ear_danger_threshold`: 0.18
- `perclos_danger_threshold`: 70%
- `pitch_danger_threshold`: 20°
- `ear_hysteresis`: 0.03
- `perclos_hysteresis`: 10%
- `pitch_hysteresis`: 5°

## 🎯 Key Improvements Over Original

1. **Signal-based vs Frame-based**: Uses processed signals instead of per-frame thresholds
2. **Noise reduction**: Convolution smoothing removes sensor noise and blinks
3. **Sustained detection**: Requires conditions to persist over time
4. **No false alarms**: Hysteresis prevents flickering alerts
5. **Multi-modal fusion**: Combines EAR, PERCLOS, and head pose signals
6. **Real-time visualization**: Live plots show signal behavior
7. **Audio feedback**: Continuous warning until driver responds

## 📈 Performance

- Real-time processing at 30 FPS
- Latency: ~0.7-1.0 seconds (due to sustained time requirements)
- Memory: Minimal (only 300 samples buffered)
- CPU: Low (efficient NumPy operations)

## 🔧 Dependencies

- opencv-python
- mediapipe
- numpy
- matplotlib
- Pillow
- tkinter (included with Python)

## 📝 Notes

- Audio alerts use `winsound` on Windows
- Face detection uses MediaPipe FaceLandmarker
- No deep learning models required (except MediaPipe's pre-trained detector)
- All signal processing is classical DSP (convolution, windowing)

---

**Author**: Driver Monitoring System Team  
**Date**: December 2025  
**Course**: Signals & Systems Application Project
