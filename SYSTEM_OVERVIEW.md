# Driver Drowsiness Detection System - Final Overview

## 🎓 **Course Project: Signals & Systems Application**

This is a **real-time driver monitoring system** that uses classical **discrete-time signal processing** to detect drowsiness and fatigue. The system follows a strict **landmark-based geometric approach** (no deep learning) with explainable, tunable parameters.

---

## 🔬 **Signal Processing Architecture**

### **1. Signal Acquisition (Discrete-Time Signals)**

The system converts continuous driver behavior into discrete-time signals sampled at **30 FPS**:

| Signal | Symbol | Description | Source |
|--------|--------|-------------|--------|
| **EAR** | `x_EAR[n]` | Eye Aspect Ratio | MediaPipe facial landmarks |
| **Pitch** | `x_pitch[n]` | Head tilt angle (negative = down) | Geometric pose estimation |
| **MAR** | `x_MAR[n]` | Mouth Aspect Ratio | MediaPipe facial landmarks |

**Formulas:**
- **EAR**: `(||P3-P13|| + ||P5-P11||) / (2 * ||P0-P8||)`
- **Pitch**: Computed from facial landmark geometry (Euler angles)
- **MAR**: `(||Upper14-Lower17|| + ||Upper12-Lower14||) / (||Upper0-Upper8|| + ||Lower12-Lower10||)`

**Sampling:** 30 FPS → Nyquist frequency = 15 Hz

---

### **2. Signal Smoothing (Convolution)**

**Time-domain convolution** with rectangular impulse response:

$$y[n] = h[n] * x[n] = \frac{1}{M} \sum_{k=0}^{M-1} x[n-k]$$

**Implementation:**
```python
kernel = np.ones(window_size) / window_size  # Normalized rectangular window
smoothed = np.convolve(signal, kernel, mode='same')
```

**Filter Parameters:**
- **EAR**: 7-sample moving average (removes blink noise)
- **Pitch**: 15-sample moving average (removes head jitter)
- **MAR**: 10-sample moving average (smooths mouth movements)

This is a **low-pass FIR filter** that attenuates high-frequency noise while preserving the underlying trend.

---

### **3. PERCLOS Computation (Sliding Window Analysis)**

**PERCLOS** = **PER**centage of eye **CLO**sure over time

$$\text{PERCLOS} = \frac{\text{# frames where EAR} < 0.20}{\text{total frames in window}} \times 100\%$$

**Parameters:**
- **Window size**: 90 samples (3 seconds)
- **Threshold**: EAR < 0.20 = "closed"
- **Result**: 0-100% metric

---

### **4. Decision Engine (State Machine with Hysteresis)**

#### **Alert States:**

```
┌─────────────────────────────────────────────────────┐
│  OK (GREEN)  →  WARNING (ORANGE)  →  DANGER (RED)  │
└─────────────────────────────────────────────────────┘
```

#### **State Transition Logic:**

**🟢 OK → 🟠 WARNING:**
- Sustained high MAR (yawning for 1.5+ seconds)
- **Response**: Orange border, text "You may be tired – consider resting"
- **NO audio alert**

**🟢 OK → 🔴 DANGER:**
- **Primary**: Smoothed EAR < 0.18 for **3+ seconds** (eyes closed)
- **Primary**: PERCLOS > 50% (excessive eye closure)
- **Secondary**: Smoothed pitch < -25° for **2+ seconds** (head down)
- **Response**: Red overlay, "DROWSINESS DETECTED!", continuous 800 Hz alarm

**🔴 DANGER → 🟢 OK (Return to Safe):**
- Requires **ALL danger metrics** to improve with hysteresis:
  - EAR ≥ 0.18 + 0.05 = **0.23**
  - PERCLOS ≤ 50% - 15% = **35%**
  - Pitch ≥ -25° + 8° = **-17°**

**🟠 WARNING → 🟢 OK:**
- MAR ≤ 0.6 - 0.1 = **0.5**
- Immediate transition (no hysteresis delay)

#### **Hysteresis (Schmitt Trigger Behavior):**

Prevents alert flickering by using **different thresholds** for entering vs. exiting states:

```
Enter DANGER: EAR < 0.18
Exit DANGER:  EAR > 0.23 (threshold + hysteresis)
```

---

## 📊 **Real-Time Visualization**

### **Three Live Signal Plots:**

1. **EAR Signal Plot**
   - Blue line: Raw EAR (noisy, includes blinks)
   - Cyan line: Smoothed EAR (convolution output)
   - Red dashed line: Danger threshold (0.18)

2. **Head Pitch Plot**
   - Orange line: Raw pitch (jittery)
   - Yellow line: Smoothed pitch
   - Red dashed lines: Danger thresholds (±25°)

3. **MAR Signal Plot**
   - Purple line: Raw MAR
   - Magenta line: Smoothed MAR
   - Orange dashed line: Warning threshold (0.6)

### **Metrics Display:**
- Smoothed EAR value
- PERCLOS percentage
- Head pitch angle
- Mouth aspect ratio
- Frame count
- Alert state and reason

---

## 🎯 **Detection Requirements (Implemented)**

### ✅ **Primary Danger Indicators (Eyes)**
- [x] EAR < 0.18 for > 3 seconds → **DANGER**
- [x] PERCLOS > 50% → **DANGER**
- [x] Red visual warning
- [x] Continuous alarm sound (800 Hz)

### ✅ **Secondary Danger Indicator (Head)**
- [x] Head pitch < -25° for > 2 seconds → **DANGER**
- [x] Same response as eye-based danger

### ✅ **Early Warning Only (Mouth)**
- [x] MAR > 0.6 sustained → **WARNING** (NOT danger)
- [x] Orange visual indicator
- [x] Text: "You may be tired – consider resting"
- [x] **NO sound** (calm warning)

### ✅ **Signal Processing**
- [x] Moving-average smoothing via convolution
- [x] Sliding-window PERCLOS
- [x] All decisions based on processed signals (not raw)

### ✅ **State Machine**
- [x] OK → Green (normal)
- [x] WARNING → Orange (mouth/fatigue)
- [x] DANGER → Red + alarm (eyes or head)
- [x] Hysteresis to prevent flickering

### ✅ **Return to Safe Logic**
- [x] From DANGER: Requires all metrics to improve with hysteresis
- [x] From WARNING: Immediate return when MAR normalizes

---

## 🛠️ **Tunable Parameters**

All parameters are adjustable for different scenarios:

### **Thresholds:**
```python
ear_danger_threshold = 0.18      # Eyes closed
perclos_danger_threshold = 50.0  # High eye closure %
pitch_danger_threshold = 25.0    # Head down (degrees)
mar_warning_threshold = 0.6      # Yawning (WARNING only)
```

### **Time Requirements (frames at 30 FPS):**
```python
ear_sustained_frames = 90   # 3.0 seconds
pitch_sustained_frames = 60 # 2.0 seconds
mar_sustained_frames = 45   # 1.5 seconds
```

### **Hysteresis Margins:**
```python
ear_hysteresis = 0.05       # Exit DANGER requires EAR > 0.23
perclos_hysteresis = 15.0   # Exit DANGER requires PERCLOS < 35%
pitch_hysteresis = 8.0      # Exit DANGER requires pitch > -17°
mar_hysteresis = 0.1        # Exit WARNING requires MAR < 0.5
```

### **Filter Kernel Sizes:**
```python
ear_filter_window = 7       # Moving average for EAR
pitch_filter_window = 15    # Moving average for pitch
mar_filter_window = 10      # Moving average for MAR
```

---

## 🚀 **How to Run**

### **Requirements:**
```bash
pip install opencv-python mediapipe numpy matplotlib Pillow
```

### **Launch:**
```bash
python gui_signals.py
```

### **Controls:**
1. Click "START MONITORING" to begin
2. System calibrates for ~3 seconds (100 samples)
3. Observe real-time plots and alerts
4. Click "STOP MONITORING" to end

---

## 📁 **Project Structure**

```
driving-monitor/
├── signals/                    # Signal processing modules
│   ├── signal_buffer.py       # Circular buffer (FIFO)
│   ├── filters.py             # Moving average convolution
│   ├── perclos.py             # PERCLOS sliding window
│   ├── decision.py            # State machine with hysteresis
│   └── audio_alert.py         # Audio warning system
├── detect/
│   ├── face.py                # EAR, MAR landmark extraction
│   └── pose.py                # Head pitch estimation
├── gui_signals.py             # Main GUI (Signals & Systems)
├── SIGNALS_README.md          # Detailed signal processing docs
└── SYSTEM_OVERVIEW.md         # This file
```

---

## 🎓 **Signals & Systems Concepts Applied**

| Concept | Implementation |
|---------|---------------|
| **Discrete-time signals** | EAR[n], Pitch[n], MAR[n] at 30 FPS |
| **Time-domain convolution** | Moving average: `y[n] = x[n] * h[n]` |
| **Impulse response** | Rectangular window: `h[n] = 1/M` |
| **Low-pass filtering** | Removes high-frequency noise (blinks, jitter) |
| **Sliding window** | PERCLOS over 3-second windows |
| **State machine** | OK ↔ WARNING ↔ DANGER with hysteresis |
| **Hysteresis** | Schmitt trigger (different enter/exit thresholds) |
| **Circular buffer** | FIFO deque for streaming data |
| **Sampling theorem** | 30 FPS → Nyquist 15 Hz |

---

## 🔧 **Design Constraints (Followed)**

✅ **Landmark-based geometric method** (no deep learning)  
✅ **Explainable decisions** (all thresholds visible)  
✅ **Real-time processing** (30 FPS)  
✅ **Simple state machine** (OK/WARNING/DANGER)  
✅ **Classical DSP only** (convolution, windowing)  

---

## 📈 **Performance**

- **Processing speed**: 30 FPS real-time
- **Latency**: 
  - Eye danger: 3 seconds (sustained requirement)
  - Head danger: 2 seconds (sustained requirement)
  - Warning: 1.5 seconds (yawn detection)
- **Memory**: Minimal (300 samples × 6 signals = ~1.8K floats)
- **CPU**: Low (efficient NumPy operations)
- **False alarm rate**: Very low (hysteresis prevents flickering)

---

## 🏆 **Key Features**

### **1. Graduated Warnings**
- **Calm early warning** for yawning (orange, no sound)
- **Strong alarm** only for critical issues (red, continuous sound)

### **2. Multi-Modal Fusion**
- Combines **eyes** (primary), **head** (secondary), **mouth** (warning)
- Each modality has appropriate response level

### **3. Robust to Noise**
- Smoothing filters remove blinks and jitter
- Sustained time requirements prevent false alarms
- Hysteresis prevents alert flickering

### **4. Real-Time Explainability**
- Live plots show raw vs. smoothed signals
- Threshold lines clearly visible
- Alert reason displayed in plain text

### **5. Professional Driver Assistance**
- Behaves like a real automotive system
- Clear visual and audio feedback
- Calm warnings escalate to alarms only when necessary

---

## 📝 **Conclusion**

This system demonstrates a complete **signals and systems application** using:
- **Discrete-time signal processing** (convolution, windowing)
- **State machine design** (hysteresis, anti-flickering)
- **Multi-modal sensor fusion** (eyes, head, mouth)
- **Real-time embedded-friendly** (low CPU, low memory)

**No deep learning required** – all detection is based on **geometric landmark analysis** and **classical DSP**, making it:
- ✅ Explainable
- ✅ Tunable
- ✅ Real-time
- ✅ Lightweight
- ✅ Educational (perfect for signals & systems course)

---

**Author**: Driver Monitoring System Team  
**Date**: December 2025  
**Course**: Signals & Systems Application Project  
**Method**: Landmark-based geometric detection (no deep learning)
