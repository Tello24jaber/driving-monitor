# Driver Monitoring System (Signals & Systems)

This repository implements a real-time driver monitoring / drowsiness detection system built around a **Signals & Systems** approach:

1) Capture frames from a webcam
2) Extract face landmarks + estimate head pose
3) Convert geometry into discrete-time **signals** (EAR, MAR, head posture, PERCLOS)
4) Apply smoothing (convolution / moving-average) and sliding-window metrics
5) Run a **3-state decision engine** with hysteresis: `OK` → `WARNING` → `DANGER`
6) Optionally fuse an **ONNX deep model** that consumes signals only and outputs $p(\text{danger})$

The “deep” component is intentionally **signals-only** (not image/CNN based). Images are used only to compute the signals.

---

## Methods Used

### 1) Landmark-based geometry (MediaPipe)
We use MediaPipe FaceMesh landmarks to measure:

- **Eyes** → Eye Aspect Ratio (EAR)
- **Mouth** → Mouth Aspect Ratio (MAR)
- **Head** → roll/pitch/yaw (pose estimation)

This is lightweight and explainable: the system decisions are driven by the time evolution of these signals.

### 2) Head pose estimation (solvePnP → Euler angles)
Head pose is computed from a small set of stable facial landmarks using:

- `cv2.solvePnP(...)` to estimate rotation (`rvec`) and translation (`tvec`)
- `cv2.Rodrigues(...)` and `cv2.RQDecomp3x3(...)` to recover roll/pitch/yaw

The debug direction overlay draws horizontal motion from **yaw** and vertical motion from **pitch**.

### 3) Discrete-time signals (sampling)
At runtime, the system samples once per video frame (typically ~30 FPS), producing discrete-time sequences:

- $x_{EAR}[n]$, $x_{MAR}[n]$, and a head posture signal
- PERCLOS is computed from a sliding window of the EAR signal

---

## Signals (What We Measure)

### Eye Aspect Ratio (EAR)
EAR decreases when eyes close. The code computes EAR for the left and right eye and also keeps an average.

At the landmark level, EAR is computed from Euclidean distances between eyelid points and eye corners.

### Mouth Aspect Ratio (MAR)
MAR increases when the mouth opens. We use it as an **early fatigue indicator** (yawning) for `WARNING` only.

### Head posture signal
The system computes roll/pitch/yaw angles, but the signal pipeline uses a **derived head-down feature** to make “nodding/down” more robust across camera setups:

`head_down = max(0, base_roll - roll)`

This reduces false positives when the driver simply looks left/right.

### PERCLOS (Percentage of Eye Closure)
PERCLOS is computed as a percentage over a sliding window on the (smoothed) EAR signal:

$$\text{PERCLOS} = \frac{\#\{\text{frames where } EAR < T\}}{\#\{\text{frames in window}\}} \times 100\%$$

In this project, PERCLOS is represented as a **percentage in $[0, 100]$**.

Note: there is also a legacy PERCLOS-style counter inside the landmark detector, but the Signals GUI uses the explicit sliding-window PERCLOS in `signals/perclos.py`.

---

## Signal Processing (Filters + Windows)

### Circular buffers (streaming time-series)
Signals are stored in a fixed-size FIFO buffer (default 300 samples, ~10 seconds at 30 FPS).

### Moving-average smoothing (convolution)
We smooth signals using a rectangular FIR filter:

$$y[n] = (h * x)[n] = \frac{1}{M} \sum_{k=0}^{M-1} x[n-k]$$

This attenuates high-frequency noise (blinks, landmark jitter) while preserving the low-frequency trend.

Typical window sizes in the Signals GUI:

- EAR: 7 samples
- Head signal: 15 samples
- MAR: 10 samples

### Sliding-window PERCLOS
PERCLOS is computed on a sliding window (default 90 samples ≈ 3 seconds) with a configurable EAR closure threshold.

---

## Decision Engine (OK / WARNING / DANGER)

Implemented in `signals/decision.py`.

### States

- `OK` (normal)
- `WARNING` (early fatigue, e.g., yawning)
- `DANGER` (critical drowsiness: sustained eye closure / high PERCLOS / sustained head-down)

### Logic summary

**DANGER (eyes / head):**

- Sustained low EAR over `ear_sustained_frames`
- High PERCLOS
- Sustained head-down deviation from a calibrated baseline

**WARNING (mouth only):**

- Sustained high MAR (yawning)
- WARNING never escalates to DANGER by itself

### Hysteresis (anti-flicker)
To prevent rapid flipping between states, the engine uses hysteresis: entering danger uses stricter thresholds than exiting.

### Baseline calibration
The decision engine learns a baseline posture from the first few seconds of data (default 90 samples) and detects head-down as a deviation from that baseline.

### Current default thresholds (code defaults)
These are the values currently configured in the code (tunable):

- `ear_danger_threshold = 0.11`
- `perclos_danger_threshold = 50.0` (percent)
- `head_deviation_threshold = 6.0` (signal deviation from baseline)
- `mar_warning_threshold = 0.6`

Timing (frames at ~30 FPS):

- `ear_sustained_frames = 90` (~3 seconds)
- `head_sustained_frames = 45` (~1.5 seconds)
- `mar_sustained_frames = 45` (~1.5 seconds)

---

## Optional Deep Signal Model (ONNX, real time)

This project can optionally load an ONNX model that consumes the same signal vector and outputs $p(\text{danger})$.

### Model contract
Input (float32): shape `(1, 4)`

`[EAR, PERCLOS, head_down, MAR]`

Output:

- either a single value (logit or probability)
- or a 2-class vector (softmax is applied if needed)

### Runtime inference
- Model file: `models/signal_danger.onnx`
- Loader: `signals/deep_signal_model.py`
- Backend: prefers `onnxruntime` (CPU), falls back to OpenCV DNN if available
- If the model is missing or can’t be loaded, the system continues normally (signals-only).

Important normalization:
- PERCLOS is produced as 0–100% in the app.
- The deep wrapper normalizes to 0–1 when needed.

### Fusion behavior
- The deep model can trigger DANGER if $p(\text{danger})$ stays high for a sustained number of frames.
- Exiting DANGER requires both classic signals and the deep probability to return to safe values (hysteresis).

---

## GUIs

- `gui_signals.py`: Signals & Systems GUI (plots + signal pipeline + decision engine + optional DL fusion)
- `gui_app.py`: Original GUI (kept for comparison / legacy usage)

---

## How To Run (Windows)

Install dependencies (in your venv):

- `pip install -r requirements.txt`

Run the Signals GUI:

- `./.venv/Scripts/python.exe gui_signals.py`

Benchmark the deep model wrapper (if you have a model):

- `./.venv/Scripts/python.exe scripts/test_model_realtime.py`

---

## Training / Creating a Starter ONNX Model (Optional)

There is a simple synthetic-data training script:

- `scripts/create_signal_model.py`

It trains a small MLP and exports `models/signal_danger.onnx`.

Notes:
- This script requires PyTorch (`torch`). If PyTorch is not installed in your active venv, run it with an interpreter that has torch installed or install torch into your venv.
- Runtime inference does not require torch.

---

## Project Structure (Key Files)

- `detect/face.py`: landmark extraction + EAR/MAR and legacy perclos counters; includes OR-eye logic and ROI helper
- `detect/pose.py`: head pose estimation (solvePnP) and direction visualization
- `signals/signal_buffer.py`: time-series buffers (circular/FIFO)
- `signals/filters.py`: moving-average filter via convolution
- `signals/perclos.py`: sliding-window PERCLOS (0–100%)
- `signals/decision.py`: OK/WARNING/DANGER state machine + hysteresis (+ optional DL fusion)
- `signals/deep_signal_model.py`: ONNX inference wrapper (onnxruntime/OpenCV)

---

## Signals & Systems Concepts Applied

- **Discrete-time signals:** EAR[n], MAR[n], head[n] sampled per frame
- **Convolution / FIR filtering:** moving-average smoothing
- **Sliding windows:** PERCLOS over recent samples
- **State machine + hysteresis:** stable alert transitions (Schmitt-trigger style)
- **Streaming buffers:** fixed-size circular buffers for real-time processing

---

## Tuning Tips

- If you get false DANGER on blinks: increase `ear_sustained_frames` or raise `ear_danger_threshold`.
- If head-down doesn’t trigger: adjust `head_deviation_threshold` and verify the head-down feature behavior for your camera placement.
- If WARNING triggers too often: raise `mar_warning_threshold` or increase `mar_sustained_frames`.
