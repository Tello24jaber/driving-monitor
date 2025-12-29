# Signal Processing Formulas Reference

This document contains all the mathematical formulas and algorithms used in the Driver Monitoring System for signal processing and drowsiness detection.

---

## Processing Approach

The system uses **TWO approaches**:

### 1. **Primary Approach: Rule-Based Signal Processing (Manual/Algorithmic)**
- Uses explicit mathematical formulas and threshold-based decision logic
- Implemented in `signals/decision.py`
- Main method for drowsiness detection
- **No machine learning required**

### 2. **Optional Approach: Deep Learning Model (Signals-Only)**
- Optional ONNX neural network model
- Implemented in `signals/deep_signal_model.py`
- Only uses if model file exists: `models/signal_danger.onnx`
- **Input**: Processed signals [EAR, PERCLOS, Pitch, MAR]
- **Output**: Probability of danger p(danger) ∈ [0, 1]
- **Note**: NOT image-based; consumes the same signals as rule-based approach

---

## 1. PRIMARY SIGNALS (Geometric Features)

### 1.1 Eye Aspect Ratio (EAR)

**Purpose**: Measure eye openness  
**Location**: `detect/face.py` → `calculate_eye_aspect_ratio()`

**Formula**:
```
For each eye:
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

where:
- p1, p4: outer/inner eye corners (horizontal landmarks)
- p2, p3, p5, p6: vertical eyelid landmarks
- ||·||: Euclidean distance (L2 norm)
```

**Implementation**:
```python
earLeft = (LA.norm(leftEye[13] - leftEye[3]) + 
           LA.norm(leftEye[11] - leftEye[5])) / 
          (2 * LA.norm(leftEye[0] - leftEye[8]))

earRight = (LA.norm(rightEye[13] - rightEye[3]) + 
            LA.norm(rightEye[11] - rightEye[5])) / 
           (2 * LA.norm(rightEye[0] - rightEye[8]))

earAvg = (earLeft + earRight) / 2
```

**Typical Values**:
- Open eyes: 0.25 - 0.45
- Closing eyes: 0.15 - 0.25
- Closed eyes: < 0.15
- **Danger threshold**: 0.11 (sustained)

---

### 1.2 Mouth Aspect Ratio (MAR)

**Purpose**: Detect yawning (fatigue indicator)  
**Location**: `detect/face.py` → `calculate_mouth_aspect_ratio()`

**Formula**:
```
MAR = (||upper14 - lower17|| + ||upper12 - lower14||) / 
      (||upper0 - upper8|| + ||lower12 - lower10||)

where:
- upper/lower landmarks: lip contour points
- ||·||: Euclidean distance
```

**Implementation**:
```python
marAvg = (LA.norm(upperLips[14] - lowerLips[17]) + 
          LA.norm(upperLips[12] - lowerLips[14])) / 
         (LA.norm(upperLips[0] - upperLips[8]) + 
          LA.norm(lowerLips[12] - lowerLips[10]))
```

**Typical Values**:
- Closed mouth: 0.0 - 0.2
- Talking: 0.2 - 0.6
- Yawning: > 0.6
- **Warning threshold**: 0.6 (sustained for 1.5 seconds)

---

### 1.3 Head Pose Angles (Roll, Pitch, Yaw)

**Purpose**: Detect head position and nodding  
**Location**: `detect/pose.py` → `calculate_angles()`

**Algorithm**:
1. **3D Pose Estimation** using PnP (Perspective-n-Point):
   ```
   cv2.solvePnP(face3d_points, face2d_points, camera_matrix, dist_coeffs)
   → returns rotation_vector (rvec), translation_vector (tvec)
   ```

2. **Convert rotation vector to rotation matrix**:
   ```
   R = cv2.Rodrigues(rvec)[0]
   ```

3. **Extract Euler angles** from rotation matrix:
   ```
   angles = cv2.RQDecomp3x3(R)[0]
   
   roll  = angles[0] * 360  (degrees, rotation around Z-axis)
   pitch = angles[1] * 360  (degrees, rotation around X-axis, up/down)
   yaw   = angles[2] * 360  (degrees, rotation around Y-axis, left/right)
   ```

**Head-Down Signal** (used in GUI):
```python
# Derived feature to detect nodding (robust to camera angle)
head_down = max(0.0, baseline_roll - current_roll)
```

**Typical Values**:
- Normal position: Calibrated as baseline during first 3 seconds
- **Deviation threshold**: 6° from baseline (sustained)
- **Head nodding**: head_down signal > 6° for 1.5 seconds

---

### 1.4 Gaze Estimation

**Purpose**: Detect if driver is looking away  
**Location**: `detect/face.py` → `estimate_gaze()`

**Formula**:
```
For each eye:
distance = ||iris_center - eye_center||

where:
- iris_center: detected iris center (x, y)
- eye_center: midpoint between eye landmarks
  eye_center_x = (inner_corner_x + outer_corner_x) / 2
  eye_center_y = (upper_lid_y + lower_lid_y) / 2

gaze_score = (distance_left + distance_right) / 2
```

**Implementation**:
```python
# Left eye center
xLeft = (leftEye[8][0] + leftEye[0][0]) / 2
yLeft = (leftEye[12][1] + leftEye[4][1]) / 2
centerLeft = np.array([xLeft, yLeft])

# Distance from iris to eye center
distLeft = LA.norm(iris_center_left - centerLeft)
distRight = LA.norm(iris_center_right - centerRight)

gazeAvg = (distLeft + distRight) / 2
```

**Usage**: Secondary indicator for "looking aside" state

---

## 2. SIGNAL PROCESSING & FILTERING

### 2.1 Moving Average Filter (Convolution-Based Smoothing)

**Purpose**: Remove noise and smooth signals  
**Location**: `signals/filters.py` → `MovingAverageFilter`

**Mathematical Formula**:
```
y[n] = (h * x)[n] = (1/M) * Σ(k=0 to M-1) x[n-k]

where:
- x[n]: input signal at time n
- y[n]: smoothed output signal
- M: window size (number of samples)
- h: rectangular kernel = [1/M, 1/M, ..., 1/M]  (M times)
- *: convolution operator
```

**Frequency Response**:
- Acts as low-pass filter
- Attenuates high-frequency noise (blinks, jitter)
- Preserves low-frequency trends (drowsiness patterns)

**Implementation**:
```python
kernel = np.ones(window_size) / window_size  # Rectangular kernel
smoothed = np.convolve(signal, kernel, mode='same')
```

**Window Sizes Used**:
- EAR signal: M = 7 samples (0.23 seconds at 30 FPS)
- Head pitch signal: M = 15 samples (0.5 seconds at 30 FPS)
- MAR signal: M = 10 samples (0.33 seconds at 30 FPS)

---

### 2.2 PERCLOS (PERcentage of eye CLOsure)

**Purpose**: Measure percentage of time eyes are closed over a time window  
**Location**: `signals/perclos.py` → `PERCLOSCalculator`

**Mathematical Formula**:
```
PERCLOS = (N_closed / N_total) × 100%

where:
- N_closed = |{n : EAR_smoothed[n] < threshold, n in window}|
- N_total = window size (number of samples in sliding window)
- threshold: EAR value below which eyes are considered "closed"
```

**Implementation**:
```python
# Get last W samples
window = ear_signal[-window_size:]

# Count closed eyes
closed_samples = np.sum(window < threshold)

# Compute percentage
perclos = (closed_samples / len(window)) * 100.0
```

**Parameters**:
- **Window size**: 90 samples (3 seconds at 30 FPS)
- **Threshold**: 0.20 (EAR below this = closed eyes)
- **Danger level**: PERCLOS > 50%

**Note**: Two PERCLOS implementations exist:
1. **Legacy** (in `detect/face.py`): Simple counter-based, uses 60-second window
2. **Signal-based** (in `signals/perclos.py`): Sliding window on smoothed EAR (USED IN GUI)

---

## 3. DECISION ENGINE (RULE-BASED)

**Location**: `signals/decision.py` → `DrowsinessDecisionEngine`

### 3.1 Three-State System with Hysteresis

**States**: OK → WARNING → DANGER

**State Transitions**:
```
OK:
  → WARNING if: sustained yawning detected
  → DANGER if: sustained eye closure OR head nodding

WARNING:
  → OK if: no warning conditions for sustained period
  → DANGER if: eye closure or head nodding detected

DANGER:
  → WARNING if: danger conditions cleared but still tired
  → OK if: all conditions normal and sustained
```

---

### 3.2 Danger Conditions (CRITICAL - Eyes/Head)

#### Primary Danger: Eye Closure

**Sustained Low EAR**:
```
DANGER if: mean(EAR[-90:]) < 0.11  (3 seconds of low EAR)

where:
- EAR[-90:]: last 90 samples (3 seconds at 30 FPS)
- 0.11: danger threshold (eyes essentially closed)
```

**High PERCLOS**:
```
DANGER if: PERCLOS > 50%

where PERCLOS computed over 90-sample sliding window
```

#### Secondary Danger: Head Position

**Head Nodding/Down**:
```
DANGER if: mean(head_down[-45:]) > 6.0°  (1.5 seconds)

where:
- head_down = max(0, baseline_roll - current_roll)
- baseline_roll: calibrated during first 3 seconds
- 6.0°: deviation threshold from baseline
```

---

### 3.3 Warning Conditions (Fatigue Indicators)

**Yawning Detection**:
```
WARNING if: mean(MAR[-45:]) > 0.6  (1.5 seconds)

where:
- MAR[-45:]: last 45 samples (1.5 seconds at 30 FPS)
- 0.6: yawning threshold
```

---

### 3.4 Hysteresis Mechanism

**Purpose**: Prevent state flickering (rapid switches between states)

**Principle**: Require more improvement to exit danger/warning than to enter

**Hysteresis Margins**:
```
To exit DANGER state:
- EAR must be > threshold + 0.05  (not just > threshold)
- PERCLOS must be < threshold - 15%
- Head deviation < threshold - 5°

To exit WARNING state:
- MAR must be < threshold - 0.1
- Must sustain for safety period
```

**Safety Periods** (sustained frames before state change):
```
Exit DANGER → WARNING/OK: 15 frames (0.5 seconds)
Exit WARNING → OK: 15 frames (0.5 seconds)
```

---

## 4. OPTIONAL: DEEP LEARNING MODEL

**Location**: `signals/deep_signal_model.py` → `SignalDeepClassifier`

### 4.1 Model Architecture

**Type**: Feedforward Neural Network (Signals-Only, NOT image-based)

**Input Vector** (4 features):
```
x = [EAR_smoothed, PERCLOS, Pitch_smoothed, MAR_smoothed]

where all values are the SAME processed signals used by rule-based approach
```

**Architecture** (example from `scripts/create_signal_model.py`):
```
Input layer:    4 neurons (EAR, PERCLOS, Pitch, MAR)
Hidden layer 1: 16 neurons, ReLU activation
Hidden layer 2: 8 neurons, ReLU activation
Output layer:   2 neurons (Safe, Danger) with softmax
```

**Output**:
```
p(danger) = softmax(logits)[danger_index]

where:
- logits: raw network output
- softmax: normalized probability distribution
- danger_index: 1 (second output neuron)
```

---

### 4.2 Model Integration

**Format**: ONNX (Open Neural Network Exchange)
- Platform-independent
- Runs via OpenCV DNN or ONNXRuntime
- File: `models/signal_danger.onnx`

**Inference**:
```python
input_vector = np.array([[ear, perclos, pitch, mar]], dtype=np.float32)
danger_prob = model.predict(input_vector)[0][1]  # Get p(danger)
```

**Decision Logic**:
```
DANGER if: mean(p(danger)[-15:]) > 0.85  (sustained for 0.5 seconds)

where:
- p(danger)[-15:]: last 15 danger probabilities
- 0.85: high confidence threshold
```

**Fallback**: If model file doesn't exist or fails to load, system continues working with rule-based approach only

---

## 5. CALIBRATION & BASELINE

### 5.1 Baseline Head Position

**Purpose**: Adapt to different camera angles and driver postures

**Formula**:
```
baseline_pitch = median(pitch[0:90])

where:
- pitch[0:90]: first 90 samples (3 seconds at 30 FPS)
- median: robust central tendency (handles outliers)
```

**Usage**:
```
head_deviation = |current_pitch - baseline_pitch|

or for head-down signal:
head_down = max(0, baseline_roll - current_roll)
```

---

### 5.2 Running Median Updates

**Purpose**: Slowly adapt baseline to long-term posture changes  
**Location**: `detect/face.py` → `define_normal_position()`

**Algorithm**:
```
Every 10 minutes (600 seconds):
  baseline_roll  = median(all_roll_values_collected)
  baseline_pitch = median(all_pitch_values_collected)
  baseline_yaw   = median(all_yaw_values_collected)
  baseline_gaze  = median(all_gaze_values_collected)
  
  Clear history, restart collection
```

**Helper Functions** (`utils.py`):
```python
def insert_sorted(arr, value):
    """Insert value into sorted list maintaining order"""
    bisect.insort(arr, value)
    return arr

def calculate_median(arr):
    """Calculate median of sorted array"""
    n = len(arr)
    mid = n // 2
    if n % 2 == 0:
        return (arr[mid - 1] + arr[mid]) / 2
    else:
        return arr[mid]
```

---

## 6. COMPLETE PROCESSING PIPELINE

### Step-by-Step Signal Flow:

```
1. CAPTURE FRAME (30 FPS)
   ↓
2. DETECT FACE LANDMARKS (MediaPipe)
   ↓
3. EXTRACT RAW SIGNALS
   - EAR_left, EAR_right → EAR_avg
   - MAR from lip landmarks
   - Roll, Pitch, Yaw from pose estimation
   ↓
4. STORE IN CIRCULAR BUFFER
   - Fixed size: 300 samples (10 seconds)
   - FIFO: oldest samples dropped
   ↓
5. APPLY SMOOTHING (Convolution)
   - EAR_smoothed = MovingAverage(EAR_raw, M=7)
   - Pitch_smoothed = MovingAverage(Pitch_raw, M=15)
   - MAR_smoothed = MovingAverage(MAR_raw, M=10)
   ↓
6. COMPUTE DERIVED METRICS
   - PERCLOS = sliding_window_closure(EAR_smoothed, W=90, T=0.20)
   - head_down = max(0, baseline_roll - current_roll)
   ↓
7. DECISION MAKING (Two Parallel Paths):
   
   Path A: Rule-Based (Primary)
   - Check sustained conditions (3 seconds for eyes, 1.5s for head/mouth)
   - Apply hysteresis margins
   - Output: OK / WARNING / DANGER + reason
   
   Path B: Deep Learning (Optional)
   - Input: [EAR_smoothed, PERCLOS, Pitch_smoothed, MAR_smoothed]
   - Output: p(danger) ∈ [0, 1]
   - Can trigger DANGER if p(danger) > 0.85 sustained
   ↓
8. COMBINED DECISION
   - DANGER if: Rule-based OR DL model detects danger
   - WARNING if: Only fatigue indicators (yawning)
   - OK if: All normal
   ↓
9. ALERT & DISPLAY
   - Visual: Color-coded frame border (Green/Orange/Red)
   - Audio: Alert sound if DANGER
   - GUI: Real-time plots of all signals
```

---

## 7. SUMMARY TABLE

| Signal | Formula Location | Purpose | Threshold | Window |
|--------|-----------------|---------|-----------|---------|
| **EAR** | `detect/face.py` | Eye openness | < 0.11 → DANGER | 90 frames (3s) |
| **MAR** | `detect/face.py` | Mouth opening (yawning) | > 0.6 → WARNING | 45 frames (1.5s) |
| **Pitch** | `detect/pose.py` | Head up/down | ±6° from baseline → DANGER | 45 frames (1.5s) |
| **PERCLOS** | `signals/perclos.py` | Eye closure % | > 50% → DANGER | 90 frames (3s) |
| **Smoothing** | `signals/filters.py` | Noise reduction | N/A | 7-15 frames |
| **Decision** | `signals/decision.py` | State machine | Hysteresis logic | Multiple |
| **DL Model** | `signals/deep_signal_model.py` | Neural prediction | p(danger) > 0.85 → DANGER | 15 frames (0.5s) |

---

## 8. KEY CONSTANTS

```python
# Sampling
FPS = 30  # Video capture frame rate

# EAR Thresholds
EAR_DANGER = 0.11  # Eyes closed
EAR_NORMAL = 0.25  # Eyes open
EAR_HYSTERESIS = 0.05

# PERCLOS
PERCLOS_WINDOW = 90  # samples (3 seconds)
PERCLOS_THRESHOLD = 0.20  # EAR threshold for "closed"
PERCLOS_DANGER = 50.0  # percentage

# Head Pose
HEAD_DEVIATION = 6.0  # degrees from baseline
HEAD_HYSTERESIS = 5.0  # degrees
BASELINE_CALIBRATION = 90  # samples (3 seconds)

# MAR (Mouth)
MAR_WARNING = 0.6  # Yawning threshold
MAR_HYSTERESIS = 0.1

# Sustained Checks
SUSTAINED_EAR = 90  # frames (3 seconds)
SUSTAINED_HEAD = 45  # frames (1.5 seconds)
SUSTAINED_MAR = 45  # frames (1.5 seconds)

# Smoothing Windows
SMOOTH_EAR = 7  # samples
SMOOTH_PITCH = 15  # samples
SMOOTH_MAR = 10  # samples

# Buffer Size
SIGNAL_BUFFER_SIZE = 300  # samples (10 seconds)

# Deep Learning (Optional)
DL_DANGER_THRESHOLD = 0.85  # probability
DL_SUSTAINED = 15  # frames (0.5 seconds)
```

---

## References

1. **Eye Aspect Ratio (EAR)**: Soukupová and Čech (2016) - "Real-Time Eye Blink Detection using Facial Landmarks"
2. **PERCLOS**: NHTSA Standard - Percentage of eyelid closure over pupil
3. **Head Pose**: OpenCV solvePnP - Perspective-n-Point algorithm
4. **Signal Processing**: Discrete-time systems and convolution theory

---

*Last Updated: December 29, 2025*
