# Signal-Based Detection Improvements

## Overview
This document describes the accuracy improvements applied to the drowsiness detection system using **signal processing and geometric methods** (no deep learning).

## Applied Improvements (Based on Reference Research)

### 1. **Faster Eye Closure Detection** ✅
- **Changed**: Eye closure threshold from 3 seconds to 2 seconds
- **Implementation**: `ear_sustained_frames = 60` (was 90 frames)
- **Benefit**: More responsive drowsiness detection - triggers alert faster
- **Signal Processing**: Uses moving average filter (window=7) on EAR signal

### 2. **Individual Eye OR Logic** ✅
- **Changed**: Detection now triggers if EITHER eye is closed (not just average)
- **Implementation**: New `calculate_perclos_with_or_logic()` method
- **Logic**: `if (earLeft <= threshold) OR (earRight <= threshold)` → trigger
- **Benefit**: More sensitive - catches partial eye closures and winking drowsiness
- **Signal Processing**: Tracks individual EAR signals for left and right eyes

### 3. **Improved Landmark Indices** ✅
- **Added**: Core landmark sets for faster detection
  - `LEFT_EYE_CORE = [33, 133]`
  - `RIGHT_EYE_CORE = [362, 263]`
  - `MOUTH_CORE = [13, 14, 78, 308]`
- **Benefit**: Proven landmarks from reference implementation, available for future optimizations
- **Signal Processing**: Full landmark sets still used for robust geometric calculations

### 4. **Enhanced ROI Extraction** ✅
- **Added**: `get_roi_with_scale(landmarks, scale=1.8)` utility function
- **Purpose**: Expands bounding box around eyes/mouth by 1.8x for better context
- **Implementation**: Centers scaled box around landmark region
- **Benefit**: Captures more facial context for improved signal quality
- **Future Use**: Can be applied to eye/mouth region extraction if needed

## Signal Processing Architecture (No Deep Learning)

### Detection Method: Geometric + Signal Processing
1. **MediaPipe Face Mesh**: 468 facial landmarks (geometric tracking)
2. **EAR Calculation**: Eye Aspect Ratio formula using Euclidean distances
3. **MAR Calculation**: Mouth Aspect Ratio formula using lip landmarks
4. **Moving Average Filters**: Convolution-based smoothing (different windows for each signal)
5. **PERCLOS Calculator**: Sliding window analysis of eye closure percentage
6. **Decision Engine**: 3-state machine with hysteresis (OK/WARNING/DANGER)

### Signal Flow
```
Video Frame
    ↓
MediaPipe Landmarks (geometric)
    ↓
EAR/MAR/Pitch Calculation (geometric formulas)
    ↓
Moving Average Filters (signal smoothing)
    ↓
PERCLOS Calculator (sliding window)
    ↓
Decision Engine (state machine with hysteresis)
    ↓
Alert (audio + visual)
```

## Key Parameters (Tuned for Accuracy)

### Detection Thresholds
- **EAR Threshold**: 0.18 (eyes closed)
- **MAR Threshold**: 0.6 (yawning)
- **Pitch Threshold**: 25° (head nodding)
- **PERCLOS Threshold**: 50% (eye closure percentage)

### Timing Requirements (at 30 FPS)
- **Eyes**: 60 frames (2 seconds) - **IMPROVED from 3s**
- **Head**: 60 frames (2 seconds)
- **Mouth**: 45 frames (1.5 seconds)

### Filter Window Sizes (Signal Smoothing)
- **EAR Filter**: 7 samples
- **Pitch Filter**: 15 samples
- **MAR Filter**: 10 samples

### Hysteresis Margins (Anti-Flickering)
- **EAR**: ±0.05
- **Pitch**: ±8.0°
- **MAR**: ±0.1
- **PERCLOS**: ±15.0%

## Testing & Validation

### System Status
✅ Application runs without errors
✅ 3 plots display real-time signals (EAR, Pitch, MAR)
✅ State transitions work (OK → WARNING → DANGER)
✅ Audio alerts trigger on DANGER state
✅ Visual alerts show correct colors (green/orange/red)

### Expected Improvements
1. **Faster Response**: 2-second eye closure vs previous 3-second
2. **Higher Sensitivity**: OR logic catches single-eye drowsiness
3. **Better Signal Quality**: ROI extraction function available for enhanced processing
4. **Proven Parameters**: Landmark indices validated by reference implementation

## Technical Notes

### Why No Deep Learning?
- **Course Requirement**: Project focuses on Signals & Systems techniques
- **Method**: Geometric calculations + signal processing (filters, convolution, state machines)
- **Benefit**: Interpretable, lightweight, runs in real-time without GPU

### Reference Research
Based on analysis of proven drowsiness detection implementation:
- Used same MediaPipe landmarks (geometric approach)
- Adapted timing parameters (2s vs 3s)
- Implemented OR logic for eyes (more sensitive)
- Added ROI scale factor for better context

### Future Enhancements
- Fine-tune thresholds based on real-world testing
- Implement adaptive thresholds (personalized calibration)
- Add more sophisticated signal fusion techniques
- Optimize filter parameters for specific lighting conditions

---
**Last Updated**: December 20, 2025
**Method**: Signal Processing + Geometric Analysis (No Deep Learning)
