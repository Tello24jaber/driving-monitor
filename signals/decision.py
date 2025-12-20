"""
Drowsiness decision engine using processed signals
"""
import numpy as np
import time

class DrowsinessDecisionEngine:
    """
    Makes drowsiness decisions using signal processing
    Implements hysteresis to prevent flickering
    States: OK -> WARNING (mouth/fatigue) -> DANGER (eyes/head)
    """
    def __init__(self):
        # EAR thresholds - primary danger indicator
        self.ear_danger_threshold = 0.11  # Eyes closed threshold (adjusted for actual facial geometry)
        
        # PERCLOS thresholds - primary danger indicator
        self.perclos_danger_threshold = 50.0  # 50% eye closure over time window
        
        # Head angle thresholds - secondary danger indicator (deviation from baseline)
        self.head_deviation_threshold = 6.0  # Degrees deviation from normal position (matches reference repo)
        
        # MAR thresholds - early warning only (no danger)
        self.mar_warning_threshold = 0.6  # Yawning threshold
        
        # Sustained time requirements (frames at 30fps)
        self.ear_sustained_frames = 90  # 3 seconds - prevent blinks from triggering
        self.head_sustained_frames = 45  # 1.5 seconds - head position change
        self.mar_sustained_frames = 45  # 1.5 seconds - yawn must persist
        self.min_samples_before_eval = 30  # Calibration period
        self.baseline_calibration_samples = 90  # 3 seconds to establish baseline head position
        
        # Hysteresis margins - prevent alert flickering
        self.ear_hysteresis = 0.05  # Return to safe requires EAR to be higher
        self.perclos_hysteresis = 15.0  # Return to safe requires PERCLOS lower
        self.head_hysteresis = 5.0  # Return to safe requires smaller deviation
        self.mar_hysteresis = 0.1  # Return from warning requires MAR lower
        
        # Baseline head position (calibrated from initial frames)
        self.baseline_pitch = None
        
        # State tracking
        self.state = 'OK'  # OK, WARNING, DANGER
        self.danger_start_time = None
        self.warning_start_time = None
        self.last_ear_values = []
        self.last_pitch_values = []
        self.last_mar_values = []
        
    def evaluate(self, smoothed_ear, perclos, smoothed_pitch, smoothed_mar=0.0):
        """
        Evaluate drowsiness state based on processed signals
        
        Args:
            smoothed_ear: Current smoothed EAR value
            perclos: Current PERCLOS percentage
            smoothed_pitch: Current smoothed head pitch (degrees, negative = down)
            smoothed_mar: Current smoothed MAR value (mouth aspect ratio)
            
        Returns:
            state: 'OK', 'WARNING', or 'DANGER'
            reason: String describing why state was triggered
        """
        # Track recent values for sustained checks
        self.last_ear_values.append(smoothed_ear)
        self.last_pitch_values.append(smoothed_pitch)
        self.last_mar_values.append(smoothed_mar)
        
        # Keep only recent history
        max_history = max(self.ear_sustained_frames, self.head_sustained_frames, self.mar_sustained_frames) + 30
        if len(self.last_ear_values) > max_history:
            self.last_ear_values.pop(0)
        if len(self.last_pitch_values) > max_history:
            self.last_pitch_values.pop(0)
        if len(self.last_mar_values) > max_history:
            self.last_mar_values.pop(0)
        
        # CALIBRATE BASELINE HEAD POSITION from initial frames
        if len(self.last_pitch_values) == self.baseline_calibration_samples:
            self.baseline_pitch = np.median(self.last_pitch_values)
            print(f"[BASELINE] Pitch: {self.baseline_pitch:.1f}°")
        
        # Only evaluate once we have enough history and baseline
        if len(self.last_ear_values) < self.min_samples_before_eval:
            self.state = 'OK'
            return self.state, f"Calibrating... ({len(self.last_ear_values)}/{self.baseline_calibration_samples} samples)"
        
        if self.baseline_pitch is None:
            self.state = 'OK'
            return self.state, "Establishing baseline head position..."
        
        # Debug: Print current values every 30 frames
        if len(self.last_ear_values) % 30 == 0:
            pitch_dev = abs(smoothed_pitch - self.baseline_pitch)
            print(f"[DEBUG] EAR: {smoothed_ear:.3f} | Pitch: {smoothed_pitch:.1f}° (dev: {pitch_dev:.1f}°, baseline: {self.baseline_pitch:.1f}°) | MAR: {smoothed_mar:.3f} | PERCLOS: {perclos:.1f}%")
        
        # === DANGER CONDITIONS (Primary: Eyes, Secondary: Head) ===
        
        danger_conditions = []
        
        # PRIMARY DANGER: Eyes closed for > 3 seconds
        if len(self.last_ear_values) >= self.ear_sustained_frames:
            recent_ear = self.last_ear_values[-self.ear_sustained_frames:]
            mean_ear = np.mean(recent_ear)
            
            if mean_ear < self.ear_danger_threshold:
                danger_conditions.append(f"Eyes closed ({mean_ear:.3f} < {self.ear_danger_threshold})")
        
        # PRIMARY DANGER: High PERCLOS
        if perclos > self.perclos_danger_threshold:
            danger_conditions.append(f"High eye closure ({perclos:.1f}% > {self.perclos_danger_threshold}%)")
        
        # SECONDARY DANGER: Head position abnormal (looking down/up or tilted)
        if len(self.last_pitch_values) >= self.head_sustained_frames:
            recent_pitch = self.last_pitch_values[-self.head_sustained_frames:]
            mean_pitch = np.mean(recent_pitch)
            
            # Check deviation from baseline (not absolute values)
            pitch_deviation = abs(mean_pitch - self.baseline_pitch)
            
            # Significant pitch deviation (head tilted down/up or sideways)
            if pitch_deviation > self.head_deviation_threshold:
                # Determine direction based on sign
                if mean_pitch < self.baseline_pitch:
                    direction = "down"
                else:
                    direction = "up"
                danger_conditions.append(f"Head {direction} (pitch: {mean_pitch:.1f}° vs baseline {self.baseline_pitch:.1f}°, dev: {pitch_deviation:.1f}°)")
        
        # === WARNING CONDITIONS (Mouth/Fatigue only - NO DANGER) ===
        
        warning_conditions = []
        
        # EARLY WARNING: Yawning (sustained high MAR)
        if len(self.last_mar_values) >= self.mar_sustained_frames:
            recent_mar = self.last_mar_values[-self.mar_sustained_frames:]
            mean_mar = np.mean(recent_mar)
            
            if mean_mar > self.mar_warning_threshold:
                warning_conditions.append(f"Yawning detected (MAR: {mean_mar:.3f})")
        
        # === STATE TRANSITIONS WITH HYSTERESIS ===
        
        if danger_conditions:
            # CRITICAL: Eyes or head issue -> DANGER state
            if self.state != 'DANGER':
                self.danger_start_time = time.time()
            self.state = 'DANGER'
            reason = " | ".join(danger_conditions)
            
        elif warning_conditions and self.state != 'DANGER':
            # MILD: Only mouth/fatigue -> WARNING state (never escalate from OK to DANGER via WARNING)
            if self.state != 'WARNING':
                self.warning_start_time = time.time()
            self.state = 'WARNING'
            reason = "You may be tired – consider resting"
            
        else:
            # NO ACTIVE CONDITIONS: Check if we can return to safe
            
            if self.state == 'DANGER':
                # Exiting DANGER: Check CURRENT EAR for immediate response, sustained for others
                
                # Check CURRENT EAR for immediate response (no hysteresis delay)
                current_ear_safe = smoothed_ear >= self.ear_danger_threshold
                
                # Check PERCLOS improvement
                perclos_safe = perclos <= (self.perclos_danger_threshold - self.perclos_hysteresis)
                
                # Check head position improvement (deviation from baseline)
                if len(self.last_pitch_values) >= 10:
                    recent_pitch = self.last_pitch_values[-10:]
                    mean_pitch = np.mean(recent_pitch)
                    
                    # Head is safe if deviation is below threshold
                    pitch_deviation = abs(mean_pitch - self.baseline_pitch)
                    head_safe = pitch_deviation <= (self.head_deviation_threshold - self.head_hysteresis)
                else:
                    head_safe = True
                
                if current_ear_safe and head_safe:
                    # Eyes open AND head in normal position - IMMEDIATE return to safe
                    self.state = 'OK'
                    self.danger_start_time = None
                    reason = "✓ Driver alert - All metrics normal"
                else:
                    # Still dangerous - remain in DANGER
                    issues = []
                    if not current_ear_safe:
                        issues.append("eyes closing")
                    if not head_safe:
                        issues.append("head abnormal")
                    reason = "Danger: " + ", ".join(issues)
                    self.state = 'DANGER'
                    
            elif self.state == 'WARNING':
                # Exiting WARNING requires MAR to drop with hysteresis
                if len(self.last_mar_values) >= self.mar_sustained_frames:
                    recent_mar = self.last_mar_values[-self.mar_sustained_frames:]
                    mean_mar = np.mean(recent_mar)
                    
                    if mean_mar <= (self.mar_warning_threshold - self.mar_hysteresis):
                        # Fatigue signs reduced - SAFE
                        self.state = 'OK'
                        self.warning_start_time = None
                        reason = "✓ Driver alert - Fatigue signs reduced"
                    else:
                        # Still tired - remain in WARNING
                        reason = "You may be tired – consider resting"
                        self.state = 'WARNING'
                else:
                    self.state = 'OK'
                    reason = "✓ Normal state"
                    
            else:
                # Already OK and no issues
                self.state = 'OK'
                # Get current values for display
                current_ear = self.last_ear_values[-1] if self.last_ear_values else 0.3
                current_pitch = self.last_pitch_values[-1] if self.last_pitch_values else 0.0
                reason = f"Normal (EAR: {current_ear:.3f}, PERCLOS: {perclos:.1f}%, Pitch: {current_pitch:.1f}°)"
        
        return self.state, reason
    
    def reset(self):
        """Reset decision engine state"""
        self.state = 'OK'
        self.danger_start_time = None
        self.warning_start_time = None
        self.last_ear_values = []
        self.last_pitch_values = []
        self.last_mar_values = []
    
    def get_danger_duration(self):
        """Get how long we've been in danger state (seconds)"""
        if self.danger_start_time is None:
            return 0.0
        return time.time() - self.danger_start_time
