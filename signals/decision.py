"""
Drowsiness decision engine using processed signals
"""
import numpy as np
import time

class DrowsinessDecisionEngine:
    """
    Makes drowsiness decisions using signal processing
    Implements hysteresis to prevent flickering
    """
    def __init__(self):
        # Thresholds - adjusted for realistic values
        # Normal open-eye EAR is typically 0.25-0.35, closed is <0.15
        self.ear_danger_threshold = 0.18  # Raised slightly from 0.15 to avoid false positives
        self.ear_warning_threshold = 0.20  # Between normal (0.25+) and danger
        self.perclos_danger_threshold = 50.0  # 50% (raised from 40% - more conservative)
        self.perclos_warning_threshold = 25.0  # Between normal and danger
        self.pitch_danger_threshold = 30.0  # degrees (raised from 25° to be more forgiving)
        self.pitch_warning_threshold = 15.0  # degrees
        
        # Sustained time requirements (frames) - require longer sustained periods
        self.ear_sustained_frames = 60  # ~2 sec at 30fps (was 45 - need more sustained)
        self.pitch_sustained_frames = 60  # ~2 sec at 30fps (was 45)
        self.min_samples_before_eval = 100  # Need at least 100 samples before even checking (don't trigger in first 3 sec)
        
        # Hysteresis margins - larger margins to prevent flickering
        self.ear_hysteresis = 0.08  # (was 0.05)
        self.perclos_hysteresis = 20.0  # (was 15.0)
        self.pitch_hysteresis = 10.0  # (was 8.0)
        
        # State tracking
        self.state = 'OK'  # OK, WARNING, DANGER
        self.danger_start_time = None
        self.warning_start_time = None
        self.last_ear_values = []
        self.last_pitch_values = []
        self.conditions_met = {'ear': False, 'perclos': False, 'pitch': False}
        
    def evaluate(self, smoothed_ear, perclos, smoothed_pitch):
        """
        Evaluate drowsiness state based on processed signals
        
        Args:
            smoothed_ear: Current smoothed EAR value
            perclos: Current PERCLOS percentage
            smoothed_pitch: Current smoothed head pitch (degrees)
            
        Returns:
            state: 'OK', 'WARNING', or 'DANGER'
            reason: String describing why state was triggered
        """
        # Track recent values for sustained checks
        self.last_ear_values.append(smoothed_ear)
        self.last_pitch_values.append(smoothed_pitch)
        
        # Keep only recent history (need at least 60 for hysteresis checks)
        if len(self.last_ear_values) > 120:
            self.last_ear_values.pop(0)
        if len(self.last_pitch_values) > 120:
            self.last_pitch_values.pop(0)
        
        # Only evaluate once we have enough history
        if len(self.last_ear_values) < self.min_samples_before_eval:
            self.state = 'OK'
            return self.state, f"Calibrating... ({len(self.last_ear_values)}/{self.min_samples_before_eval} samples)"
        
        # Get recent sustained values
        recent_ear = self.last_ear_values[-self.ear_sustained_frames:]
        recent_pitch = self.last_pitch_values[-self.pitch_sustained_frames:]
        mean_ear = np.mean(recent_ear)
        mean_pitch = np.abs(np.mean(recent_pitch))
        
        # Evaluate danger conditions
        danger_conditions = []
        
        # Condition 1: Sustained critically low EAR
        if mean_ear < self.ear_danger_threshold:
            danger_conditions.append(f"Low EAR ({mean_ear:.3f})")
        
        # Condition 2: High PERCLOS
        if perclos > self.perclos_danger_threshold:
            danger_conditions.append(f"High PERCLOS ({perclos:.1f}%)")
        
        # Condition 3: Sustained significant head tilt
        if mean_pitch > self.pitch_danger_threshold:
            danger_conditions.append(f"Head tilted ({mean_pitch:.1f}°)")
        
        # State transitions with hysteresis
        if danger_conditions:
            # Has danger conditions
            if self.state != 'DANGER':
                self.danger_start_time = time.time()
            self.state = 'DANGER'
            reason = " | ".join(danger_conditions)
            
        else:
            # No danger conditions - check if we can safely exit DANGER state
            if self.state == 'DANGER':
                # Need ALL metrics to improve significantly to exit
                ear_improved = mean_ear >= (self.ear_danger_threshold + self.ear_hysteresis)
                perclos_improved = perclos <= (self.perclos_danger_threshold - self.perclos_hysteresis)
                pitch_improved = mean_pitch <= (self.pitch_danger_threshold - self.pitch_hysteresis)
                
                if ear_improved and perclos_improved and pitch_improved:
                    # All metrics have improved - exit to OK
                    self.state = 'OK'
                    self.danger_start_time = None
                    self.last_ear_values = []
                    self.last_pitch_values = []
                    reason = "All metrics improved - returning to normal"
                else:
                    # Still some issues - stay in DANGER
                    issues = []
                    if not ear_improved:
                        issues.append(f"EAR still low ({mean_ear:.3f})")
                    if not perclos_improved:
                        issues.append(f"PERCLOS still high ({perclos:.1f}%)")
                    if not pitch_improved:
                        issues.append(f"Head still tilted ({mean_pitch:.1f}°)")
                    reason = "Still in danger - " + " | ".join(issues)
                    self.state = 'DANGER'
            else:
                # In OK state with no danger conditions
                self.state = 'OK'
                reason = f"Normal (EAR: {mean_ear:.3f}, PERCLOS: {perclos:.1f}%, Pitch: {mean_pitch:.1f}°)"
        
        return self.state, reason
    
    def reset(self):
        """Reset decision engine state"""
        self.state = 'OK'
        self.danger_start_time = None
        self.last_ear_values = []
        self.last_pitch_values = []
    
    def get_danger_duration(self):
        """Get how long we've been in danger state (seconds)"""
        if self.danger_start_time is None:
            return 0.0
        return time.time() - self.danger_start_time
