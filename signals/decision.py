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
        # Thresholds
        self.ear_danger_threshold = 0.18
        self.perclos_danger_threshold = 70.0  # 70%
        self.pitch_danger_threshold = 20.0  # degrees (head down)
        
        # Sustained time requirements (frames)
        self.ear_sustained_frames = 20  # ~0.67 sec at 30fps
        self.pitch_sustained_frames = 30  # ~1 sec at 30fps
        
        # Hysteresis margins
        self.ear_hysteresis = 0.03
        self.perclos_hysteresis = 10.0
        self.pitch_hysteresis = 5.0
        
        # State tracking
        self.state = 'OK'  # OK, WARNING, DANGER
        self.danger_start_time = None
        self.last_ear_values = []
        self.last_pitch_values = []
        
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
        
        # Keep only recent history
        if len(self.last_ear_values) > 60:
            self.last_ear_values.pop(0)
        if len(self.last_pitch_values) > 60:
            self.last_pitch_values.pop(0)
        
        # Check danger conditions
        danger_reasons = []
        
        # Condition 1: Sustained low EAR
        if len(self.last_ear_values) >= self.ear_sustained_frames:
            recent_ear = self.last_ear_values[-self.ear_sustained_frames:]
            if np.mean(recent_ear) < self.ear_danger_threshold:
                danger_reasons.append(f"Sustained low EAR ({np.mean(recent_ear):.3f})")
        
        # Condition 2: High PERCLOS
        if perclos > self.perclos_danger_threshold:
            danger_reasons.append(f"High PERCLOS ({perclos:.1f}%)")
        
        # Condition 3: Sustained head nod (pitch down is positive)
        if len(self.last_pitch_values) >= self.pitch_sustained_frames:
            recent_pitch = self.last_pitch_values[-self.pitch_sustained_frames:]
            if np.mean(recent_pitch) > self.pitch_danger_threshold:
                danger_reasons.append(f"Head nodding ({np.mean(recent_pitch):.1f}°)")
        
        # State machine with hysteresis
        previous_state = self.state
        
        if danger_reasons:
            # Enter DANGER state
            if self.state != 'DANGER':
                self.danger_start_time = time.time()
            self.state = 'DANGER'
            reason = " | ".join(danger_reasons)
            
        elif self.state == 'DANGER':
            # Check if ALL conditions improved enough to exit DANGER (hysteresis)
            # We need ALL metrics to be in safe zone to exit
            ear_safe = False
            perclos_safe = False
            pitch_safe = False
            
            # Check EAR improvement
            if len(self.last_ear_values) >= self.ear_sustained_frames:
                recent_ear = self.last_ear_values[-self.ear_sustained_frames:]
                # Exit threshold is higher than entry (hysteresis)
                if np.mean(recent_ear) >= (self.ear_danger_threshold + self.ear_hysteresis):
                    ear_safe = True
            
            # Check PERCLOS improvement
            # Exit threshold is lower than entry (hysteresis)
            if perclos <= (self.perclos_danger_threshold - self.perclos_hysteresis):
                perclos_safe = True
            
            # Check pitch improvement
            if len(self.last_pitch_values) >= self.pitch_sustained_frames:
                recent_pitch = self.last_pitch_values[-self.pitch_sustained_frames:]
                # Exit threshold is lower than entry (hysteresis)
                if np.mean(recent_pitch) <= (self.pitch_danger_threshold - self.pitch_hysteresis):
                    pitch_safe = True
            
            # Only exit DANGER if ALL conditions are safe
            if ear_safe and perclos_safe and pitch_safe:
                self.state = 'OK'
                self.danger_start_time = None
                self.last_ear_values = []  # Clear history for fresh start
                self.last_pitch_values = []
                reason = "All conditions improved"
            else:
                reasons = []
                if not ear_safe:
                    reasons.append("EAR still low")
                if not perclos_safe:
                    reasons.append("PERCLOS still high")
                if not pitch_safe:
                    reasons.append("Pitch still high")
                reason = "Still in danger: " + ", ".join(reasons)
                
        else:
            # Normal state
            self.state = 'OK'
            reason = "All conditions normal"
        
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
