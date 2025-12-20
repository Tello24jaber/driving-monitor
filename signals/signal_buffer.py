"""
Circular buffer for storing discrete-time signals
"""
import numpy as np
from collections import deque
import time

class SignalBuffer:
    """
    Stores time-series data for signal processing
    """
    def __init__(self, max_size=300):
        """
        Args:
            max_size: Buffer size (default 300 samples ≈ 10 sec at 30 FPS)
        """
        self.max_size = max_size
        self.ear_left = deque(maxlen=max_size)
        self.ear_right = deque(maxlen=max_size)
        self.ear_avg = deque(maxlen=max_size)
        self.head_pitch = deque(maxlen=max_size)
        self.head_roll = deque(maxlen=max_size)
        self.head_yaw = deque(maxlen=max_size)
        self.mar = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        
    def add_sample(self, ear_left, ear_right, pitch, roll, yaw, mar=0.0):
        """Add new sample to all buffers"""
        ear_avg = (ear_left + ear_right) / 2.0
        
        self.ear_left.append(ear_left)
        self.ear_right.append(ear_right)
        self.ear_avg.append(ear_avg)
        self.head_pitch.append(pitch)
        self.head_roll.append(roll)
        self.head_yaw.append(yaw)
        self.mar.append(mar)
        self.timestamps.append(time.time())
        
    def get_array(self, signal_name):
        """Get signal as numpy array"""
        buffer_map = {
            'ear_left': self.ear_left,
            'ear_right': self.ear_right,
            'ear_avg': self.ear_avg,
            'pitch': self.head_pitch,
            'roll': self.head_roll,
            'yaw': self.head_yaw,
            'mar': self.mar
        }
        return np.array(buffer_map.get(signal_name, []))
    
    def get_last_n(self, signal_name, n):
        """Get last n samples of a signal"""
        arr = self.get_array(signal_name)
        if len(arr) < n:
            return arr
        return arr[-n:]
    
    def is_ready(self, min_samples=30):
        """Check if buffer has enough samples for processing"""
        return len(self.ear_avg) >= min_samples
    
    def __len__(self):
        return len(self.ear_avg)
    
    def clear(self):
        """Clear all buffers"""
        self.ear_left.clear()
        self.ear_right.clear()
        self.ear_avg.clear()
        self.head_pitch.clear()
        self.head_roll.clear()
        self.head_yaw.clear()
        self.mar.clear()
        self.timestamps.clear()
