"""
PERCLOS (PERcentage of eye CLOsure) computation
"""
import numpy as np

class PERCLOSCalculator:
    """
    Computes PERCLOS using sliding window on EAR signal
    PERCLOS = percentage of time eyes are closed in a time window
    """
    def __init__(self, window_size=90, threshold=0.2):
        """
        Args:
            window_size: Number of samples for sliding window (default 90 ≈ 3 sec)
            threshold: EAR threshold for "closed" eyes (default 0.2)
        """
        self.window_size = window_size
        self.threshold = threshold
        
    def compute(self, ear_signal):
        """
        Compute PERCLOS from EAR signal
        
        Args:
            ear_signal: 1D numpy array of EAR values (smoothed recommended)
            
        Returns:
            PERCLOS value (0-100%)
        """
        if len(ear_signal) == 0:
            return 0.0
            
        if len(ear_signal) < self.window_size:
            # Not enough samples yet
            window = ear_signal
        else:
            # Use last window_size samples
            window = ear_signal[-self.window_size:]
        
        # Count samples where eyes are closed
        closed_samples = np.sum(window < self.threshold)
        
        # Compute percentage
        perclos = (closed_samples / len(window)) * 100.0
        
        return perclos
