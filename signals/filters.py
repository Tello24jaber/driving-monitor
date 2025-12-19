"""
Signal smoothing filters using convolution
"""
import numpy as np

class MovingAverageFilter:
    """
    Moving average filter using discrete convolution
    y[n] = (1/M) * Σ x[n-k] for k = 0 to M-1
    """
    def __init__(self, window_size=7):
        """
        Args:
            window_size: Number of samples to average (default 7)
        """
        self.window_size = window_size
        # Create rectangular kernel normalized to 1
        self.kernel = np.ones(window_size) / window_size
        
    def apply(self, signal):
        """
        Apply moving average filter via convolution
        
        Args:
            signal: 1D numpy array
            
        Returns:
            Smoothed signal (same length as input)
        """
        if len(signal) < self.window_size:
            return signal
        
        # Use 'same' mode to keep output same length as input
        smoothed = np.convolve(signal, self.kernel, mode='same')
        
        return smoothed
