"""
Audio alert system for drowsiness warnings
"""
import threading
import numpy as np

class AudioAlert:
    """
    Plays continuous warning beep when danger is detected
    Uses winsound for Windows compatibility
    """
    def __init__(self, frequency=800, duration=300):
        """
        Args:
            frequency: Beep frequency in Hz (default 800 Hz)
            duration: Beep duration in ms (default 300 ms)
        """
        self.frequency = frequency
        self.duration = duration
        self.is_playing = False
        self.audio_thread = None
        self.stop_flag = False
        
        # Try to import winsound (Windows)
        try:
            import winsound
            self.winsound = winsound
            self.use_winsound = True
        except ImportError:
            self.winsound = None
            self.use_winsound = False
            print("Warning: winsound not available. Audio alerts may not work.")
    
    def _play_loop(self):
        """Background thread to play continuous beep"""
        if not self.use_winsound:
            return
            
        try:
            while not self.stop_flag:
                # Play beep
                self.winsound.Beep(self.frequency, self.duration)
                # Pause between beeps
                import time
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Audio error: {e}")
    
    def start(self):
        """Start playing alert sound"""
        if self.is_playing:
            return
        
        if not self.use_winsound:
            # Fallback: print to console
            print("\a")  # System beep
            return
        
        self.stop_flag = False
        self.is_playing = True
        self.audio_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.audio_thread.start()
    
    def stop(self):
        """Stop playing alert sound"""
        if not self.is_playing:
            return
        
        self.stop_flag = True
        self.is_playing = False
        
        if self.audio_thread:
            self.audio_thread.join(timeout=0.5)
    
    def __del__(self):
        """Cleanup"""
        self.stop()
