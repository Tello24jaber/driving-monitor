# -----------------------------------------------------------------------------
# Driver Monitoring System - GUI Interface
# -----------------------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
from utils import *
from detect.face import *
from detect.pose import *
from state import *
import mediapipe_compat
import mediapipe as mp

class DriverMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Monitoring System")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2c3e50')
        
        # Monitoring state
        self.is_running = False
        self.cap = None
        self.thread = None
        
        # Statistics
        self.drowsiness_count = 0
        self.total_frames = 0
        self.start_time = None
        
        # Create GUI components
        self.create_widgets()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#34495e', height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="🚗 Driver Drowsiness Monitoring System", 
            font=("Arial", 20, "bold"),
            bg='#34495e',
            fg='white'
        )
        title_label.pack(pady=10)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Video feed
        left_panel = tk.Frame(content_frame, bg='#34495e', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        video_label = tk.Label(left_panel, text="Camera Feed", font=("Arial", 14, "bold"), 
                              bg='#34495e', fg='white')
        video_label.pack(pady=5)
        
        self.video_canvas = tk.Label(left_panel, bg='black')
        self.video_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right panel - Controls and status
        right_panel = tk.Frame(content_frame, bg='#34495e', width=300, relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Controls section
        controls_frame = tk.LabelFrame(right_panel, text="Controls", font=("Arial", 12, "bold"),
                                      bg='#34495e', fg='white', padx=10, pady=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = tk.Button(
            controls_frame,
            text="▶ Start Monitoring",
            command=self.toggle_monitoring,
            font=("Arial", 12, "bold"),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=3,
            cursor='hand2',
            height=2
        )
        self.start_button.pack(fill=tk.X, pady=5)
        
        # Status section
        status_frame = tk.LabelFrame(right_panel, text="System Status", font=("Arial", 12, "bold"),
                                    bg='#34495e', fg='white', padx=10, pady=10)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="● Inactive",
            font=("Arial", 11, "bold"),
            bg='#34495e',
            fg='#95a5a6'
        )
        self.status_label.pack(anchor=tk.W, pady=2)
        
        self.camera_label = tk.Label(
            status_frame,
            text="📷 Camera: Not Connected",
            font=("Arial", 10),
            bg='#34495e',
            fg='white'
        )
        self.camera_label.pack(anchor=tk.W, pady=2)
        
        self.time_label = tk.Label(
            status_frame,
            text="⏱️ Running Time: 00:00:00",
            font=("Arial", 10),
            bg='#34495e',
            fg='white'
        )
        self.time_label.pack(anchor=tk.W, pady=2)
        
        # Statistics section
        stats_frame = tk.LabelFrame(right_panel, text="Statistics", font=("Arial", 12, "bold"),
                                   bg='#34495e', fg='white', padx=10, pady=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.frames_label = tk.Label(
            stats_frame,
            text="Frames Processed: 0",
            font=("Arial", 10),
            bg='#34495e',
            fg='white'
        )
        self.frames_label.pack(anchor=tk.W, pady=2)
        
        self.drowsy_label = tk.Label(
            stats_frame,
            text="Drowsiness Alerts: 0",
            font=("Arial", 10),
            bg='#34495e',
            fg='white'
        )
        self.drowsy_label.pack(anchor=tk.W, pady=2)
        
        # Alert section
        alert_frame = tk.LabelFrame(right_panel, text="Alerts", font=("Arial", 12, "bold"),
                                   bg='#34495e', fg='white', padx=10, pady=10)
        alert_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.alert_text = tk.Text(
            alert_frame,
            height=8,
            font=("Arial", 9),
            bg='#2c3e50',
            fg='white',
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.alert_text.pack(fill=tk.BOTH, expand=True)
        self.alert_text.insert(tk.END, "System ready. Click 'Start Monitoring' to begin.\n")
        self.alert_text.config(state=tk.DISABLED)
        
        # Bottom status bar
        status_bar = tk.Frame(self.root, bg='#34495e', height=30)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.bottom_status = tk.Label(
            status_bar,
            text="Ready",
            font=("Arial", 9),
            bg='#34495e',
            fg='white',
            anchor=tk.W
        )
        self.bottom_status.pack(side=tk.LEFT, padx=10)
    
    def toggle_monitoring(self):
        if not self.is_running:
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def start_monitoring(self):
        try:
            # Try to open camera
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera. Please check your camera connection.")
                return
            
            self.is_running = True
            self.start_time = time.time()
            self.drowsiness_count = 0
            self.total_frames = 0
            
            # Update UI
            self.start_button.config(
                text="⏸ Stop Monitoring",
                bg='#e74c3c',
                activebackground='#c0392b'
            )
            self.status_label.config(text="● Active", fg='#27ae60')
            self.camera_label.config(text="📷 Camera: Connected")
            self.bottom_status.config(text="Monitoring active...")
            
            self.add_alert("✓ Monitoring started successfully")
            
            # Start monitoring thread
            self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.thread.start()
            
            # Start UI update loop
            self.update_ui()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
            self.is_running = False
    
    def stop_monitoring(self):
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Update UI
        self.start_button.config(
            text="▶ Start Monitoring",
            bg='#27ae60',
            activebackground='#229954'
        )
        self.status_label.config(text="● Inactive", fg='#95a5a6')
        self.camera_label.config(text="📷 Camera: Not Connected")
        self.bottom_status.config(text="Ready")
        
        # Clear video canvas
        self.video_canvas.config(image='')
        
        self.add_alert("⏹ Monitoring stopped")
    
    def monitor_loop(self):
        try:
            # Initialize face mesh with compatibility layer
            faceMesh = mp.solutions.face_mesh.FaceMesh(
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Thresholds
            marThresh = 0.7
            marThresh2 = 0.15
            headThresh = 6
            earThresh = 0.28
            blinkThresh = 10
            gazeThresh = 5
            
            captureFps = self.cap.get(cv2.CAP_PROP_FPS)
            if captureFps == 0:
                captureFps = 30  # Default fallback
            
            driverState = DriverState(marThresh, marThresh2, headThresh, earThresh, blinkThresh, gazeThresh)
            headPose = HeadPose(faceMesh)
            faceDetector = FaceDetector(faceMesh, captureFps, marThresh, marThresh2, headThresh, earThresh, blinkThresh)
            
            startTime = time.time()
            drowsinessCounter = 0
            
            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                self.total_frames += 1
                
                # Process frame
                frame, results = headPose.process_image(frame)
                frame = headPose.estimate_pose(frame, results, True)
                roll, pitch, yaw = headPose.calculate_angles()
                
                frame, sleepEyes, mar, gaze, yawning, baseR, baseP, baseY, baseG = faceDetector.evaluate_face(
                    frame, results, roll, pitch, yaw, True
                )
                
                frame, state = driverState.eval_state(
                    frame, sleepEyes, mar, roll, pitch, yaw, gaze, yawning, baseR, baseP, baseG
                )
                
                # Update drowsiness counter
                if state == "Drowsy":
                    drowsinessCounter += 1
                
                drowsinessTime = drowsinessCounter / captureFps
                drowsy = drowsinessTime / 60
                
                # Reset counter every minute
                if time.time() - startTime >= 60:
                    drowsinessCounter = 0
                    startTime = time.time()
                
                # Alert if drowsy
                if drowsy > 0.08:
                    self.drowsiness_count += 1
                    self.add_alert("⚠️ WARNING: Drowsiness detected!")
                
                # Store frame for display
                self.current_frame = frame
                
                # Small delay to prevent overwhelming the CPU
                time.sleep(0.01)
                
        except Exception as e:
            self.add_alert(f"❌ Error: {str(e)}")
            self.is_running = False
    
    def update_ui(self):
        if self.is_running:
            # Update video feed
            if hasattr(self, 'current_frame'):
                frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_canvas.imgtk = imgtk
                self.video_canvas.configure(image=imgtk)
            
            # Update statistics
            self.frames_label.config(text=f"Frames Processed: {self.total_frames}")
            self.drowsy_label.config(text=f"Drowsiness Alerts: {self.drowsiness_count}")
            
            # Update running time
            if self.start_time:
                elapsed = int(time.time() - self.start_time)
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                seconds = elapsed % 60
                self.time_label.config(text=f"⏱️ Running Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Schedule next update
            self.root.after(30, self.update_ui)
    
    def add_alert(self, message):
        def _add():
            self.alert_text.config(state=tk.NORMAL)
            timestamp = time.strftime("%H:%M:%S")
            self.alert_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.alert_text.see(tk.END)
            self.alert_text.config(state=tk.DISABLED)
        
        # Thread-safe UI update
        self.root.after(0, _add)
    
    def on_closing(self):
        if self.is_running:
            if messagebox.askokcancel("Quit", "Monitoring is active. Do you want to quit?"):
                self.stop_monitoring()
                time.sleep(0.5)  # Give time for thread to stop
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    root = tk.Tk()
    app = DriverMonitorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
