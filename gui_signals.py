# -----------------------------------------------------------------------------
# Driver Monitoring System - Signals & Systems GUI Interface
# -----------------------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import numpy as np
from utils import *
from detect.face import *
from detect.pose import *
from state import *
import mediapipe_compat
import mediapipe as mp

# Import signal processing modules
from signals import (
    SignalBuffer,
    MovingAverageFilter,
    PERCLOSCalculator,
    DrowsinessDecisionEngine,
    AudioAlert
)

# Import matplotlib for plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DriverMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Monitoring System - Signals & Systems")
        self.root.geometry("1600x900")
        self.root.configure(bg='#1e1e1e')
        
        # Monitoring state
        self.is_running = False
        self.cap = None
        self.thread = None
        
        # Statistics
        self.drowsiness_count = 0
        self.total_frames = 0
        self.start_time = None
        
        # Signal processing components
        self.signal_buffer = SignalBuffer(max_size=300)
        self.ear_filter = MovingAverageFilter(window_size=7)
        self.pitch_filter = MovingAverageFilter(window_size=15)
        self.perclos_calc = PERCLOSCalculator(window_size=90, threshold=0.20)  # Higher threshold to match decision engine
        self.decision_engine = DrowsinessDecisionEngine()
        self.audio_alert = AudioAlert(frequency=800, duration=300)
        
        # Current signal values
        self.current_smoothed_ear = 0.0
        self.current_perclos = 0.0
        self.current_smoothed_pitch = 0.0
        self.current_state = 'OK'
        self.current_reason = ''
        
        # Create GUI components
        self.create_widgets()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # Main container
        main_container = tk.Frame(self.root, bg='#1e1e1e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side: Video and controls
        left_panel = tk.Frame(main_container, bg='#1e1e1e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Title
        title_frame = tk.Frame(left_panel, bg='#2d2d2d', height=50)
        title_frame.pack(fill=tk.X, pady=(0, 5))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🚗 DRIVER DROWSINESS MONITORING - SIGNALS & SYSTEMS", 
            font=("Arial", 16, "bold"),
            bg='#2d2d2d',
            fg='white'
        )
        title_label.pack(pady=10)
        
        # Video feed
        video_frame = tk.Frame(left_panel, bg='black', relief=tk.RAISED, borderwidth=2)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.video_canvas = tk.Label(video_frame, bg='black')
        self.video_canvas.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Alert banner (hidden by default)
        self.alert_banner = tk.Frame(left_panel, bg='#27ae60', height=80, relief=tk.RAISED, borderwidth=3)
        self.alert_banner.pack(fill=tk.X, pady=(0, 5))
        self.alert_banner.pack_propagate(False)
        
        self.alert_state_label = tk.Label(
            self.alert_banner,
            text="● SYSTEM OK",
            font=("Arial", 24, "bold"),
            bg='#27ae60',
            fg='white'
        )
        self.alert_state_label.pack(pady=5)
        
        self.alert_reason_label = tk.Label(
            self.alert_banner,
            text="All conditions normal",
            font=("Arial", 12),
            bg='#27ae60',
            fg='white'
        )
        self.alert_reason_label.pack()
        
        # Control panel
        control_frame = tk.Frame(left_panel, bg='#2d2d2d', relief=tk.RAISED, borderwidth=2)
        control_frame.pack(fill=tk.X)
        
        self.start_button = tk.Button(
            control_frame,
            text="▶ START MONITORING",
            command=self.toggle_monitoring,
            font=("Arial", 14, "bold"),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=3,
            cursor='hand2',
            height=2
        )
        self.start_button.pack(fill=tk.X, padx=10, pady=10)
        
        # Right side: Plots and metrics
        right_panel = tk.Frame(main_container, bg='#1e1e1e', width=700)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Metrics panel
        metrics_frame = tk.LabelFrame(
            right_panel,
            text="REAL-TIME METRICS",
            font=("Arial", 12, "bold"),
            bg='#2d2d2d',
            fg='white',
            padx=10,
            pady=10,
            relief=tk.RAISED,
            borderwidth=2
        )
        metrics_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Metrics grid
        metrics_grid = tk.Frame(metrics_frame, bg='#2d2d2d')
        metrics_grid.pack(fill=tk.X)
        
        # Row 1: EAR
        tk.Label(metrics_grid, text="Smoothed EAR:", font=("Arial", 11, "bold"), 
                bg='#2d2d2d', fg='#3498db', anchor='w').grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.ear_value_label = tk.Label(metrics_grid, text="0.000", font=("Arial", 11), 
                                        bg='#2d2d2d', fg='white', anchor='e')
        self.ear_value_label.grid(row=0, column=1, sticky='e', padx=5, pady=3)
        
        # Row 2: PERCLOS
        tk.Label(metrics_grid, text="PERCLOS:", font=("Arial", 11, "bold"), 
                bg='#2d2d2d', fg='#e74c3c', anchor='w').grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.perclos_value_label = tk.Label(metrics_grid, text="0.0%", font=("Arial", 11), 
                                           bg='#2d2d2d', fg='white', anchor='e')
        self.perclos_value_label.grid(row=1, column=1, sticky='e', padx=5, pady=3)
        
        # Row 3: Head Pitch
        tk.Label(metrics_grid, text="Head Pitch:", font=("Arial", 11, "bold"), 
                bg='#2d2d2d', fg='#f39c12', anchor='w').grid(row=2, column=0, sticky='w', padx=5, pady=3)
        self.pitch_value_label = tk.Label(metrics_grid, text="0.0°", font=("Arial", 11), 
                                          bg='#2d2d2d', fg='white', anchor='e')
        self.pitch_value_label.grid(row=2, column=1, sticky='e', padx=5, pady=3)
        
        # Row 4: Frames
        tk.Label(metrics_grid, text="Frames:", font=("Arial", 11, "bold"), 
                bg='#2d2d2d', fg='#9b59b6', anchor='w').grid(row=3, column=0, sticky='w', padx=5, pady=3)
        self.frames_value_label = tk.Label(metrics_grid, text="0", font=("Arial", 11), 
                                           bg='#2d2d2d', fg='white', anchor='e')
        self.frames_value_label.grid(row=3, column=1, sticky='e', padx=5, pady=3)
        
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
        
        # Plots panel
        plots_frame = tk.LabelFrame(
            right_panel,
            text="SIGNAL ANALYSIS",
            font=("Arial", 12, "bold"),
            bg='#2d2d2d',
            fg='white',
            padx=5,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2
        )
        plots_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.create_plots(plots_frame)
    
    def create_plots(self, parent):
        """Create matplotlib plots for signal visualization"""
        self.fig = Figure(figsize=(7, 6), facecolor='#1e1e1e')
        
        # EAR plot (top)
        self.ax_ear = self.fig.add_subplot(211, facecolor='#2d2d2d')
        self.ax_ear.set_title('Eye Aspect Ratio (EAR) Signal', color='white', fontsize=11, fontweight='bold')
        self.ax_ear.set_ylabel('EAR', color='white', fontsize=9)
        self.ax_ear.tick_params(colors='white', labelsize=8)
        self.ax_ear.grid(True, alpha=0.2, color='gray')
        self.ax_ear.set_ylim([0, 0.5])
        
        # Head pitch plot (bottom)
        self.ax_pitch = self.fig.add_subplot(212, facecolor='#2d2d2d')
        self.ax_pitch.set_title('Head Pitch Signal', color='white', fontsize=11, fontweight='bold')
        self.ax_pitch.set_xlabel('Sample Number', color='white', fontsize=9)
        self.ax_pitch.set_ylabel('Pitch (degrees)', color='white', fontsize=9)
        self.ax_pitch.tick_params(colors='white', labelsize=8)
        self.ax_pitch.grid(True, alpha=0.2, color='gray')
        self.ax_pitch.set_ylim([-30, 30])
        
        self.fig.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize empty lines
        self.line_raw_ear, = self.ax_ear.plot([], [], 'b-', alpha=0.3, linewidth=1, label='Raw EAR')
        self.line_smooth_ear, = self.ax_ear.plot([], [], 'cyan', linewidth=2, label='Smoothed EAR')
        self.line_threshold_ear = self.ax_ear.axhline(y=0.18, color='red', linestyle='--', 
                                                       linewidth=1, alpha=0.7, label='Danger Threshold')
        self.ax_ear.legend(loc='upper right', fontsize=8, facecolor='#2d2d2d', 
                          edgecolor='white', labelcolor='white')
        
        self.line_raw_pitch, = self.ax_pitch.plot([], [], 'orange', alpha=0.3, linewidth=1, label='Raw Pitch')
        self.line_smooth_pitch, = self.ax_pitch.plot([], [], 'yellow', linewidth=2, label='Smoothed Pitch')
        self.line_threshold_pitch = self.ax_pitch.axhline(y=30.0, color='red', linestyle='--', 
                                                           linewidth=1, alpha=0.7, label='Danger Threshold')
        self.ax_pitch.axhline(y=-30.0, color='red', linestyle='--', linewidth=1, alpha=0.7)  # Negative threshold
        self.ax_pitch.legend(loc='upper right', fontsize=8, facecolor='#2d2d2d', 
                            edgecolor='white', labelcolor='white')
    
    def update_plots(self):
        """Update the signal plots"""
        if not self.signal_buffer.is_ready(30):
            return
        
        # Get signals
        raw_ear = self.signal_buffer.get_array('ear_avg')
        raw_pitch = self.signal_buffer.get_array('pitch')
        
        # Apply smoothing
        smoothed_ear = self.ear_filter.apply(raw_ear)
        smoothed_pitch = self.pitch_filter.apply(raw_pitch)
        
        # Limit display to last 150 samples
        display_samples = 150
        if len(raw_ear) > display_samples:
            raw_ear = raw_ear[-display_samples:]
            smoothed_ear = smoothed_ear[-display_samples:]
        if len(raw_pitch) > display_samples:
            raw_pitch = raw_pitch[-display_samples:]
            smoothed_pitch = smoothed_pitch[-display_samples:]
        
        x_ear = np.arange(len(raw_ear))
        x_pitch = np.arange(len(raw_pitch))
        
        # Update EAR plot
        self.line_raw_ear.set_data(x_ear, raw_ear)
        self.line_smooth_ear.set_data(x_ear, smoothed_ear)
        self.ax_ear.set_xlim([0, max(150, len(raw_ear))])
        
        # Update pitch plot
        self.line_raw_pitch.set_data(x_pitch, raw_pitch)
        self.line_smooth_pitch.set_data(x_pitch, smoothed_pitch)
        self.ax_pitch.set_xlim([0, max(150, len(raw_pitch))])
        
        # Redraw
        self.canvas.draw_idle()
    
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
            
            # Reset signal processing components
            self.signal_buffer.clear()
            self.decision_engine.reset()
            
            # Update UI
            self.start_button.config(
                text="⏸ STOP MONITORING",
                bg='#e74c3c',
                activebackground='#c0392b'
            )
            
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
        
        # Stop audio alert
        self.audio_alert.stop()
        
        # Update UI
        self.start_button.config(
            text="▶ START MONITORING",
            bg='#27ae60',
            activebackground='#229954'
        )
        
        # Clear video canvas
        self.video_canvas.config(image='')
        
        # Reset alert banner
        self.alert_banner.config(bg='#27ae60')
        self.alert_state_label.config(text="● SYSTEM OK", bg='#27ae60')
        self.alert_reason_label.config(text="Monitoring stopped", bg='#27ae60')
    
    def monitor_loop(self):
        try:
            # Initialize face mesh
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
                captureFps = 30
            
            driverState = DriverState(marThresh, marThresh2, headThresh, earThresh, blinkThresh, gazeThresh)
            headPose = HeadPose(faceMesh)
            faceDetector = FaceDetector(faceMesh, captureFps, marThresh, marThresh2, headThresh, earThresh, blinkThresh)
            
            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                self.total_frames += 1
                
                # Process frame
                frame, results = headPose.process_image(frame)
                frame = headPose.estimate_pose(frame, results, True)  # Display landmarks
                roll, pitch, yaw = headPose.calculate_angles()
                
                frame, sleepEyes, mar, gaze, yawning, baseR, baseP, baseY, baseG = faceDetector.evaluate_face(
                    frame, results, roll, pitch, yaw, True  # Display landmarks and metrics
                )
                
                # Get EAR values directly from faceDetector
                ear_left = faceDetector.ear if hasattr(faceDetector, 'ear') else 0.3
                ear_right = faceDetector.ear if hasattr(faceDetector, 'ear') else 0.3
                
                # Add to signal buffer
                self.signal_buffer.add_sample(ear_left, ear_right, pitch, roll, yaw)
                
                # Process signals (only if enough samples)
                if self.signal_buffer.is_ready(30):
                    # Get raw signals
                    raw_ear = self.signal_buffer.get_array('ear_avg')
                    raw_pitch = self.signal_buffer.get_array('pitch')
                    
                    # Apply smoothing via convolution
                    smoothed_ear = self.ear_filter.apply(raw_ear)
                    smoothed_pitch = self.pitch_filter.apply(raw_pitch)
                    
                    # Get current smoothed values
                    self.current_smoothed_ear = smoothed_ear[-1]
                    self.current_smoothed_pitch = smoothed_pitch[-1]
                    
                    # Compute PERCLOS
                    self.current_perclos = self.perclos_calc.compute(smoothed_ear)
                    
                    # Make drowsiness decision
                    self.current_state, self.current_reason = self.decision_engine.evaluate(
                        self.current_smoothed_ear,
                        self.current_perclos,
                        self.current_smoothed_pitch
                    )
                    
                    # Handle audio alert and visual feedback
                    if self.current_state == 'DANGER':
                        self.audio_alert.start()
                        self.drowsiness_count += 1
                        
                        # Draw RED border on frame
                        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 15)
                        
                        # Add warning text with background
                        warning_text = "DROWSINESS DETECTED!"
                        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                        text_x = (frame.shape[1] - text_size[0]) // 2
                        text_y = 80
                        
                        # Background rectangle for text
                        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                                    (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
                        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                                    (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), 3)
                        
                        cv2.putText(frame, warning_text, (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    else:
                        self.audio_alert.stop()
                        
                        # Draw GREEN border on frame
                        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 8)
                        
                        # Green status indicator
                        cv2.circle(frame, (30, 30), 12, (0, 255, 0), -1)
                        cv2.putText(frame, "OK", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Display signal-based metrics on frame (bottom left, with background)
                    metrics_y_start = frame.shape[0] - 130
                    cv2.rectangle(frame, (5, metrics_y_start - 5), (280, frame.shape[0] - 5), (0, 0, 0), -1)
                    cv2.rectangle(frame, (5, metrics_y_start - 5), (280, frame.shape[0] - 5), (100, 100, 100), 2)
                    
                    cv2.putText(frame, f"Smoothed EAR: {self.current_smoothed_ear:.3f}", 
                              (10, metrics_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, f"PERCLOS: {self.current_perclos:.1f}%", 
                              (10, metrics_y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                    cv2.putText(frame, f"Head Pitch: {self.current_smoothed_pitch:.1f} deg", 
                              (10, metrics_y_start + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                    
                    # State indicator
                    state_color = (0, 255, 0) if self.current_state == 'OK' else (0, 0, 255)
                    cv2.putText(frame, f"State: {self.current_state}", 
                              (10, metrics_y_start + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
                
                # Store frame for display
                self.current_frame = frame
                
                # Small delay
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
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
            
            # Update metrics labels
            self.ear_value_label.config(text=f"{self.current_smoothed_ear:.3f}")
            self.perclos_value_label.config(text=f"{self.current_perclos:.1f}%")
            self.pitch_value_label.config(text=f"{self.current_smoothed_pitch:.1f}°")
            self.frames_value_label.config(text=str(self.total_frames))
            
            # Update alert banner based on state
            if self.current_state == 'DANGER':
                self.alert_banner.config(bg='#e74c3c')
                self.alert_state_label.config(text="⚠ DANGER - DROWSINESS DETECTED!", bg='#e74c3c')
                self.alert_reason_label.config(text=self.current_reason, bg='#e74c3c')
            elif self.current_state == 'WARNING':
                self.alert_banner.config(bg='#f39c12')
                self.alert_state_label.config(text="⚠ WARNING", bg='#f39c12')
                self.alert_reason_label.config(text=self.current_reason, bg='#f39c12')
            else:
                self.alert_banner.config(bg='#27ae60')
                self.alert_state_label.config(text="● SYSTEM OK", bg='#27ae60')
                self.alert_reason_label.config(text="All conditions normal", bg='#27ae60')
            
            # Update plots
            if self.signal_buffer.is_ready(30):
                self.update_plots()
            
            # Schedule next update
            self.root.after(50, self.update_ui)
    
    def on_closing(self):
        if self.is_running:
            if messagebox.askokcancel("Quit", "Monitoring is active. Do you want to quit?"):
                self.stop_monitoring()
                time.sleep(0.5)
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    root = tk.Tk()
    app = DriverMonitorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
