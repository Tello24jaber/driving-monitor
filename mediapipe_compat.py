# -----------------------------------------------------------------------------
# MediaPipe Compatibility Layer for Legacy Solutions API
# -----------------------------------------------------------------------------
# This module provides compatibility between old and new mediapipe APIs
# -----------------------------------------------------------------------------

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

# Create a mock solutions module structure to make old API work with new API
if not hasattr(mp, 'solutions'):
    class MockResults:
        """Mock results object compatible with old API"""
        def __init__(self, face_landmarks_list=None):
            self.multi_face_landmarks = face_landmarks_list
    
    class MockLandmarkList:
        """Mock landmark list compatible with old API"""
        def __init__(self, landmarks):
            self.landmark = landmarks
    
    class MockLandmark:
        """Mock landmark compatible with old API"""
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    class MockFaceMesh:
        """Mock FaceMesh that uses new mediapipe API"""
        def __init__(self, static_image_mode=False, max_num_faces=1, 
                   refine_landmarks=False, min_detection_confidence=0.5, 
                   min_tracking_confidence=0.5):
            """Initialize Face Mesh with new API"""
            self.static_image_mode = static_image_mode
            self.max_num_faces = max_num_faces
            self.refine_landmarks = refine_landmarks
            self.min_detection_confidence = min_detection_confidence
            self.min_tracking_confidence = min_tracking_confidence
            
            # Try to use FaceLandmarker if model file available
            model_path = 'face_landmarker.task'
            import os
            
            if not os.path.exists(model_path):
                print(f"Warning: Face landmarker model not found at '{model_path}'")
                print("Downloading model file...")
                try:
                    import urllib.request
                    model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
                    urllib.request.urlretrieve(model_url, model_path)
                    print("Model downloaded successfully!")
                except Exception as e:
                    print(f"Failed to download model: {e}")
                    print("Face detection will not work properly.")
                    self.landmarker = None
                    self.use_landmarker = False
                    self.frame_count = 0
                    return
            
            # Create FaceLandmarker with new API
            try:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO if not static_image_mode else vision.RunningMode.IMAGE,
                    num_faces=max_num_faces,
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_tracking_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False
                )
                
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.use_landmarker = True
                print("FaceLandmarker initialized successfully!")
            except Exception as e:
                print(f"Warning: Could not create FaceLandmarker: {e}")
                print("Face detection will not work properly.")
                self.landmarker = None
                self.use_landmarker = False
            
            self.frame_count = 0
            
        def process(self, image):
            """
            Process image and return results in old API format
            """
            if not self.use_landmarker or self.landmarker is None:
                # Return empty results if landmarker not available
                return MockResults(None)
            
            try:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Process with new API
                if self.static_image_mode:
                    detection_result = self.landmarker.detect(mp_image)
                else:
                    self.frame_count += 1
                    detection_result = self.landmarker.detect_for_video(mp_image, self.frame_count)
                
                # Convert new API results to old API format
                if detection_result and detection_result.face_landmarks:
                    face_landmarks_list = []
                    for face_landmarks in detection_result.face_landmarks:
                        # Convert landmarks
                        landmarks = []
                        for landmark in face_landmarks:
                            landmarks.append(MockLandmark(landmark.x, landmark.y, landmark.z))
                        face_landmarks_list.append(MockLandmarkList(landmarks))
                    
                    return MockResults(face_landmarks_list)
                else:
                    return MockResults(None)
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                return MockResults(None)
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.landmarker:
                self.landmarker.close()
    
    class MockSolutions:
        class drawing_utils:
            @staticmethod
            def draw_landmarks(image, landmark_list, connections=None, 
                             landmark_drawing_spec=None, connection_drawing_spec=None):
                """Draw landmarks on image"""
                if landmark_list is None:
                    return
                
                h, w, _ = image.shape
                for landmark in landmark_list.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        class drawing_styles:
            @staticmethod
            def get_default_face_mesh_tesselation_style():
                return None
            
            @staticmethod
            def get_default_face_mesh_contours_style():
                return None
            
            @staticmethod
            def get_default_face_mesh_iris_connections_style():
                return None
        
        class face_mesh:
            FACEMESH_TESSELATION = []
            FACEMESH_CONTOURS = []
            FACEMESH_IRISES = []
            FACEMESH_LEFT_EYE = []
            FACEMESH_RIGHT_EYE = []
            
            FaceMesh = MockFaceMesh
    
    mp.solutions = MockSolutions()

print("MediaPipe compatibility layer loaded successfully")
