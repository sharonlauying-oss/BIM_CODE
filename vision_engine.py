# core/vision_engine.py
import os
import cv2
import urllib.request
import importlib
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Landmark connections vector to draw the skeleton mesh map
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class SmoothedPoint:
    """
    Lightweight data proxy class that mimics MediaPipe landmarks.
    Allows standard .x, .y, .z dot-notation access while bypassing MediaPipe's read-only limits.
    """
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class VisionEngine(QThread):
    frame_ready = pyqtSignal(QImage)
    gesture_output = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._is_running = False
        self.cap = None
        self.current_lesson = ""
        self.current_step = 1
        self.gesture_instance = None
        
        # ─── 🧊 EMA Filter Smoothing Configuration ───
        # alpha: Smoothing coefficient. Lower = smoother (higher lag), Higher = faster response.
        # Range 0.50 - 0.65 is optimal for gesture tracking.
        self.alpha = 0.55  
        
        # Cache memory split by left/right hand to completely prevent cross-contamination errors
        self.prev_hands = {"Right": None, "Left": None}  
        
        print("[Engine] Initializing Vision Engine...")
        os.makedirs("assets/models", exist_ok=True)
        self.model_path = 'assets/models/hand_landmarker.task'
        self._download_model_if_needed()

        try:
            from mediapipe.tasks.python import BaseOptions
	    base_options = BaseOptions(
	        model_asset_path=self.model_path,
	        delegate=BaseOptions.Delegate.CPU
	    )
	    options = vision.HandLandmarkerOptions(
	        base_options=base_options, 
	        num_hands=2,
	        model_complexity=0
	    )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("[Engine] MediaPipe Multi-Hand System Initialized.")
        except Exception as e:
            print(f"[Engine ERROR] Initialization failed: {e}")

    def _download_model_if_needed(self):
        if not os.path.exists(self.model_path):
            print("[Engine] File missing. Downloading hand model task...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

    def set_lesson(self, lesson_name):
        self.current_lesson = lesson_name
        module_name = f"gestures.{lesson_name.replace(' ', '_').lower()}"
        try:
            gesture_module = importlib.import_module(module_name)
            importlib.reload(gesture_module)
            if hasattr(gesture_module, 'GestureChecker'):
                self.gesture_instance = gesture_module.GestureChecker()
                print(f"[Engine] Loaded bound module for {module_name}")
            else:
                self.gesture_instance = None
        except Exception as e:
            print(f"[Engine WARNING] Loading module failed: {e}")
            self.gesture_instance = None

    def set_step(self, step: int):
        self.current_step = step
        if self.gesture_instance and hasattr(self.gesture_instance, 'reset_dynamic_tracking'):
            self.gesture_instance.reset_dynamic_tracking()

    def run(self):
        self._is_running = True
        self.cap = cv2.VideoCapture(0) 
    
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.msleep(30)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(mp_image)
                
                output_data = False if self.current_step == 1 else (self.gesture_instance.stroke_count if self.gesture_instance else 0)
                
                if detection_result and detection_result.hand_landmarks:
                    h, w, _ = frame.shape
                    raw_hands_list = []
                    detected_hands_this_frame = set()  # Track hand labels detected in the current frame
            
                    for idx, (hand_landmarks, hand_world_landmarks) in enumerate(zip(detection_result.hand_landmarks, detection_result.hand_world_landmarks)):
                        handedness_label = detection_result.handedness[idx][0].category_name if idx < len(detection_result.handedness) else "Unknown"
                        real_hand = "Left" if handedness_label == "Right" else "Right"
                        detected_hands_this_frame.add(real_hand)
                
                        # ─── 🎛️ EMA Filter Core Injection ───
                        smoothed_lm = []
                        smoothed_wm = []
                        prev_data = self.prev_hands.get(real_hand)
                        
                        if prev_data is None:
                            # First frame or hand just re-entered screen: Baseline tracking setup
                            smoothed_lm = [SmoothedPoint(pt.x, pt.y, pt.z) for pt in hand_landmarks]
                            smoothed_wm = [SmoothedPoint(pt.x, pt.y, pt.z) for pt in hand_world_landmarks]
                        else:
                            # Continuous tracking frames: Blend historical values with new data to filter spikes
                            prev_lm = prev_data["lm"]
                            prev_wm = prev_data["wm"]
                            for i in range(21):
                                # 1. Smooth 2D normalized screen landmarks
                                lx = self.alpha * hand_landmarks[i].x + (1 - self.alpha) * prev_lm[i].x
                                ly = self.alpha * hand_landmarks[i].y + (1 - self.alpha) * prev_lm[i].y
                                lz = self.alpha * hand_landmarks[i].z + (1 - self.alpha) * prev_lm[i].z
                                smoothed_lm.append(SmoothedPoint(lx, ly, lz))
                                
                                # 2. Smooth 3D Metric World Coordinate landmarks
                                wx = self.alpha * hand_world_landmarks[i].x + (1 - self.alpha) * prev_wm[i].x
                                wy = self.alpha * hand_world_landmarks[i].y + (1 - self.alpha) * prev_wm[i].y
                                wz = self.alpha * hand_world_landmarks[i].z + (1 - self.alpha) * prev_wm[i].z
                                smoothed_wm.append(SmoothedPoint(wx, wy, wz))
                        
                        # Save smoothed tracking matrix to memory cache
                        self.prev_hands[real_hand] = {"lm": smoothed_lm, "wm": smoothed_wm}
                        # ─── 🎛️ EMA Filter End ───

                        hand_data = {
                            "landmarks": smoothed_lm,         # Sends out perfectly smoothed 2D coordinates
                            "world_landmarks": smoothed_wm,   # Sends out perfectly smoothed 3D coordinates
                            "real_hand": real_hand
                        }
                        raw_hands_list.append(hand_data)
                    
                    # Safety flush: Clear memory cache for missing hands to prevent teleporting/stretching bugs
                    for hand_type in ["Right", "Left"]:
                        if hand_type not in detected_hands_this_frame:
                            self.prev_hands[hand_type] = None
            
                    # ─── 🚀 2. Core Routing Filter: Dynamically Read Hand Mode Configuration ───
                    # Replaced hardcoded list with flexible object attribute querying
                    if self.gesture_instance and getattr(self.gesture_instance, "is_double_hand", False):
                        final_hands_list = raw_hands_list  # Double Hand Mode: Accept all detected hands
                    else:
                        final_hands_list = [hand for hand in raw_hands_list if hand["real_hand"] == "Right"]  # Single Hand Mode: Right hand only
            
                    # 3. Draw filtered, ultra-smooth hand skeleton layout onto the visual image canvas
                    for hand in final_hands_list:
                        lm = hand["landmarks"]
                        for connection in HAND_CONNECTIONS:
                            start_pt = (int(lm[connection[0]].x * w), int(lm[connection[0]].y * h))
                            end_pt = (int(lm[connection[1]].x * w), int(lm[connection[1]].y * h))
                            cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
                        for point in lm:
                            cv2.circle(frame, (int(point.x * w), int(point.y * h)), 4, (0, 0, 255), -1)
            
                    # 4. Pipeline Execution: Feed tracking data directly into the active checker subclass
                    if self.gesture_instance:
                        output_data = self.gesture_instance.check_gesture(final_hands_list, self.current_step)
                        self.recognition_result = output_data
                        self.gesture_output.emit(output_data)

                # Canvas rendering pipeline preparation
                rgb_to_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h_f, w_f, ch_f = rgb_to_show.shape
                qt_image = QImage(rgb_to_show.data, w_f, h_f, ch_f * w_f, QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)
                
            except Exception as e:
                print(f"[Engine Loop Error]: {e}")

            self.msleep(30)

        if self.cap and self.cap.isOpened():
            self.cap.release()

    def stop(self):
        self._is_running = False
        self.wait()