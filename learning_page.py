# ui/learning_page.py
import os
import cv2
import time
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar, QStackedWidget
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from core.vision_engine import VisionEngine

class LearningPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_lesson = ""
        self.current_step = 1
        self.has_two_steps = False
        
        # Video instruction systems
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        self.cap = None
        self.audio_player = QMediaPlayer(None, QMediaPlayer.LowLatency)
        
        # Precise timekeeping parameters for Step 1
        self.correct_gesture_start_time = None
        
        # Initialize background engine threads
        self.vision_engine = VisionEngine()
        
        self.init_ui()

    def init_ui(self):
        # Main layout for the entire page
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a Stacked Widget to switch between Learning UI and Congratulations UI
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # ==========================================
        # 1. LEARNING UI (Index 0)
        # ==========================================
        self.learning_widget = QWidget()
        learning_layout = QVBoxLayout(self.learning_widget)
        learning_layout.setContentsMargins(20, 20, 20, 20)

        # Top Display Area
        top_layout = QHBoxLayout()
        self.title_label = QLabel("Loading...")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #2C3E50;")
        
        self.step_label = QLabel("Step 1/1")
        self.step_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.step_label.setStyleSheet("color: #7F8C8D; background-color: #E5E8E8; padding: 5px; border-radius: 5px;")
        
        top_layout.addWidget(self.title_label)
        top_layout.addStretch()
        top_layout.addWidget(self.step_label)
        learning_layout.addLayout(top_layout)

        # Video / Camera Splits
        content_layout = QHBoxLayout()
        self.video_area = QLabel("Instructional Area")
        self.video_area.setStyleSheet("background-color: #2C3E50; border-radius: 10px; color: white;")
        self.video_area.setAlignment(Qt.AlignCenter)
        self.video_area.setMinimumSize(480, 360) 
        
        self.camera_area = QLabel("Camera Stream")
        self.camera_area.setStyleSheet("background-color: #000000; border-radius: 10px; color: #7F8C8D;")
        self.camera_area.setAlignment(Qt.AlignCenter)
        self.camera_area.setMinimumSize(480, 360)
        
        content_layout.addWidget(self.video_area, 1)
        content_layout.addWidget(self.camera_area, 1)
        learning_layout.addLayout(content_layout, 1)

        # UI Progress Feedback Trackers
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 2px solid #BDC3C7; border-radius: 8px; text-align: center; height: 32px; font-weight: bold; font-size: 14px;}
            QProgressBar::chunk { background-color: #2ECC71; border-radius: 6px; }
        """)
        learning_layout.addWidget(self.progress_bar)

        # Control Row Layout
        self.btn_exit = QPushButton("Exit to Menu")
        self.btn_exit.setFixedSize(180, 60)
        self.btn_exit.setStyleSheet("background-color: #E74C3C; color: white; font-size: 16px;")
        self.btn_exit.clicked.connect(self.safe_exit_to_menu) # 💡 Modified to safe handler

        self.btn_sim_next = QPushButton("Debug: Force Skip")
        self.btn_sim_next.setFixedSize(180, 60)
        self.btn_sim_next.setStyleSheet("background-color: #9B59B6; color: white; font-size: 14px;")
        self.btn_sim_next.clicked.connect(self.simulate_step_pass)

        self.btn_replay = QPushButton("Replay Video")
        self.btn_replay.setFixedSize(180, 60)
        self.btn_replay.setStyleSheet("background-color: #F39C12; color: white; font-size: 16px;")
        self.btn_replay.clicked.connect(self.replay_media)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.btn_exit)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_sim_next)
        control_layout.addWidget(self.btn_replay)
        learning_layout.addLayout(control_layout)
        
        self.stacked_widget.addWidget(self.learning_widget)

        # ==========================================
        # 2. CONGRATULATIONS UI (Index 1)
        # ==========================================
        self.congrats_widget = QWidget()
        congrats_layout = QVBoxLayout(self.congrats_widget)
        congrats_layout.setAlignment(Qt.AlignCenter)
        
        # Congrats Label
        self.congrats_label = QLabel("🎉 Congratulations!\nLesson Completed Successfully! 🎉")
        self.congrats_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.congrats_label.setStyleSheet("color: #27AE60; margin-bottom: 40px;")
        self.congrats_label.setAlignment(Qt.AlignCenter)
        
        # Buttons Layout
        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.setSpacing(30)
        
        self.btn_learn_again = QPushButton("Learn Again")
        self.btn_learn_again.setFixedSize(250, 80)
        self.btn_learn_again.setStyleSheet("background-color: #3498DB; color: white; font-size: 20px; font-weight: bold; border-radius: 15px;")
        self.btn_learn_again.clicked.connect(self.restart_lesson)
        
        self.btn_back_menu = QPushButton("Back to Main Menu")
        self.btn_back_menu.setFixedSize(250, 80)
        self.btn_back_menu.setStyleSheet("background-color: #E74C3C; color: white; font-size: 20px; font-weight: bold; border-radius: 15px;")
        self.btn_back_menu.clicked.connect(self.safe_exit_to_menu) # 💡 Modified to safe handler
        
        btn_layout.addWidget(self.btn_learn_again)
        btn_layout.addWidget(self.btn_back_menu)
        
        congrats_layout.addWidget(self.congrats_label)
        congrats_layout.addLayout(btn_layout)
        
        self.stacked_widget.addWidget(self.congrats_widget)

    def load_lesson(self, lesson_name):
        self.stacked_widget.setCurrentIndex(0) # Ensure we are showing the learning UI
        self.current_lesson = lesson_name
        self.current_step = 1
        self.progress_bar.setValue(0)
        self.correct_gesture_start_time = None
        
        self.setFocusPolicy(Qt.StrongFocus) # 允许页面接收强键盘焦点
        self.setFocus()                     # 立即夺回焦点
        QTimer.singleShot(100, lambda: self.setFocus()) # 100毫秒后再次强行锁死焦点，防止按钮抢走
        
        # 💡 Re-bind signals cleanly before starting thread
        try: self.vision_engine.frame_ready.disconnect()
        except: pass
        try: self.vision_engine.gesture_output.disconnect()
        except: pass
        
        self.vision_engine.frame_ready.connect(self.update_camera_frame)
        self.vision_engine.gesture_output.connect(self.handle_gesture_pipeline)
        
        self.vision_engine.set_lesson(lesson_name)
        if not self.vision_engine.isRunning():
            self.vision_engine.start()
        
        file_safe_name = self.current_lesson.replace(" ", "_")
        image_path = f"assets/images/{file_safe_name}.png"
        self.has_two_steps = os.path.exists(image_path)
        self.display_current_step_media()

    def handle_gesture_pipeline(self, data):
        if self.current_step == 1:
            is_valid = bool(data)
            if is_valid:
                if self.correct_gesture_start_time is None:
                    self.correct_gesture_start_time = time.time()
                
                elapsed = time.time() - self.correct_gesture_start_time
                progress = int((elapsed / 2.0) * 100)
                
                if progress >= 100:
                    self.progress_bar.setValue(100)
                    print("[System UI] Static posture requirement cleared!")
                    self.correct_gesture_start_time = None
                    self.simulate_step_pass()
                else:
                    self.progress_bar.setValue(progress)
            else:
                self.correct_gesture_start_time = None
                self.progress_bar.setValue(0)
                
        elif self.current_step == 2:
            stroke_count = int(data)
            if stroke_count == 0:
                self.progress_bar.setValue(0)
            elif stroke_count == 1:
                self.progress_bar.setValue(33)
            elif stroke_count == 2:
                self.progress_bar.setValue(66)
            elif stroke_count >= 3:
                self.progress_bar.setValue(100)
                print("[System UI] Dynamic tracking requirement cleared!")
                self.simulate_step_pass()

    def display_current_step_media(self):
        self.stop_video()
        self.vision_engine.set_step(self.current_step)
        file_safe_name = self.current_lesson.replace(" ", "_")
        
        if self.has_two_steps:
            self.step_label.setText(f"Step {self.current_step} / 2")
            if self.current_step == 1:
                self.title_label.setText(f"Learning: {self.current_lesson} (Hold Posture 2s)")
                self.btn_replay.setEnabled(False)
                self.btn_replay.setText("Static Reference")
                
                image_path = f"assets/images/{file_safe_name}.png"
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    self.video_area.setPixmap(pixmap.scaled(self.video_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.title_label.setText(f"Learning: {self.current_lesson} (Wave Hand 3 Times)")
                self.btn_replay.setEnabled(True)
                self.btn_replay.setText("Replay Video")
                self.start_video_playback(f"assets/videos/{file_safe_name}.mp4")
        else:
            self.step_label.setText("Step 1 / 1")
            self.title_label.setText(f"Learning: {self.current_lesson}")
            self.btn_replay.setEnabled(True)
            self.start_video_playback(f"assets/videos/{file_safe_name}.mp4")

    def start_video_playback(self, video_path):
        if os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_timer.start(int(1000 / fps) if fps > 0 else 33)
            self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(video_path))))
            self.audio_player.play()
        else:
            self.video_area.setText(f"Missing Resource Asset:\n{video_path}")

    def update_video_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
                self.video_area.setPixmap(QPixmap.fromImage(qt_image).scaled(self.video_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.stop_video()

    def update_camera_frame(self, qt_image):
        self.camera_area.setPixmap(QPixmap.fromImage(qt_image).scaled(self.camera_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def simulate_step_pass(self):
        self.progress_bar.setValue(0)
        self.correct_gesture_start_time = None
        if self.has_two_steps and self.current_step == 1:
            self.current_step = 2
            self.display_current_step_media()
        else:
            self.show_congratulations()

    def show_congratulations(self):
        """Displays the congratulations screen safely."""
        self.stop_video()
        self.shutdown_vision_engine_safely() # 💡 Shutdown thread right away to save resource
        self.stacked_widget.setCurrentIndex(1)

    def restart_lesson(self):
        self.load_lesson(self.current_lesson)
        self.setFocus() # 👈 让整个页面强行夺回键盘焦点，别留给按钮

    def safe_exit_to_menu(self):
        """💡 Thread-safe workflow to handle menu transitions without racing conditions."""
        self.stop_video()
        self.shutdown_vision_engine_safely()
        self.stop_lesson()
        self.main_window.go_to_menu()

    def shutdown_vision_engine_safely(self):
        """💡 Critical Fix: Disconnects and blocks till worker thread completely finishes C++ tasks."""
        # 1. Disconnect slots immediately so no residual frames trigger UI rendering
        try: self.vision_engine.frame_ready.disconnect()
        except: pass
        try: self.vision_engine.gesture_output.disconnect()
        except: pass
        
        # 2. Signal the loop state variable and join the underlying system thread
        if self.vision_engine.isRunning():
            self.vision_engine.stop()  # Sets _is_running = False and internal wait occurs inside engine.py
            self.vision_engine.wait()  # Hard block ensuring thread is 100% dead before returning

    # 🔥 【核心修改点】纯净媒体重播函数：剥离核心算法状态更新
    def replay_media(self):
        """🎥 纯净媒体重播：只负责停止并重新启动当前的视频与音频流，绝不发送 set_step 信号触碰算法状态计数"""
        if not self.current_lesson:
            return
            
        file_safe_name = self.current_lesson.replace(" ", "_")
        video_path = f"assets/videos/{file_safe_name}.mp4"
        
        # 1. 停止当前正在播放的媒体（释放视频cap和音频指针）
        self.stop_video()
        
        # 2. 仅仅重新开启视频流播放，完全不触碰底层 vision_engine 的状态
        self.start_video_playback(video_path)
        print(f"[UI] Pure Replay Video for '{self.current_lesson}' Step {self.current_step} - (Stroke count progress retained!)")

    def stop_video(self):
        self.video_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        try: self.audio_player.stop()
        except: pass

    def stop_lesson(self):
        self.current_lesson = ""
        self.current_step = 1
        self.progress_bar.setValue(0)
        self.correct_gesture_start_time = None
        self.video_area.clear()
        self.video_area.setText("Instructional Area")
        self.camera_area.clear()
        self.camera_area.setText("Camera Stream")
        self.stacked_widget.setCurrentIndex(0)

    def keyPressEvent(self, event):
        """
        🚀 通用全量数据采集器快捷键：
        - [空格] 单次捕获静态姿势（立即保存）
        - [S] 开始/停止连续采集（录制动态动作）
        - [R] 重置所有采集数据
        """
        # 拦截所有采集器快捷键，防止触发按钮点击
        if event.key() in [Qt.Key_Space, Qt.Key_S, Qt.Key_R]:
            # 只有在Test课程页面才生效
            if self.current_lesson == "Test":
                if hasattr(self.vision_engine, 'gesture_instance') and self.vision_engine.gesture_instance:
                    gesture = self.vision_engine.gesture_instance
                    
                    # 空格键：单次捕获
                    if event.key() == Qt.Key_Space:
                        gesture.trigger_single_capture()
                    
                    # S键：开始/停止连续采集
                    elif event.key() == Qt.Key_S:
                        gesture.toggle_collection()
                    
                    # R键：重置所有数据
                    elif event.key() == Qt.Key_R:
                        gesture.reset_all()
            
            # 🔥 关键：拦截事件，不传给父类，防止误点Exit按钮
            event.accept()
            return
        
        # 其他按键正常处理
        super().keyPressEvent(event)