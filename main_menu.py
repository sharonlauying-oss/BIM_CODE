from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QScrollArea, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MainMenu(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title = QLabel("Bahasa Isyarat Malaysia (BIM) Learning")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2C3E50; margin-bottom: 20px;")
        layout.addWidget(title)

        # Scroll Area for Touchscreen
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        # Container for the grid of buttons
        grid_widget = QWidget()
        grid_widget.setStyleSheet("background-color: transparent;")
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(20)

        # Define Lessons (A-Z, plus Words)
        alphabets = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        words = ["Hello", "Selamat Jalan", "Tolong", "Test"]
        all_lessons = alphabets + words

        # Populate buttons in a grid (e.g., 4 columns)
        col_count = 4
        for index, lesson in enumerate(all_lessons):
            btn = QPushButton(lesson)
            btn.setFixedSize(200, 100) # Big size for touch
            btn.setFont(QFont("Arial", 16, QFont.Bold))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                }
                QPushButton:pressed {
                    background-color: #2980B9;
                }
            """)
            # Lambda trick to pass the specific lesson name
            btn.clicked.connect(lambda checked, name=lesson: self.main_window.go_to_learning(name))
            
            row = index // col_count
            col = index % col_count
            grid_layout.addWidget(btn, row, col)

        scroll_area.setWidget(grid_widget)
        layout.addWidget(scroll_area)

        self.setLayout(layout)