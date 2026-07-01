import sys
from PyQt5.QtWidgets import QApplication, QStackedWidget
from PyQt5.QtCore import Qt
from ui.main_menu import MainMenu
from ui.learning_page import LearningPage

class BIM_App(QStackedWidget):
    def __init__(self):
        super().__init__()
        # Set window size suitable for Raspberry Pi 7-inch screen (approx 1024x600)
        self.setWindowTitle("BIM Learning System")
        self.resize(1024, 600)
	self.setWindowFlags(Qt.FramelessWindowHint)
        
        # Initialize UI pages
        self.main_menu = MainMenu(self)
        self.learning_page = LearningPage(self)

        # Add pages to the stack
        self.addWidget(self.main_menu)
        self.addWidget(self.learning_page)

        # Show Main Menu on startup
        self.setCurrentWidget(self.main_menu)

    def go_to_learning(self, lesson_name):
        """Triggered when a lesson button is clicked in the main menu."""
        self.learning_page.load_lesson(lesson_name)
        self.setCurrentWidget(self.learning_page)

    def go_to_menu(self):
        """Triggered when 'Exit' is clicked in the learning page."""
        self.learning_page.stop_lesson()
        self.setCurrentWidget(self.main_menu)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Global stylesheet for a cleaner look
    app.setStyleSheet("""
        QWidget { background-color: #F8F9FA; }
        QPushButton { border-radius: 10px; font-weight: bold; }
    """)
    window = BIM_App()
    window.show()
    sys.exit(app.exec_())