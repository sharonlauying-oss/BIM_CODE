# BIM_CODE
Offline Sign Language Education Kit E2G06
BIM_Project/
│
├── main.py                  # 🚀 Single program entry, manages app launch & global page routing
├── requirements.txt         # Dependencies list for one-click environment setup on Raspberry Pi
│
├── assets/                  # 📦 All offline static resources for offline execution
│   ├── models/              # Store local MediaPipe model files (e.g. hand_landmarker.task)
│   ├── videos/              # Store tutorial videos (e.g. Hello.mp4, Tolong.mp4)
│   ├── images/              # Store button icons, gesture reference images, UI backgrounds
│   └── fonts/               # (Optional) Custom fonts to avoid garbled text on Raspberry Pi
│
├── ui/                      # 🖥️ UI module (only controls layout, no underlying algorithm logic)
│   ├── __init__.py
│   ├── main_menu.py         # Home menu page (touch swipe & vocabulary selection supported)
│   └── learning_page.py     # Learning page (video/image on left, camera feed on right + bottom buttons & progress bar)
│
├── core/                    # ⚙️ Core business module (core engine of the whole system)
│   ├── __init__.py
│   ├── vision_engine.py     # Initialize MediaPipe & camera capture (separate UI and algorithm to avoid frame lag)
│
└── gestures/                # ✌️ Independent gesture recognition rules (add one file per new vocabulary)
    ├── __init__.py
    ├── base_gesture.py      # Base parent class with unified standard interface for all gesture checks
    ├── hello.py             # Custom coordinate judgment logic for Hello gesture
    ├── selamat_jalan.py     # Custom coordinate judgment logic for selamat_jalan gesture
    ├── tolong.py            # Custom coordinate judgment logic for tolong gesture
    ├── A.py                 # Letters A ~ Z available
    └── ... .py              # Remaining letter files from B to Z

