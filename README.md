# Facial Recognition System

A simple but effective Facial Recognition System built with Python and OpenCV.
This project allows you to:
- Capture and store faces from a webcam.
- Import images of known people (e.g., celebrities).
- Train a recognition model automatically.
- Recognize faces in real-time via webcam or from uploaded images.

# Features
- Face collection -> Collect face data using webcam or import image folders
- Automatic Training -> Faces are automatically trained into a model
- Recogition Modes:
  - Webcam recognition (live)
  - Single image recognition
- Confidence Display -> Shows prediction confidence with each recognition
- Error Handling -> Handles missing images or invalid paths gracefully

# Project-Structure

Facial-recognition/
│
├── dataset/           # Stores collected/imported face images
│   ├── .gitkeep
│   
│
├── models/            # Stores trained models
│   ├── .gitkeep
│   
│
├── collect_faces.py   # Capture/import faces into dataset
├── trainer.py         # Train model from dataset
├── recognizer.py      # Recognize faces (webcam or image)
├── main.py            # Central script to run all features
│
└── README.md          # Project documentation
└── .gitignore

# Installation

1. Clone this repository:
git clone https://github.com/your-username/Facial-recognition.git
cd Facial-recognition

2. Install dependencies
pip install opencv-python numpy

# Usage

Run the project from main.py:
- python main.py

You'll be given options to:
1. Capture new face data via a webcam
2. Import a folder of face images
3. Train the model
4. Recognize faces (Webcam or image model)

