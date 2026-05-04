# \# AI Eye Tracking System

# 

# \## Overview

# 

# This project is a real-time AI eye tracking system built with Python, OpenCV, and MediaPipe. The system uses webcam input to detect a user’s face, extract facial landmarks, identify eye regions, track pupil movement, estimate gaze direction, and monitor screen attention.

# 

# The project also includes an online learning attention monitoring use case, where the system estimates whether a student is attentive or distracted during a virtual learning session.

# 

# \---

# 

# \## Features

# 

# \- Real-time webcam input

# \- Face detection

# \- Facial landmark detection using MediaPipe Face Landmarker

# \- Eye region extraction

# \- Eye cropping

# \- Basic pupil tracking

# \- Gaze direction estimation

# &#x20; - Looking Left

# &#x20; - Looking Center

# &#x20; - Looking Right

# \- Screen attention tracking

# &#x20; - Attentive

# &#x20; - Distracted

# \- Attention score calculation

# \- Session timer

# \- Total attentive time tracking

# \- Distraction count

# \- Business use case: Online Learning Attention Monitor

# 

# \### Bonus Features

# 

# \- Blink detection using Eye Aspect Ratio

# \- Blink count display

# \- Heatmap visualization of approximate gaze direction

# \- Head pose estimation using OpenCV solvePnP

# &#x20; - Head Forward

# &#x20; - Head Left

# &#x20; - Head Right

# &#x20; - Head Up

# &#x20; - Head Down

# 

# \---

# 

# \## Technologies

# 

# The project was implemented using:

# 

# \- Python

# \- OpenCV

# \- MediaPipe

# \- NumPy

# \- Matplotlib

# 

# \---

# 

# \## Project Structure

# 

# ```text

# ai-eye-tracking/

# │

# ├── main.py

# ├── requirements.txt

# ├── README.md

# ├── face\_landmarker.task

# │

# ├── src/

# │   ├── \_\_init\_\_.py

# │   ├── face\_mesh.py

# │   ├── eye\_region.py

# │   ├── pupil\_tracking.py

# │   ├── gaze\_estimation.py

# │   ├── attention.py

# │   ├── session\_tracker.py

# │   ├── blink\_detection.py

# │   ├── heatmap.py

# │   └── head\_pose.py

# │

# └── outputs/

# &#x20;   └── heatmaps/

# Installation

# 1\. Clone the repository

# git clone https://github.com/Nassah101/ai-eye-tracking-system.git

# cd ai-eye-tracking-system

# 2\. Create a virtual environment

# python -m venv venv

# 3\. Activate the virtual environment

# Windows CMD

# venv\\Scripts\\activate.bat

# Windows PowerShell

# venv\\Scripts\\activate

# Mac/Linux

# source venv/bin/activate

# 4\. Install requirements

# pip install -r requirements.txt

# Requirements

# 

# The requirements.txt file should contain:

# 

# opencv-python

# mediapipe

# numpy

# matplotlib

# MediaPipe Model File

# 

# This project uses the MediaPipe Face Landmarker Tasks API.

# 

# Download the face\_landmarker.task model file from:

# 

# https://storage.googleapis.com/mediapipe-models/face\_landmarker/face\_landmarker/float16/latest/face\_landmarker.task

# 

# Place the file in the project root folder:

# 

# ai-eye-tracking/face\_landmarker.task

# How to Run

# 

# After installing the requirements and adding the face\_landmarker.task file, run:

# 

# python main.py

# 

# The webcam window will open and display the real-time eye tracking system.

# 

# Press:

# 

# q

# 

# to quit the application.

# 

# When the application closes, a gaze heatmap image will be saved automatically in:

# 

# outputs/heatmaps/

# Business Use Case

# Online Learning Attention Monitor

# 

# This system can be applied in online learning platforms to estimate whether a student is attentive during virtual classes, remote assessments, or digital learning sessions.

# 

# The application displays:

# 

# Gaze direction

# Attention state

# Attention score

# Session duration

# Total attentive time

# Distraction count

# Blink count

# Head direction

# Student engagement level

# 

# This can help educators and learning platforms understand student engagement patterns during online learning sessions.

# 

# How It Works

# 

# The system follows this pipeline:

# 

# Capture webcam frame using OpenCV.

# Detect face and facial landmarks using MediaPipe Face Landmarker.

# Extract left and right eye landmark points.

# Crop the eye regions from the video frame.

# Detect the pupil using grayscale conversion, blurring, thresholding, and contour detection.

# Estimate gaze direction based on pupil position inside the eye region.

# Estimate screen attention using face visibility, eye detection, and gaze direction.

# Detect blinks using Eye Aspect Ratio.

# Estimate head direction using selected face landmarks and OpenCV solvePnP.

# Record approximate gaze positions and save a heatmap after the session.

# Output Display

# 

# The real-time dashboard displays:

# 

# Online Learning Attention Monitor

# 

# Gaze Direction: Looking Center

# Head Direction: Head Forward

# Attention State: Attentive

# Attention Score: 85%

# Session Time: 00:45

# Attentive Time: 00:32

# Distraction Count: 2

# Blink Count: 4

# Student Engagement: Highly Engaged

# Heatmap Output

# 

# The system records approximate gaze positions during the session and saves a heatmap image after the user exits the program.

# 

# Example output path:

# 

# outputs/heatmaps/gaze\_heatmap\_YYYYMMDD\_HHMMSS.png

# 

# The heatmap gives a simple visual summary of where the user’s attention was concentrated during the session.

# 

# Limitations

# The system is sensitive to poor lighting.

# Pupil tracking may be affected by shadows, glasses, camera quality, and eye visibility.

# Gaze estimation is approximate and based on pupil position within the eye crop.

# The system is not calibrated for precise screen coordinates.

# The heatmap is an approximate gaze-direction visualization, not a full screen-calibrated gaze heatmap.

# Head pose estimation is based on approximate 3D face model points and may vary depending on camera angle.

# This is a prototype and not a medical or clinical-grade eye tracking system.

# Future Improvements

# 

# Possible improvements include:

# 

# Full screen calibration for more accurate gaze tracking

# More stable pupil detection under different lighting conditions

# Improved blink detection threshold calibration

# Better head pose smoothing

# Logging session results to CSV

# Generating a final attention report after each session

# Adding a graphical user interface

# Supporting multiple users

# Author

# 

# Hassan Wasiu

# 

# Project Status

# 

# Core assignment completed.

# 

# Completed modules:

# 

# Face detection

# Facial landmark detection

# Eye region extraction

# Pupil tracking

# Gaze estimation

# Screen attention tracking

# Online learning business use case

# Blink detection

# Heatmap visualization

# Head pose estimation

