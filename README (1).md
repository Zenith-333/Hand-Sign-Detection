---
title: Hand Gesture Detector
emoji: 🤟
colorFrom: green
colorTo: black
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 🤟 Hand Gesture Detector

Real-time hand gesture detection using **YOLOv8**, trained on the [Hand Gesture Recognition dataset](https://universe.roboflow.com/lebanese-university-grkoz/hand-gesture-recognition-y5827).

## Features
- 📷 **Live webcam** detection
- 🖼️ **Image upload** detection
- 🎥 **Video upload** with downloadable output

## Gestures Detected
| Gesture | Fingers |
|---------|---------|
| one | ☝️ 1 finger |
| two | ✌️ 2 fingers |
| three | 🤟 3 fingers |
| four | 🖖 4 fingers |
| five | 🖐️ 5 fingers |

## Model
- Architecture: YOLOv8n
- Trained for 100 epochs on 839 images
- Input size: 416×416
