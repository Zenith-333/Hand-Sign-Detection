# 🤚 Hand Sign Detection App

A Streamlit web app that detects hand gestures using a YOLOv8 model trained on sign language dataset.

## Features
- 📷 Upload an image
- 🎥 Upload a video
- 📸 Live webcam capture

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (including `best.pt`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Deploy!

## Files
- `app.py` — main Streamlit application
- `best.pt` — trained YOLOv8 model weights
- `requirements.txt` — Python dependencies
