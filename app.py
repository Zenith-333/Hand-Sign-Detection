import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Sign Detector",
    page_icon="🤚",
    layout="wide"
)

st.title("🤚 Hand Sign Detection")
st.markdown("Detect hand gestures using a YOLOv8 model trained on sign language dataset.")

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
class_names = model.names

st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Classes:**")
st.sidebar.markdown(", ".join(class_names.values()))

# ── Mode Selection ─────────────────────────────────────────────────────────────
mode = st.radio("Select Input Mode", ["📷 Image", "🎥 Video", "📸 Webcam"], horizontal=True)
st.markdown("---")

# ── Helper: Run Detection ──────────────────────────────────────────────────────
def run_detection(source, conf):
    results = model.predict(source=source, conf=conf, verbose=False)
    result = results[0]
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = []
    for box in result.boxes:
        cls_name = class_names[int(box.cls[0])]
        confidence_score = float(box.conf[0])
        detections.append((cls_name, confidence_score))

    return annotated_rgb, detections

# ── IMAGE MODE ─────────────────────────────────────────────────────────────────
if mode == "📷 Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Detection Result")
            with st.spinner("Detecting..."):
                annotated, detections = run_detection(img_array, confidence)
            st.image(annotated, use_column_width=True)

        st.subheader("📋 Detected Hand Signs")
        if detections:
            for cls_name, conf_score in detections:
                st.success(f"**{cls_name.upper()}** — Confidence: {conf_score:.2%}")
        else:
            st.warning("No hand signs detected. Try lowering the confidence threshold.")

# ── VIDEO MODE ─────────────────────────────────────────────────────────────────
elif mode == "🎥 Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        st.info("Processing video frame by frame...")
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video writer
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress = st.progress(0)
        frame_placeholder = st.empty()
        detection_counts = {}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, detections = run_detection(frame, confidence)
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            for cls_name, _ in detections:
                detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1

            # Show every 10th frame as preview
            if frame_idx % 10 == 0:
                frame_placeholder.image(annotated, caption=f"Frame {frame_idx}/{total_frames}", use_column_width=True)

            frame_idx += 1
            progress.progress(min(frame_idx / max(total_frames, 1), 1.0))

        cap.release()
        writer.release()

        st.success(f"✅ Done! Processed {frame_idx} frames.")

        st.subheader("📋 Detected Hand Signs (across all frames)")
        if detection_counts:
            for cls_name, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
                st.info(f"**{cls_name.upper()}** — detected in {count} frame(s)")
        else:
            st.warning("No hand signs detected.")

        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download Annotated Video", f, file_name="detected_output.mp4", mime="video/mp4")

        os.unlink(tfile.name)

# ── WEBCAM MODE ────────────────────────────────────────────────────────────────
elif mode == "📸 Webcam":
    st.info("📸 Capture a photo using your webcam below.")

    img_file = st.camera_input("Take a photo of your hand sign")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Captured Photo")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Detection Result")
            with st.spinner("Detecting..."):
                annotated, detections = run_detection(img_array, confidence)
            st.image(annotated, use_column_width=True)

        st.subheader("📋 Detected Hand Signs")
        if detections:
            for cls_name, conf_score in detections:
                st.success(f"**{cls_name.upper()}** — Confidence: {conf_score:.2%}")
        else:
            st.warning("No hand signs detected. Try again with better lighting.")

st.markdown("---")
st.caption("Lab 8.0 — Hand Gesture Detection | YOLOv8 | Artificial Intelligence 8.0")
