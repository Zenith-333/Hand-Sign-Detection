import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Detector",
    page_icon="🤟",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }
    h1 { color: #1aff91; font-family: monospace; }
    .metric-box {
        background: #1a1a1a;
        border: 1px solid #1aff91;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ── Header ───────────────────────────────────────────────────
st.title("🤟 Hand Gesture Detector")
st.markdown("Real-time hand gesture detection using **YOLOv8** — trained on 5 gestures: *one, two, three, four, five*")
st.divider()

# ── Mode selector ────────────────────────────────────────────
mode = st.radio("Choose input mode:", ["📷 Webcam (Real-time)", "🖼️ Upload Image", "🎥 Upload Video"], horizontal=True)
conf_thresh = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

st.divider()

# ── Helper: run detection on a frame ─────────────────────────
def detect(frame_bgr):
    results = model.predict(frame_bgr, conf=conf_thresh, verbose=False)[0]
    annotated = results.plot()
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])
        detections.append((label, confidence))
    return annotated, detections

# ── Mode: Webcam ─────────────────────────────────────────────
if mode == "📷 Webcam (Real-time)":
    st.warning("⚠️ Webcam access is not supported on Streamlit Cloud. Please run the app locally for live webcam detection, or use the Image/Video upload modes.")

    run = st.checkbox("▶ Start Webcam (local only)")
    FRAME_WINDOW = st.image([])
    detection_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. This feature only works when running locally.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot read from webcam.")
                    break

                annotated, detections = detect(frame)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated_rgb, channels="RGB", use_container_width=True)

                if detections:
                    det_text = " | ".join([f"**{lbl}** ({conf:.0%})" for lbl, conf in detections])
                    detection_placeholder.markdown(f"🟢 Detected: {det_text}")
                else:
                    detection_placeholder.markdown("🔴 No gesture detected")

            cap.release()

# ── Mode: Upload Image ────────────────────────────────────────
elif mode == "🖼️ Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        annotated, detections = detect(frame_bgr)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("**Detected**")
            st.image(annotated_rgb, use_container_width=True)

        if detections:
            st.success("Detections:")
            for lbl, conf in detections:
                st.markdown(f'<div class="metric-box">🤟 <b>{lbl}</b> — {conf:.1%} confidence</div>', unsafe_allow_html=True)
        else:
            st.warning("No gestures detected. Try a clearer image or lower the confidence threshold.")

# ── Mode: Upload Video ────────────────────────────────────────
elif mode == "🎥 Upload Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        st.info("Processing video... this may take a moment.")
        cap = cv2.VideoCapture(tfile.name)

        out_path = tfile.name.replace(".mp4", "_out.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        progress = st.progress(0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        FRAME_WINDOW = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = detect(frame)
            out.write(annotated)

            if frame_idx % 10 == 0:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(rgb, use_container_width=True)
                progress.progress(min(frame_idx / max(total, 1), 1.0))
            frame_idx += 1

        cap.release()
        out.release()
        progress.progress(1.0)
        st.success("✅ Video processed!")

        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download processed video", f, file_name="hand_gesture_output.mp4", mime="video/mp4")

        os.unlink(tfile.name)
