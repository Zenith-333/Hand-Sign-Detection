import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import glob

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Detector",
    page_icon="🤟",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background-color: #0a0a0a; color: #f0f0f0; }
    .title-block {
        background: linear-gradient(135deg, #00ff88 0%, #00c4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .subtitle {
        color: #888;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #00ff88;
        font-family: 'Space Mono', monospace;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .info-box {
        background: #111;
        border-left: 3px solid #00ff88;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #aaa;
        margin: 0.8rem 0;
    }
    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid #2a2a2a;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ── Emoji Map ──────────────────────────────────────────────────
CLASS_EMOJI = {
    "one":   "☝️",
    "two":   "✌️",
    "three": "🤟",
    "four":  "🖖",
    "five":  "🖐️",
}

# ── Header ─────────────────────────────────────────────────────
st.markdown('<div class="title-block">🤟 Hand Gesture Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLOv8 Object Detection · AI 8.0 Lab Activity · Lebanese University Dataset</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    Detects hand gestures: one ☝️ · two ✌️ · three 🤟 · four 🖖 · five 🖐️<br>
    Upload an image or video — the model will draw bounding boxes with class labels and confidence scores.
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📸 Image Detection", "🎥 Video Detection"])

# ── Helper ─────────────────────────────────────────────────────
def run_detection(image_np, conf=0.25):
    results = model.predict(image_np, conf=conf, iou=0.45, verbose=False)
    result = results[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label  = result.names[cls_id]
        conf_  = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": conf_,
            "bbox": (x1, y1, x2, y2)
        })
    return annotated, detections

# ── TAB 1: Image Detection ─────────────────────────────────────
with tab1:
    st.markdown("### Upload a Hand Gesture Image")
    st.markdown("""
    <div class="info-box">
        💡 Take a clear photo of your hand showing 1–5 fingers with good lighting and a simple background.
    </div>
    """, unsafe_allow_html=True)

    conf_img = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05, key="img_conf")
    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        annotated, detections = run_detection(image_np, conf=conf_img)

        with col2:
            st.markdown("**Detection Output**")
            st.image(annotated, use_container_width=True)

        st.markdown("---")
        st.markdown("### Detection Results")

        if detections:
            cols = st.columns(min(len(detections), 4))
            for i, d in enumerate(detections):
                emoji = CLASS_EMOJI.get(d["label"].lower(), "🤚")
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:2.2rem">{emoji}</div>
                        <div class="metric-value">{d['confidence']:.0%}</div>
                        <div class="metric-label">{d['label']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("#### Bounding Box Details")
            for i, d in enumerate(detections):
                emoji = CLASS_EMOJI.get(d["label"].lower(), "🤚")
                x1, y1, x2, y2 = d["bbox"]
                st.markdown(f"""
                <div class="info-box">
                    {emoji} <strong>{d['label'].upper()}</strong> —
                    Confidence: <strong>{d['confidence']:.2%}</strong> |
                    Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No hand gesture detected. Try lowering the confidence threshold or use a clearer image.")

# ── TAB 2: Video Detection ─────────────────────────────────────
with tab2:
    st.markdown("### Upload a Hand Gesture Video")
    st.markdown("""
    <div class="info-box">
        💡 Record a short video showing hand gestures (one to five fingers) and upload it here.
        The model detects gestures frame by frame and saves the output video.
    </div>
    """, unsafe_allow_html=True)

    conf_vid = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05, key="vid_conf")
    uploaded_vid = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.flush()

        st.info("⏳ Processing video... please wait.")
        progress = st.progress(0)
        status   = st.empty()

        cap          = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix="_output.mp4").name
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out      = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        all_detections = []
        frame_idx      = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results  = model.predict(frame, conf=conf_vid, iou=0.45, verbose=False)
            annotated = results[0].plot()
            out.write(annotated)
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label  = results[0].names[cls_id]
                all_detections.append(label)
            frame_idx += 1
            progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
            status.text(f"Processing frame {frame_idx} / {total_frames}")

        cap.release()
        out.release()
        status.empty()

        st.success(f"✅ Done! {frame_idx} frames processed.")

        # Display output video
        with open(out_path, "rb") as f:
            video_bytes = f.read()

        st.markdown("**Detection Output Video:**")
        st.video(video_bytes)

        st.download_button(
            label="⬇️ Download Output Video",
            data=video_bytes,
            file_name="hand_gesture_output.mp4",
            mime="video/mp4"
        )

        # Gesture summary
        if all_detections:
            st.markdown("---")
            st.markdown("### Gesture Summary")
            from collections import Counter
            counts = Counter(all_detections)
            cols   = st.columns(min(len(counts), 5))
            for i, (label, count) in enumerate(counts.most_common()):
                emoji = CLASS_EMOJI.get(label.lower(), "🤚")
                with cols[i % 5]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:2rem">{emoji}</div>
                        <div class="metric-value">{count}</div>
                        <div class="metric-label">{label} frames</div>
                    </div>
                    """, unsafe_allow_html=True)

        os.unlink(tfile.name)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#444; font-family:'Space Mono',monospace; font-size:0.72rem;">
    AI 8.0 Lab Activity · Hand Gesture Detection · YOLOv8n<br>
    Dataset: Hand Gesture Recognition — Lebanese University · Roboflow Universe
</div>
""", unsafe_allow_html=True)
