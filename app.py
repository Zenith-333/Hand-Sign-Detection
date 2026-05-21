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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #e8f4fd 0%, #f0e8ff 35%, #fde8f4 65%, #e8fdf4 100%);
        background-attachment: fixed;
        min-height: 100vh;
        color: #1a1a2e;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: -20%;
        left: -10%;
        width: 60%;
        height: 60%;
        background: radial-gradient(circle, rgba(99, 179, 237, 0.25) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: blobMove1 12s ease-in-out infinite alternate;
    }

    .stApp::after {
        content: '';
        position: fixed;
        bottom: -10%;
        right: -10%;
        width: 55%;
        height: 55%;
        background: radial-gradient(circle, rgba(183, 148, 246, 0.22) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: blobMove2 15s ease-in-out infinite alternate;
    }

    @keyframes blobMove1 {
        0%   { transform: translate(0, 0) scale(1); }
        100% { transform: translate(5%, 8%) scale(1.08); }
    }

    @keyframes blobMove2 {
        0%   { transform: translate(0, 0) scale(1); }
        100% { transform: translate(-6%, -5%) scale(1.06); }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.75);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.08), 0 2px 8px rgba(0,0,0,0.04);
        padding: 2rem;
        margin-bottom: 1.2rem;
        position: relative;
        z-index: 1;
    }

    .hero-wrap {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255,255,255,0.8);
        border-radius: 24px;
        padding: 2.5rem 2.8rem 2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 40px rgba(99,102,241,0.10);
        position: relative;
        overflow: hidden;
        z-index: 1;
    }

    .hero-wrap::before {
        content: '🤟';
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 7rem;
        opacity: 0.07;
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }

    .hero-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #6b7280;
        letter-spacing: 0.04em;
        font-weight: 400;
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(79,70,229,0.12), rgba(219,39,119,0.10));
        border: 1px solid rgba(79,70,229,0.2);
        border-radius: 50px;
        padding: 0.3rem 0.9rem;
        font-size: 0.72rem;
        font-weight: 600;
        color: #4f46e5;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        font-family: 'DM Mono', monospace;
    }

    .info-glass {
        background: rgba(255,255,255,0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.7);
        border-left: 3px solid #7c3aed;
        border-radius: 0 14px 14px 0;
        padding: 0.85rem 1.1rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #4b5563;
        margin: 0.8rem 0;
        box-shadow: 0 4px 16px rgba(124,58,237,0.06);
        position: relative;
        z-index: 1;
    }

    .metric-glass {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-radius: 18px;
        padding: 1.4rem 1rem;
        text-align: center;
        margin: 0.4rem 0;
        box-shadow: 0 4px 20px rgba(99,102,241,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        z-index: 1;
    }

    .metric-glass:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(99,102,241,0.14);
    }

    .metric-emoji { font-size: 2.2rem; margin-bottom: 0.4rem; }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        background: linear-gradient(120deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'DM Mono', monospace;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.72rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    .gesture-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }

    .gesture-pill {
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.9);
        border-radius: 50px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: #374151;
        box-shadow: 0 2px 8px rgba(99,102,241,0.07);
    }

    .bbox-row {
        background: rgba(255,255,255,0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.7);
        border-radius: 12px;
        padding: 0.75rem 1.1rem;
        margin: 0.5rem 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #374151;
        box-shadow: 0 2px 10px rgba(99,102,241,0.05);
        position: relative;
        z-index: 1;
    }

    .bbox-label { font-weight: 600; color: #4f46e5; }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.5);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 0.3rem;
        border: 1px solid rgba(255,255,255,0.75);
        gap: 0.3rem;
        position: relative;
        z-index: 1;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        color: #6b7280;
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.9) !important;
        color: #4f46e5 !important;
        box-shadow: 0 2px 12px rgba(99,102,241,0.12);
    }

    div[data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.8);
        box-shadow: 0 4px 20px rgba(99,102,241,0.08);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.45);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        border: 1.5px dashed rgba(124,58,237,0.3);
        padding: 0.5rem;
    }

    h3 { font-weight: 700; color: #1f2937; letter-spacing: -0.02em; }
    h4 { font-weight: 600; color: #374151; }
    hr { border: none; border-top: 1px solid rgba(99,102,241,0.12); margin: 1.2rem 0; }

    .footer-glass {
        background: rgba(255,255,255,0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.7);
        border-radius: 14px;
        padding: 1rem 1.5rem;
        text-align: center;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: #9ca3af;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.25); border-radius: 10px; }
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

# ── Hero Header ────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">YOLOv8 · Computer Vision</div>
    <div class="hero-title">Hand Gesture Detector</div>
    <div class="hero-sub">AI 8.0 Lab Activity &nbsp;·&nbsp; Lebanese University Dataset &nbsp;·&nbsp; Transfer Learning · 100 Epochs</div>
    <div class="gesture-pills" style="margin-top:1rem;">
        <span class="gesture-pill">☝️ One</span>
        <span class="gesture-pill">✌️ Two</span>
        <span class="gesture-pill">🤟 Three</span>
        <span class="gesture-pill">🖖 Four</span>
        <span class="gesture-pill">🖐️ Five</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📸  Image Detection", "🎥  Video Detection"])

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
    <div class="info-glass">
        💡 Take a clear photo of your hand showing 1–5 fingers with good lighting and a plain background for best results.
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
                    <div class="metric-glass">
                        <div class="metric-emoji">{emoji}</div>
                        <div class="metric-value">{d['confidence']:.0%}</div>
                        <div class="metric-label">{d['label']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("#### Bounding Box Details")
            for d in detections:
                emoji = CLASS_EMOJI.get(d["label"].lower(), "🤚")
                x1, y1, x2, y2 = d["bbox"]
                st.markdown(f"""
                <div class="bbox-row">
                    {emoji} <span class="bbox-label">{d['label'].upper()}</span>
                    &nbsp;—&nbsp; Confidence: <strong>{d['confidence']:.2%}</strong>
                    &nbsp;|&nbsp; Box: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No hand gesture detected. Try lowering the confidence threshold or use a clearer image.")

# ── TAB 2: Video Detection ─────────────────────────────────────
with tab2:
    st.markdown("### Upload a Hand Gesture Video")
    st.markdown("""
    <div class="info-glass">
        💡 Record a short clip showing hand gestures (1–5 fingers). The model processes each frame and outputs an annotated video.
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
            results   = model.predict(frame, conf=conf_vid, iou=0.45, verbose=False)
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
                    <div class="metric-glass">
                        <div class="metric-emoji">{emoji}</div>
                        <div class="metric-value">{count}</div>
                        <div class="metric-label">{label} frames</div>
                    </div>
                    """, unsafe_allow_html=True)

        os.unlink(tfile.name)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer-glass">
    AI 8.0 Lab Activity &nbsp;·&nbsp; Hand Gesture Detection &nbsp;·&nbsp; YOLOv8n<br>
    Dataset: Hand Gesture Recognition — Lebanese University · Roboflow Universe
</div>
""", unsafe_allow_html=True)
