import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

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
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    return YOLO(model_path)

try:
    model = load_model()
    class_names = model.names
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ── Friendly Labels ────────────────────────────────────────────────────────────
_raw_to_friendly = {
    "brother": "Brother", "call": "Call", "daughter": "Daughter",
    "father": "Father", "food": "Food", "good": "Good", "grand": "Grand",
    "hello": "Hello", "here": "Here", "how": "How",
    "iloveyou": "I Love You", "i love you": "I Love You",
    "love": "Love", "morning": "Morning", "mother": "Mother",
    "night": "Night", "no": "No", "nothing": "Nothing", "one": "One",
    "peace": "Peace", "ready": "Ready", "run": "Run", "sister": "Sister",
    "small": "Small", "son": "Son", "sorry": "Sorry", "stop": "Stop",
    "thank you": "Thank You", "thankyou": "Thank You", "thank_you": "Thank You",
    "what": "What", "when": "When", "where": "Where", "which": "Which",
    "who": "Who", "why": "Why", "you": "You",
}

def get_friendly_label(raw_cls_name: str) -> str:
    normalized = raw_cls_name.strip().lower().replace("_", " ")
    friendly = _raw_to_friendly.get(normalized, raw_cls_name.replace("_", " ").title())
    return f"{friendly} sign"

# ── Resize helper — keeps inference fast on CPU ────────────────────────────────
def resize_for_inference(img_array: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Downscale large images so inference doesn't time out on Streamlit Cloud."""
    h, w = img_array.shape[:2]
    if max(h, w) <= max_side:
        return img_array
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Gesture Classes:**")
st.sidebar.markdown(", ".join(class_names.values()))

with st.sidebar.expander("🔍 Debug: Raw model class names"):
    for idx, name in class_names.items():
        st.write(f"`{idx}`: `{name}`")

# ── Mode Selection ─────────────────────────────────────────────────────────────
mode = st.radio("Select Input Mode", ["📷 Image", "🎥 Video", "📸 Webcam"], horizontal=True)
st.markdown("---")

# ── Detection Helper ───────────────────────────────────────────────────────────
def run_detection(source, conf, is_bgr=False):
    """
    source  : numpy array (RGB or BGR depending on is_bgr)
    is_bgr  : True only for raw OpenCV video frames (cap.read() → BGR)
              PIL-sourced images are already RGB — do NOT convert them
    """
    # FIX 1: Only convert BGR→RGB for genuine OpenCV frames (video mode).
    # PIL images are already RGB; converting them would swap R and B channels
    # making the model see wrong colors → wrong/no detections.
    if is_bgr and isinstance(source, np.ndarray):
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    # FIX 2: Resize large images before inference.
    # Streamlit Cloud has no GPU. Phone photos (3000×4000+) cause inference to
    # take so long that the WebSocket connection times out (keepalive ping timeout
    # in logs). Capping at 640px matches YOLOv8's native training resolution.
    source = resize_for_inference(source, max_side=640)

    # FIX 3: CPU only — don't waste time trying CUDA first on Streamlit Cloud.
    results = model.predict(source=source, conf=conf, verbose=False, device="cpu")

    result = results[0]
    annotated = result.plot()                            # returns BGR
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detections = []
    for box in result.boxes:
        raw_cls = class_names[int(box.cls[0])]
        conf_score = float(box.conf[0])
        detections.append((get_friendly_label(raw_cls), conf_score))

    return annotated_rgb, detections

# ── IMAGE MODE ─────────────────────────────────────────────────────────────────
if mode == "📷 Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)   # already RGB

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, width="stretch")
        with col2:
            st.subheader("Detection Result")
            with st.spinner("Detecting..."):
                annotated, detections = run_detection(img_array, confidence, is_bgr=False)
            st.image(annotated, width="stretch")

        st.subheader("📋 Detected Hand Signs")
        if detections:
            for label, conf_score in detections:
                st.success(f"**{label.upper()}** — Confidence: {conf_score:.2%}")
        else:
            st.warning("No hand signs detected. Try lowering the confidence threshold.")

# ── VIDEO MODE ─────────────────────────────────────────────────────────────────
elif mode == "🎥 Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = out_file.name
        out_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress = st.progress(0)
        frame_placeholder = st.empty()
        detection_counts = {}
        frame_idx = 0

        st.info("Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # cap.read() returns BGR → is_bgr=True
            annotated, detections = run_detection(frame, confidence, is_bgr=True)
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            for label, _ in detections:
                detection_counts[label] = detection_counts.get(label, 0) + 1
            if frame_idx % 15 == 0:
                frame_placeholder.image(annotated, caption=f"Frame {frame_idx}/{total_frames}", width="stretch")
            frame_idx += 1
            progress.progress(min(frame_idx / max(total_frames, 1), 1.0))

        cap.release()
        writer.release()

        st.success(f"✅ Done! Processed {frame_idx} frames.")
        st.subheader("📋 Detected Hand Signs")
        if detection_counts:
            for label, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
                st.info(f"**{label.upper()}** — {count} frame(s)")
        else:
            st.warning("No hand signs detected.")

        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download Annotated Video", f, file_name="output.mp4", mime="video/mp4")

        os.unlink(tfile.name)

# ── WEBCAM MODE ────────────────────────────────────────────────────────────────
elif mode == "📸 Webcam":
    st.info("📸 Click below to capture a photo of your hand sign.")
    img_file = st.camera_input("Take a photo")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        img_array = np.array(image)   # already RGB

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Captured Photo")
            st.image(image, width="stretch")
        with col2:
            st.subheader("Detection Result")
            with st.spinner("Detecting..."):
                annotated, detections = run_detection(img_array, confidence, is_bgr=False)
            st.image(annotated, width="stretch")

        st.subheader("📋 Detected Hand Signs")
        if detections:
            for label, conf_score in detections:
                st.success(f"**{label.upper()}** — Confidence: {conf_score:.2%}")
        else:
            st.warning("No hand signs detected. Try again with better lighting.")

st.markdown("---")
st.caption("Lab 8.0 — Hand Gesture Detection | YOLOv8 | Artificial Intelligence 8.0")
