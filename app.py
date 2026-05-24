"""
Sign Language Interpreter using Deep Learning
Author: Syeda Fizzah Batool
Web interface powered by Streamlit
"""

import streamlit as st
import os

st.set_page_config(
    page_title="Sign Language Interpreter",
    page_icon="🤟",
    layout="wide",
)

st.title("🤟 Sign Language Interpreter using Deep Learning")
st.markdown("**By Syeda Fizzah Batool** | MIT License")
st.markdown("---")

# ── Optional heavy deps (not available on Streamlit Cloud — model not in repo) ──
try:
    import cv2
    import numpy as np
    import pickle
    import sqlite3
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model as _keras_load
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_PATH = os.path.join("Code", "cnn_model_keras2.h5")
HIST_PATH  = os.path.join("Code", "hist")
DB_PATH    = os.path.join("Code", "gesture_db.db")
model_ready = os.path.exists(MODEL_PATH) and os.path.exists(HIST_PATH)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app uses a **Convolutional Neural Network** trained on 44 "
        "American Sign Language (ASL) characters to interpret hand gestures "
        "in real time."
    )
    st.markdown("---")
    st.header("How it works")
    st.markdown(
        "1. The webcam feed is segmented using HSV back-projection.\n"
        "2. Hand contour is extracted from a 300×300 ROI.\n"
        "3. The contour image is resized to 50×50 and passed to the CNN.\n"
        "4. Predictions with >70 % confidence are accepted.\n"
        "5. Confirmed gestures are spoken aloud via text-to-speech."
    )
    st.markdown("---")
    st.header("Tech Stack")
    st.markdown("- Python 3\n- Keras / TensorFlow\n- OpenCV\n- SQLite\n- Streamlit")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_demo, tab_live, tab_setup = st.tabs(["📽️ Demo", "📷 Live Recognition", "⚙️ Setup Guide"])

# ── Demo tab ─────────────────────────────────────────────────────────────────
with tab_demo:
    st.subheader("Project Demonstrations")
    st.markdown(
        "The model recognises 44 ASL characters in real time from a webcam feed. "
        "Below are recorded demonstrations."
    )
    gif_dir = "img"
    gifs = [f for f in ["demo.gif", "demo2.gif", "demo3.gif", "demo4.gif", "demo5.gif"]
            if os.path.exists(os.path.join(gif_dir, f))]
    if gifs:
        col1, col2 = st.columns(2)
        for i, gif in enumerate(gifs):
            (col1 if i % 2 == 0 else col2).image(
                os.path.join(gif_dir, gif), use_container_width=True
            )
    else:
        st.info("Demo GIFs not found — make sure the `img/` folder is present.")

    st.markdown("---")
    st.subheader("Screenshots")
    shots = [f for f in ["Capture1.PNG", "Capture.PNG"]
             if os.path.exists(os.path.join(gif_dir, f))]
    if shots:
        cols = st.columns(len(shots))
        for col, shot in zip(cols, shots):
            col.image(os.path.join(gif_dir, shot), use_container_width=True)

# ── Live Recognition tab ──────────────────────────────────────────────────────
with tab_live:
    if not CV2_AVAILABLE or not TF_AVAILABLE:
        st.info(
            "### Live recognition is not available on the cloud deployment\n\n"
            "This tab requires **OpenCV** and **TensorFlow**, which are large "
            "packages that cannot be installed on Streamlit Cloud's free tier "
            "for Python 3.14.\n\n"
            "**To run live recognition locally:**\n"
            "```bash\n"
            "pip install tensorflow>=2.13,<2.16 opencv-python numpy scikit-learn h5py pyttsx3\n"
            "python Code/final.py\n"
            "```\n\n"
            "Or for the Streamlit version locally:\n"
            "```bash\n"
            "pip install streamlit tensorflow>=2.13,<2.16 opencv-python numpy\n"
            "streamlit run app.py\n"
            "```"
        )
    elif not model_ready:
        st.warning(
            "**Model not trained yet.**  \n"
            "Follow the **Setup Guide** tab to train the CNN and generate "
            "`Code/cnn_model_keras2.h5`, then re-run the app."
        )
    else:
        # ── Full live recognition (runs locally after training) ──────────────
        @st.cache_resource(show_spinner="Loading model…")
        def load_resources():
            import pickle
            m = _keras_load(MODEL_PATH)
            with open(HIST_PATH, "rb") as f:
                h = pickle.load(f)
            return m, h

        model, hist = load_resources()
        st.success("✅ Model loaded successfully!")

        def get_pred_text(pred_class):
            try:
                import sqlite3
                conn = sqlite3.connect(DB_PATH)
                row = conn.execute(
                    "SELECT g_name FROM gesture WHERE g_id=?", (pred_class,)
                ).fetchone()
                conn.close()
                return row[0] if row else ""
            except Exception:
                return ""

        def process_frame(bgr_img):
            img = cv2.flip(bgr_img, 1)
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            x, y, w, h = 300, 100, 300, 300
            h_img, w_img = thresh.shape[:2]
            x = min(x, w_img - 1); y = min(y, h_img - 1)
            w = min(w, w_img - x); h = min(h, h_img - y)
            roi = thresh[y:y+h, x:x+w]
            contours = cv2.findContours(roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            pred_text, confidence = "", 0.0
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 10000:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    save_img = roi[y1:y1+h1, x1:x1+w1]
                    if w1 > h1:
                        pad = int((w1 - h1) / 2)
                        save_img = cv2.copyMakeBorder(save_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, 0)
                    elif h1 > w1:
                        pad = int((h1 - w1) / 2)
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, 0)
                    save_img = cv2.resize(save_img, (50, 50))
                    arr = np.array(save_img, dtype=np.float32).reshape(1, 50, 50, 1)
                    probs = model.predict(arr, verbose=0)[0]
                    pred_class = int(np.argmax(probs))
                    confidence = float(np.max(probs)) * 100
                    if confidence > 70:
                        pred_text = get_pred_text(pred_class)
            annotated = bgr_img.copy()
            cv2.rectangle(annotated, (300, 100), (600, 400), (0, 255, 0), 2)
            return pred_text, confidence, roi, annotated

        st.markdown("Place your hand inside the **green box** region before capturing.")
        camera_image = st.camera_input("📸 Capture a gesture")
        if camera_image:
            nparr = np.frombuffer(camera_image.getvalue(), np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            pred_text, confidence, thresh_roi, annotated = process_frame(bgr)
            c1, c2, c3 = st.columns(3)
            c1.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Captured frame", use_container_width=True)
            c2.image(thresh_roi, caption="Hand threshold (ROI)", use_container_width=True)
            with c3:
                st.markdown("### Result")
                if pred_text:
                    st.success(f"**Gesture:** {pred_text}")
                    st.metric("Confidence", f"{confidence:.1f} %")
                else:
                    st.warning("No gesture detected (confidence < 70 %).")

# ── Setup Guide tab ───────────────────────────────────────────────────────────
with tab_setup:
    st.subheader("Training the Model — One-time Local Setup")
    st.markdown(
        "The trained model is **not bundled** in this repository because it must "
        "be trained on your own hand under your own lighting conditions for best accuracy."
    )
    st.markdown("""
**Step 1 — Install dependencies**
```bash
pip install tensorflow>=2.13,<2.16 opencv-python numpy scikit-learn h5py pyttsx3
```

---

**Step 2 — Calibrate hand detection**
```bash
cd Code
python set_hand_histogram.py
```
Place your hand in the on-screen grid → press **`c`** to capture → press **`s`** to save.

---

**Step 3 — Capture gesture samples**
```bash
python create_gestures.py
```
Enter a gesture ID (0, 1, 2 …) and a label (e.g. `Hello`).
Press **`c`** to start — collects 1 200 images per gesture automatically.

---

**Step 4 — Augment the dataset**
```bash
python Rotate_images.py
```
Flips every image horizontally, doubling the dataset size.

---

**Step 5 — Prepare train / val / test splits**
```bash
python load_images.py
```

---

**Step 6 — Train the CNN**
```bash
python cnn_model_train.py
```
Trains for 15 epochs and saves `cnn_model_keras2.h5` at peak validation accuracy.

---

**Step 7 — Run live recognition**
```bash
# Desktop window (full app with text-to-speech):
python final.py

# OR browser-based Streamlit version:
streamlit run app.py
```
    """)
    st.info(
        "💡 **Tip:** Capture gestures under consistent lighting and re-run "
        "`set_hand_histogram.py` whenever your lighting environment changes."
    )
    st.markdown("---")
    st.markdown(
        "**Project by Syeda Fizzah Batool** — MIT License  \n"
        "Real-time ASL interpretation using CNN + OpenCV + Streamlit"
    )
