"""
Sign Language Interpreter using Deep Learning
Author: Syeda Fizzah Batool
Web interface powered by Streamlit
"""

import streamlit as st
import cv2
import numpy as np
import sqlite3
import pickle
import os

st.set_page_config(
    page_title="Sign Language Interpreter",
    page_icon="🤟",
    layout="wide",
)

st.title("🤟 Sign Language Interpreter using Deep Learning")
st.markdown("**By Syeda Fizzah Batool** | MIT License")
st.markdown("---")

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
    st.header("How to use")
    st.markdown(
        "1. Allow camera access when prompted.\n"
        "2. Place your hand inside the **green box** region.\n"
        "3. Click **Capture** — the model predicts your gesture.\n"
        "4. Predictions with >70 % confidence are accepted."
    )
    st.markdown("---")
    st.header("Tech Stack")
    st.markdown("- Python 3\n- Keras / TensorFlow\n- OpenCV\n- SQLite\n- Streamlit")

# ── Demo GIFs tab / Live tab ──────────────────────────────────────────────────
tab_demo, tab_live, tab_setup = st.tabs(["📽️ Demo", "📷 Live Recognition", "⚙️ Setup Guide"])

with tab_demo:
    st.subheader("Project Demonstrations")
    col1, col2 = st.columns(2)
    gif_dir = "img"
    gifs = [f for f in ["demo.gif","demo2.gif","demo3.gif","demo4.gif","demo5.gif"]
            if os.path.exists(os.path.join(gif_dir, f))]
    for i, gif in enumerate(gifs):
        (col1 if i % 2 == 0 else col2).image(
            os.path.join(gif_dir, gif), use_container_width=True
        )
    if not gifs:
        st.info("Demo GIFs not found. Make sure the `img/` folder is present.")

with tab_live:
    if not model_ready:
        st.warning(
            "**Model not found.** "
            "The CNN model must be trained before live recognition works. "
            "See the **Setup Guide** tab for instructions."
        )
        st.info(
            "Once you have trained the model (`Code/cnn_model_keras2.h5`) "
            "and the hand histogram (`Code/hist`), restart this app and live "
            "recognition will activate automatically."
        )
    else:
        @st.cache_resource(show_spinner="Loading model…")
        def load_resources():
            from tensorflow.keras.models import load_model as _load
            m = _load(MODEL_PATH)
            with open(HIST_PATH, "rb") as f:
                h = pickle.load(f)
            return m, h

        model, hist = load_resources()
        st.success("✅ Model loaded successfully!")

        def get_pred_text(pred_class):
            try:
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
            pred_text = ""
            confidence = 0.0
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

        camera_image = st.camera_input("📸 Capture a gesture (place hand in the green box area)")
        if camera_image:
            nparr = np.frombuffer(camera_image.getvalue(), np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            pred_text, confidence, thresh_roi, annotated = process_frame(bgr)
            c1, c2, c3 = st.columns(3)
            c1.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Captured frame (green box = hand region)", use_container_width=True)
            c2.image(thresh_roi, caption="Hand threshold (ROI)", use_container_width=True)
            with c3:
                st.markdown("### Result")
                if pred_text:
                    st.success(f"**Gesture:** {pred_text}")
                    st.metric("Confidence", f"{confidence:.1f} %")
                else:
                    st.warning("No gesture detected (confidence < 70 % or no hand found).")
                    if confidence > 0:
                        st.metric("Best confidence", f"{confidence:.1f} %")

with tab_setup:
    st.subheader("Training the Model (One-time Setup)")
    st.markdown(
        "The model is **not included** in the repository because it must be "
        "trained on your own hand gestures for best accuracy. Follow these steps:"
    )
    st.markdown("""
**Step 1 — Calibrate hand detection**
```
cd Code
python set_hand_histogram.py
```
Place your hand in the on-screen grid and press **`c`** to capture, then **`s`** to save.

---

**Step 2 — Capture gesture samples**
```
python create_gestures.py
```
Enter a gesture ID (0, 1, 2 …) and a label (e.g. `Hello`). Press **`c`** to start capturing.
The script collects 1200 images per gesture.

---

**Step 3 — Augment the dataset**
```
python Rotate_images.py
```
Flips every image to double the dataset.

---

**Step 4 — Prepare train/val/test splits**
```
python load_images.py
```

---

**Step 5 — Train the CNN**
```
python cnn_model_train.py
```
This saves `cnn_model_keras2.h5` when validation accuracy peaks.

---

**Step 6 — Run live recognition (desktop)**
```
python final.py
```
Or restart this Streamlit app for browser-based recognition.
    """)
    st.info(
        "💡 Tip: For the best results, capture gestures under consistent lighting "
        "and re-run `set_hand_histogram.py` if you change your environment."
    )
