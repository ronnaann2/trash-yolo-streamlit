import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Trash Detection", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("YOLOv8 Trash / Metal Detection")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    img_np = np.array(img)

    results = model(img_np)[0]
    annotated = results.plot()  # numpy array (BGR)

    st.image(img, caption="Original", use_column_width=True)
    st.image(annotated, caption="Detections", use_column_width=True)

    # simple summary
    classes = [model.names[int(c)] for c in results.boxes.cls]
    st.write("Detections:", classes)
