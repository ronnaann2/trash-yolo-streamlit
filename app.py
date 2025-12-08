import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. LOAD MODEL (With Caching)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO('best.pt')   # <-- your trained model

model = load_model()


# ---------------------------------------------------------
# 2. PIXEL PERCENTAGE CALCULATOR
# ---------------------------------------------------------
def compute_pixel_percentage(results, model):
    r = results[0]

    # No segmentation output
    if r.masks is None:
        return 0, 0, False

    total_metal_pixels = 0
    total_nonmetal_pixels = 0

    class_names = model.names

    # Loop through each mask + class
    for mask, cls in zip(r.masks.data, r.boxes.cls):
        mask_np = mask.cpu().numpy()
        pixel_count = np.sum(mask_np)

        label = class_names[int(cls)]

        if "metal" in label.lower():
            total_metal_pixels += pixel_count
        else:
            total_nonmetal_pixels += pixel_count

    total_pixels = total_metal_pixels + total_nonmetal_pixels

    if total_pixels == 0:
        return 0, 0, False

    metal_pct = (total_metal_pixels / total_pixels) * 100
    nonmetal_pct = 100 - metal_pct

    return metal_pct, nonmetal_pct, True



# ---------------------------------------------------------
# 3. UI TABS
# ---------------------------------------------------------
st.title("Orlan's Junkshop Scrap Cleaner")
tab1, tab2, tab3 = st.tabs(["🔍 Scrap Checker", "🗂 Dataset Overview", "📊 Model Performance"])



# ======================== TAB 1: SCRAP CHECKER ========================
with tab1:

    st.write("Upload an image or use your camera to detect scrap and compute pixel percentage of metal vs. trash.")

    # Sidebar settings
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    pass_threshold = st.sidebar.slider("Metal % Required for PASS", 0, 100, 95, 1)

    # Upload or camera
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    camera_file = st.camera_input("Or take a picture")
    image_source = uploaded_file if uploaded_file else camera_file

    if image_source:
        image = Image.open(image_source)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Analyzing..."):
                
                # Run YOLO detection
                results = model.predict(image, conf=conf_threshold)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Detected Objects", use_container_width=True)

                # Pixel percentage calculation
                metal_pct, nonmetal_pct, valid = compute_pixel_percentage(results, model)

                if not valid:
                    st.warning("⚠️ No segmentation masks detected. Try lowering confidence or use a clearer image.")
                else:
                    col1, col2 = st.columns(2)
                    col1.metric("🔩 Metal %", f"{metal_pct:.2f}%")
                    col2.metric("🗑️ Non-metal %", f"{nonmetal_pct:.2f}%")

                    st.markdown("---")

                    # Pass/Fail Logic
                    if metal_pct >= pass_threshold:
                        st.success(f"✅ PASS — Metal: {metal_pct:.1f}% (Required ≥ {pass_threshold}%)")
                    else:
                        st.error(f"❌ FAIL — Metal: {metal_pct:.1f}% (Required ≥ {pass_threshold}%)")



# ======================== TAB 2: DATASET OVERVIEW ========================
with tab2:

    st.image("data_cover.png", caption="Dataset Overview", use_container_width=True)

    st.markdown("### 🗂 Dataset Summary & Preparation")

    model_info = pd.DataFrame({
        "Property": ["Model", "Total Labeled Images", "Classes"],
        "Value": ["YOLOv8 Small Segmentation", "692", "2 (metal, trash)"]
    }).set_index("Property")
    st.table(model_info)

    st.markdown("---")
    st.subheader("Dataset Split Distribution")

    split_data = pd.DataFrame({
        "Split": ["Training", "Validation", "Testing"],
        "Images": [1452, 139, 69],
        "Percent": ["87%", "8%", "4%"]
    }).set_index("Split")
    st.table(split_data)

    st.info("More training data allows the model to learn better, while smaller test data ensures unbiased evaluation.")

    st.markdown("---")
    st.subheader("Preprocessing Applied")
    st.markdown(
        """
        - Auto-Orient — fixes camera rotation  
        - Resize — 640×640 for YOLO  
        - Adaptive Contrast — improves visibility in low light  
        """
    )

    st.markdown("---")
    st.subheader("Augmentations Used")
    st.markdown(
        """
        - Brightness variation: −20% to +20%  
        - Blur: up to 2.5px  
        - Noise: up to 0.1% pixels  
        - **3× augmentation** per training image  
        """
    )

    st.markdown("---")
    st.subheader("🔍 Key Dataset Insights")
    st.markdown(
        """
        - Designed for real junkshop scrap conditions  
        - Handles dirt, rust, and deformities  
        - Increases recycling profitability — prevents metal from being thrown away  
        """
    )



# ======================== TAB 3: MODEL PERFORMANCE ========================
with tab3:
    st.markdown("### 📊 Model Evaluation Metrics (on Test Set)")
    st.write("These metrics confirm the detector is reliable for real scrap operations.")

    st.image("train_metrics.png", caption="Training and Validation Loss Curves")
    st.write("Smooth downward loss trend → very strong learning & low overfitting.")

    st.markdown("---")

    st.image("confusion_matrix.png", caption="Confusion Matrix")
    st.write("Correct predictions dominate — system rarely throws away valuable metal.")

    st.markdown("---")

    st.image("f1_confidence_curve.png", caption="F1-Confidence Curve")
    st.write("Stable high F1 score even with imperfect scrap images → very robust model.")
