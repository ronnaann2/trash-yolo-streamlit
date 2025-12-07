import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ---------------------------------------------------------
# 1. LOAD MODEL (With Caching)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# ---------------------------------------------------------
# 2. UI TABS
# ---------------------------------------------------------
st.title("Orlan's Junkshop Scrap Cleaner")
tab1, tab2 = st.tabs(["🔍 Scrap Checker", "📊 Model Performance"])

# ======================== TAB 1: SCRAP CHECKER ========================
with tab1:
    st.write("Upload an image or use your camera to detect scrap.")

    # Sidebar settings ONLY for this functionality
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    pass_threshold = st.sidebar.slider("Metal % Required for PASS", 0, 100, 95, 1)

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    camera_file = st.camera_input("Or take a picture")

    image_source = uploaded_file if uploaded_file else camera_file

    if image_source:
        image = Image.open(image_source)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Analyzing..."):
                results = model.predict(image, conf=conf_threshold)
                res_plotted = results[0].plot()

                st.image(res_plotted, caption="Detected Objects", use_container_width=True)

                # Object counts
                boxes = results[0].boxes
                total_objects = len(boxes)
                
                if total_objects == 0:
                    st.warning("⚠️ No objects detected. Try lowering confidence threshold.")
                else:
                    class_names = model.names  
                    metal_count = sum(1 for box in boxes if "metal" in class_names[int(box.cls[0])].lower())
                    trash_count = total_objects - metal_count
                    metal_percentage = (metal_count / total_objects) * 100

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Objects", total_objects)
                    col2.metric("🔩 Metal", f"{metal_count} ({metal_percentage:.1f}%)")
                    col3.metric("🗑️ Trash", f"{trash_count} ({100-metal_percentage:.1f}%)")

                    st.markdown("---")
                    if metal_percentage >= pass_threshold:
                        st.success(f"✅ PASS — Metal: {metal_percentage:.1f}% (Req: ≥{pass_threshold}%)")
                    else:
                        st.error(f"❌ FAIL — Metal: {metal_percentage:.1f}% (Req: ≥{pass_threshold}%)")


# ======================== TAB 2: METRICS ========================
with tab2:
    st.markdown("### Model Evaluation Metrics (on Test Set)")
    st.write("Performance of the trained YOLOv8 model based on validation/testing results.")

    colA, colB = st.columns(2)

    with colA:
        st.image("confusion_matrix.png", caption="Confusion Matrix")
        st.caption("Shows classification correctness for metal vs trash vs background.")

    with colB:
        st.image("f1_confidence_curve.png", caption="F1-Confidence Curve")
        st.caption("Indicates the best confidence threshold (≈0.455) for peak performance.")

    st.markdown("---")
    st.info("More metrics coming soon: Precision-Recall Curve, mAP charts, and per-class performance.")
