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
tab1, tab2, tab3 = st.tabs(["🔍 Scrap Checker", "📊 Model Performance", "🗂 Dataset Overview"])


# ======================== TAB 1: SCRAP CHECKER ========================
with tab1:
    st.write("Upload an image or use your camera to detect scrap.")

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


# ======================== TAB 2: MODEL METRICS ========================
with tab2:
    st.markdown("### Model Evaluation Metrics (on Test Set)")
    st.write(
        "These visualizations summarize how well the model performs in identifying and distinguishing "
        "scrap metal from trash. These results ensure the Scrap Checker can be trusted during "
        "real-world operations inside Orlan’s Junkshop."
    )

    st.image("train_metrics.png", caption="Training and Validation Loss Curves")
    st.markdown(
        """
        **Training Loss Interpretation**  
        • Both training and validation loss decrease smoothly → stable learning  
        • Very small gap → **low overfitting risk**  
        • Segmentation loss improves strongly → accurate object boundary detection  
        
        📌 The model **generalizes well** and handles unseen junk effectively.
        """
    )
    
    st.markdown("---")

    st.image("confusion_matrix.png", caption="Confusion Matrix")
    st.markdown(
        """
        **Classification Reliability**  
        • **912** correct metal classifications  
        • **236** correct trash classifications  
        • Few misclassifications → high trustworthiness  
        
        📌 The system prevents valuable metal from being mistakenly labeled as trash — increasing profit.
        """
    )

    st.markdown("---")

    st.image("f1_confidence_curve.png", caption="F1-Confidence Curve")
    st.markdown(
        """
        **Optimal Confidence for Real-World Use**  
        • Highest F1 ≈ **0.93** at 0.455 confidence  
        • Stable curve → robust even with imperfect scrap conditions  
        
        📌 Recommended default: **0.50 confidence** — strong balance of speed & accuracy.
        """
    )


# ======================== TAB 3: DATASET OVERVIEW ========================
with tab3:
    st.markdown("### 🗂 Dataset Summary & Preparation")

    st.table({
        "Property": ["Model", "Total Labeled Images", "Classes"],
        "Value": ["YOLOv8 Small", "692", "2 (metal, trash)"]
    })

    st.markdown("---")
    st.subheader("Dataset Split Distribution")
    st.table({
        "Split": ["Training", "Validation", "Testing"],
        "Images": [1452, 139, 69],
        "Percent": ["87%", "8%", "4%"]
    })
    st.info("More training data increases learning effectiveness while small test data confirms real performance.")

    st.markdown("---")
    st.subheader("Preprocessing Applied")
    st.markdown(
        """
        • Auto-Orient → fixes image rotation  
        • Resize → 640×640 to match YOLO required size  
        • Adaptive Contrast Enhancement → better detection in poor lighting  
        """
    )

    st.success("Preprocessing ensures clear visibility of scrap, even in low-light environments.")

    st.markdown("---")
    st.subheader("Augmentations Used")
    st.markdown(
        """
        • Brightness variation (−20% to +20%)  
        • Blur up to 2.5px  
        • Noise up to 0.1% pixels  
        • **3× synthetic versions per image**  
        """
    )
    st.warning("This improves real-world detection on rusted, bent, dirty, or motion-blurred scrap.")

    st.markdown("---")
    st.subheader("🔍 Key Dataset Insights")
    st.markdown(
        """
        ✔ Prepared specifically for a **junkshop environment**  
        ✔ Designed to reduce metal misclassification → higher earnings  
        ✔ Output is reliable on both clean and dirty scrap  
        """
    )
