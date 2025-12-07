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
    st.write(
        "These performance visualizations are based on the model's validation/testing results. "
        "They help determine how reliably the system can classify scrap metal versus trash when used "
        "by users in Orlan’s Junkshop."
    )
    
    # ---------------------------------------------------------
    # Confusion Matrix (Row 1)
    # ---------------------------------------------------------
    st.image("confusion_matrix.png", caption="Confusion Matrix")
    st.markdown(
        """
        **Confusion Matrix Interpretation**  
        • The model correctly identifies **912 metal items** and **236 trash items**, showing strong reliability.  
        • A small number of background areas are misclassified as scrap (**67 cases**), but this does not heavily affect performance.  
        • Mislabeling between metal and trash is **very low** (only **5 metal → trash** and **11 trash → metal**).  
        
        📌 This means the Scrap Checker is highly capable of preventing recyclable metal from being misclassified as trash — leading to reduced losses and improved recycling efficiency.
        """
    )
    
    st.markdown("---")

    # ---------------------------------------------------------
    # F1 Confidence Curve (Row 2)
    # ---------------------------------------------------------
    st.image("f1_confidence_curve.png", caption="F1-Confidence Curve")
    st.markdown(
        """
        **F1-Confidence Curve Interpretation**  
        • The peak performance occurs around **0.455 confidence threshold**, where F1 ≈ **0.93**.  
        • The curve remains high and stable across a wide confidence range (~0.3–0.85), showing excellent **balance between precision and recall**.  
        • This confirms that the model consistently performs well even if object shapes, lighting, or camera angles vary.  
        
        📌 For practical use, a default confidence of **0.50** ensures accurate detection while still finding metals that are partially covered, dirty, or deformed.
        """
    )
