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
# 2. UI LAYOUT
# ---------------------------------------------------------
st.title("Orlan's Junkshop Scrap Cleaner")
st.write("Upload an image or use your camera to detect scrap.")

# Sidebar for settings
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
pass_threshold = st.sidebar.slider("Metal % Required for PASS", 0, 100, 95, 1)

# ---------------------------------------------------------
# 3. IMAGE INPUT
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
camera_file = st.camera_input("Or take a picture")
image_source = uploaded_file if uploaded_file else camera_file

# ---------------------------------------------------------
# 4. PREDICTION & DISPLAY
# ---------------------------------------------------------
if image_source:
    image = Image.open(image_source)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect Objects"):
        with st.spinner("Analyzing..."):
            # Run inference
            results = model.predict(image, conf=conf_threshold)
            res_plotted = results[0].plot()
            
            st.image(res_plotted, caption="Detected Objects", use_container_width=True)
            
            # ---------------------------------------------------------
            # CALCULATE METAL PERCENTAGE
            # ---------------------------------------------------------
            boxes = results[0].boxes
            total_objects = len(boxes)
            
            if total_objects == 0:
                st.warning("⚠️ No objects detected. Try lowering confidence threshold.")
            else:
                # Count metal objects
                metal_count = 0
                trash_count = 0
                class_names = model.names  # {0: 'metal', 1: 'trash'} or similar
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id].lower()
                    
                    if 'metal' in class_name:
                        metal_count += 1
                    else:
                        trash_count += 1
                
                # Calculate percentage
                metal_percentage = (metal_count / total_objects) * 100
                
                # ---------------------------------------------------------
                # DISPLAY RESULTS
                # ---------------------------------------------------------
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Objects", total_objects)
                col2.metric("🔩 Metal", f"{metal_count} ({metal_percentage:.1f}%)")
                col3.metric("🗑️ Trash", f"{trash_count} ({100-metal_percentage:.1f}%)")
                
                # ---------------------------------------------------------
                # PASS/FAIL LOGIC
                # ---------------------------------------------------------
                st.markdown("---")
                if metal_percentage >= pass_threshold:
                    st.success(f"✅ **PASS** - Metal content: {metal_percentage:.1f}% (Required: ≥{pass_threshold}%)")
                else:
                    st.error(f"❌ **FAIL** - Metal content: {metal_percentage:.1f}% (Required: ≥{pass_threshold}%)")



                # ---------------------------------------------------------
                # MODEL METRICS SECTION
                # ---------------------------------------------------------
                st.markdown("## 📊 Model Evaluation Metrics")
                st.write("Below are evaluation results of the trained model on the test dataset.")

                col_metrics_1, col_metrics_2 = st.columns(2)

                with col_metrics_1:
                    st.image("confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
                    st.caption("Shows how well the model correctly classified metal vs trash. Darker squares indicate stronger accuracy.")

                with col_metrics_2:
                    st.image("f1_confidence_curve.png", caption="F1-Confidence Curve", use_container_width=True)
                    st.caption("Displays how the F1-Score changes relative to confidence thresholds. Optimal performance occurs near ~0.455.")

