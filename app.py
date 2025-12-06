import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ---------------------------------------------------------
# 1. LOAD MODEL (With Caching)
# ---------------------------------------------------------
# We use @st.cache_resource so the model loads only ONCE, 
# not every time the user clicks a button.
@st.cache_resource
def load_model():
    # Update this path to your actual best.pt file
    return YOLO('best.pt')

model = load_model()

# ---------------------------------------------------------
# 2. UI LAYOUT
# ---------------------------------------------------------
st.title("🔩 Scrap & Metal Detector")
st.write("Upload an image or use your camera to detect scrap.")

# Sidebar for settings
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# ---------------------------------------------------------
# 3. IMAGE INPUT (The "Processing" Step)
# ---------------------------------------------------------
# Option 1: File Upload
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Option 2: Camera Input (Optional)
camera_file = st.camera_input("Or take a picture")

# Logic to prioritize inputs (Upload > Camera)
image_source = uploaded_file if uploaded_file else camera_file

# ---------------------------------------------------------
# 4. PREDICTION & DISPLAY
# ---------------------------------------------------------
if image_source:
    # STEP A: Convert Streamlit Buffer -> PIL Image
    # This is the only "processing" you need.
    image = Image.open(image_source)

    # Display original image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # STEP B: Run YOLO Prediction
    if st.button("Detect Objects"):
        with st.spinner("Analyzing..."):
            # Run inference
            results = model.predict(image, conf=conf_threshold)

            # STEP C: Show Results
            # Plot the results on the image (returns a numpy array)
            res_plotted = results[0].plot()
            
            # Display the annotated image
            st.image(res_plotted, caption="Detected Objects", use_container_width=True)

            # Optional: Show counts
            # Get the boxes from the result
            boxes = results[0].boxes
            st.success(f"Found {len(boxes)} objects!")