import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import time


#GLOBAL SIM STATE 

if "good_bin" not in st.session_state:
    st.session_state["good_bin"] = 0
if "bad_bin" not in st.session_state:
    st.session_state["bad_bin"] = 0


#LOAD MODEL

@st.cache_resource
def load_model():
    # Make sure best.pt is in the same folder
    return YOLO('best.pt')

model = load_model()


#LOAD ENVIRONMENT IMAGES

@st.cache_resource
def load_env_images():
    try:
        floor = plt.imread('floor.jpg')
        belt = plt.imread('conveyor_belt.png') 
        return floor, belt
    except FileNotFoundError:
        return None, None


#PIXEL PERCENTAGE 

def compute_pixel_percentage(results, model):
    r = results[0]
    if r.masks is None:
        return 0, 0, False

    total_metal_pixels = 0
    total_nonmetal_pixels = 0
    class_names = model.names

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


#CONVEYOR SIMULATION

def run_conveyor_sim(item_image_np, metal_pct, is_good, pass_threshold, sim_placeholder):
    img_floor, img_belt = load_env_images()
    if img_floor is None or img_belt is None:
        sim_placeholder.error("Environment images not found. Ensure 'floor.jpg' and 'conveyor_belt.png' are present.")
        return

   
    PUSHER_STROKE = 20
    BIN_CAPACITY = 5  # <-- SET BACK TO 5 FOR FORKLIFT LOGIC
    CONVEYOR_SPEED = 8.0


    state = {
        "conveyor_moving": True,
        "items": [],
        "good_bin_count": st.session_state["good_bin"],
        "bad_bin_count": st.session_state["bad_bin"],
        "current_item_processing": None,
        "pusher_position": 0,
        "forklift_active": False,
        "scan_timer": 0,
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#222222')
    ax.set_facecolor('#222222')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.axis('off')

    ax_feed = fig.add_axes([0.05, 0.60, 0.25, 0.25])
    ax_feed.set_facecolor('black')
    ax_feed.set_xticks([])
    ax_feed.set_yticks([])


    def draw_image_main(img, x, y, width, height, z):
        ax.imshow(img, extent=[x, x + width, y, y + height], zorder=z)

    def draw_detailed_bin(x, y, width, height, color, label, text_color='white'):
        z_bin = 4
        body_color = matplotlib.colors.to_rgb(color)
        body_color = [c * 0.8 for c in body_color]
        ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=body_color, edgecolor='black', linewidth=2, zorder=z_bin))
        ax.add_patch(patches.Rectangle((x - 1, y + height - 3), width + 2, 4, facecolor=color, edgecolor='black', linewidth=2, zorder=z_bin + 1))
        
        rib_color = 'black'
        rib_width = 0.6
        for i in range(1, 4):
            rib_x = x + (width / 4) * i
            ax.add_patch(patches.Rectangle((rib_x - rib_width / 2, y + 1), rib_width, height - 5, facecolor=rib_color, zorder=z_bin + 1, alpha=0.3))
        
        ax.text(x + width / 2, y + height / 2, label, ha='center', va='center', color=text_color, weight='bold', fontsize=9, zorder=z_bin + 2)

    def draw_hud_main(item):
        box_w, box_h = 16, 16
        box_x, box_y = item.x - 8, item.y - 8
        color = 'cyan'
        lw = 2
        z = 20
        ax.plot([box_x, box_x + 4], [box_y + box_h, box_y + box_h], color=color, lw=lw, zorder=z)
        ax.plot([box_x, box_x], [box_y + box_h, box_y + box_h - 4], color=color, lw=lw, zorder=z)
        ax.plot([box_x + box_w, box_x + box_w - 4], [box_y, box_y], color=color, lw=lw, zorder=z)
        ax.plot([box_x + box_w, box_x + box_w], [box_y, box_y + 4], color=color, lw=lw, zorder=z)

        if state['scan_timer'] > 2:
            status_text = "PASS" if item.is_good else "REJECT"
            stamp_color = '#00ff00' if item.is_good else '#ff0000'
            txt = ax.text(item.x, item.y + 12, status_text, color=stamp_color, fontsize=16, weight='bold', ha='center', zorder=z + 5)
            txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    class Item:
        def __init__(self, img, metal_pct, is_good_flag):
            self.x = 0
            self.y = 42
            self.status = "new"
            self.image = img
            self.metal_content = round(float(metal_pct), 2)
            self.is_good = bool(is_good_flag)

    def draw_layout_main():
        draw_image_main(img_floor, 0, 0, 100, 100, z=0)
        draw_image_main(img_belt, 0, 40, 80, 12, z=2)

        #Piston
        ax.add_patch(patches.Rectangle((72, 55), 6, 10, facecolor='#444444', edgecolor='black', zorder=1))
        piston_extension = (state["pusher_position"] / PUSHER_STROKE) * 18
        ax.add_patch(patches.Rectangle((74.5, 55 - piston_extension), 1, piston_extension, facecolor='#c0c0c0', zorder=1))
        head_y = 55 - piston_extension - 3
        ax.add_patch(patches.Rectangle((71, head_y), 8, 3, facecolor='#ff0000', edgecolor='black', zorder=4))
        ax.add_patch(patches.Rectangle((72, 35), 6, 4, facecolor='#222', edgecolor='cyan', zorder=6))
        ax.plot([75, 75], [39, 50], color='grey', lw=2, zorder=1)

        #Bins
        draw_detailed_bin(85, 15, 18, 22, '#228B22', f"GOOD:\n{state['good_bin_count']}/{BIN_CAPACITY}")
        draw_detailed_bin(70, 5, 18, 22, '#DC143C', f"REJECTS:\n{state['bad_bin_count']}")

        #FORKLIFT
        if state["forklift_active"]:
            bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="red", ec="black", lw=2)
            ax.text(50, 20, "BIN FULL: FORKLIFT DISPATCHED\n(System Paused)", ha='center', color='white', weight='bold', fontsize=10, bbox=bbox_props, zorder=25)

    def update_frame(frame):
        ax.clear()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

        ax_feed.clear()
        ax_feed.set_facecolor('black')
        ax_feed.set_xlim(65, 85)
        ax_feed.set_ylim(35, 55)
        ax_feed.set_xticks([])
        ax_feed.set_yticks([])
        ax_feed.imshow(img_belt, extent=[60, 90, 40, 52])
        ax_feed.text(66, 52, "LIVE SENSOR FEED", color='red', fontsize=8, weight='bold')


        if len(state["items"]) == 0 and state["conveyor_moving"] and frame == 0:
            state["items"].append(Item(item_image_np, metal_pct, is_good))



        if state["good_bin_count"] >= BIN_CAPACITY:
            state["forklift_active"] = True
            state["conveyor_moving"] = False
            


            if np.random.rand() < 0.05:
                state["good_bin_count"] = 0 #Empty
                st.session_state["good_bin"] = 0 #Update
                state["forklift_active"] = False
                state["conveyor_moving"] = True #Restart


        if state["conveyor_moving"]:
            for item in state["items"]:
                if item.status == "new":
                    item.x += CONVEYOR_SPEED
                    if item.x >= 75:
                        item.x = 75
                        state["conveyor_moving"] = False
                        item.status = "scanning"
                        state["scan_timer"] = 0

        items_to_remove = []
        for item in state["items"]:
            draw_image_main(item.image, item.x - 7, item.y - 7, 14, 14, z=3)
            if 60 < item.x < 90:
                ax_feed.imshow(item.image, extent=[item.x - 7, item.x + 7, item.y - 7, item.y + 7])

            if item.status == "scanning":
                draw_hud_main(item)
                perc_color = '#00ff00' if item.metal_content >= pass_threshold else '#ff0000'
                ax_feed.text(66, 37, f"READING: {item.metal_content:.2f}%", color=perc_color, weight='bold', fontsize=9)
                state["scan_timer"] += 1
                if state["scan_timer"] > 5:
                    item.status = "decision"

            elif item.status == "decision":
                draw_hud_main(item)
                if item.is_good:
                    item.x += 8
                    if item.x > 80: item.y -= 8

                    if item.y < 28 and item.x > 82:
                        state["good_bin_count"] += 1
                        st.session_state["good_bin"] = state["good_bin_count"] 
                        items_to_remove.append(item)
                        state["conveyor_moving"] = True
                else:
                    if state["pusher_position"] < PUSHER_STROKE:
                        state["pusher_position"] += 10
                        item.y -= 8
                        item.x += 1
                    else:
                        state["bad_bin_count"] += 1
                        st.session_state["bad_bin"] = state["bad_bin_count"]
                        items_to_remove.append(item)
                        state["pusher_position"] = 0
                        state["conveyor_moving"] = True

        for i in items_to_remove:
            state["items"].remove(i)

        draw_layout_main()

    # Run Animation
    total_frames = 120
    for f in range(total_frames):
        update_frame(f)
        sim_placeholder.pyplot(fig)
        time.sleep(0.03)


#UI

st.title("Orlan's Junkshop Scrap Cleaner")

#SIDEBAR
st.sidebar.header("1. Input Data")
uploaded_file = st.sidebar.file_uploader("Upload Scrap Image", type=['jpg', 'jpeg', 'png'])
st.sidebar.markdown("**OR**")
camera_file = st.sidebar.camera_input("Take a Picture")

image_source = uploaded_file if uploaded_file else camera_file

st.sidebar.markdown("---")
st.sidebar.header("2. Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
pass_threshold = st.sidebar.slider("Metal % Required for PASS", 0, 100, 95, 1)

st.sidebar.markdown("---")
st.sidebar.header("3. Controls")
run_btn = st.sidebar.button("🚀 Detect & Simulate", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("### Bin Status")
if st.sidebar.button("🗑️ Reset Bin Counters"):
    st.session_state["good_bin"] = 0
    st.session_state["bad_bin"] = 0
    st.sidebar.success("Counters reset!")
    st.rerun() 

st.sidebar.metric("Total Good", st.session_state["good_bin"])
st.sidebar.metric("Total Rejected", st.session_state["bad_bin"])

#MAIN
tab1, tab2, tab3 = st.tabs(["🔍 Results & Simulation", "🗂 Dataset Overview", "📊 Model Performance"])

with tab1:
    if not image_source:
        st.info("👈 Please upload an image or take a photo in the Sidebar to begin.")
    else:
        st.write("Image Loaded. Ready to process.")
        
        if run_btn:
            image = Image.open(image_source).convert("RGB")
            
            col_res, col_sim = st.columns([1, 1])
            
            with col_res:
                st.subheader("Detection Results")
                with st.spinner("Running YOLOv8..."):
                    results = model.predict(image, conf=conf_threshold)
                    res_plotted = results[0].plot()
                    st.image(res_plotted, caption="Detected Objects", use_container_width=True)
                    
                    metal_pct, nonmetal_pct, valid = compute_pixel_percentage(results, model)

                    if not valid:
                        st.warning("⚠️ No objects detected.")
                        sim_placeholder = None
                    else:
                        c1, c2 = st.columns(2)
                        c1.metric("🔩 Metal %", f"{metal_pct:.2f}%")
                        c2.metric("🗑️ Non-metal %", f"{nonmetal_pct:.2f}%")
                        
                        is_good = metal_pct >= pass_threshold
                        if is_good:
                            st.success(f"✅ PASS (≥ {pass_threshold}%)")
                        else:
                            st.error(f"❌ FAIL (< {pass_threshold}%)")

            if valid:
                with col_sim:
                    st.subheader("Process Simulation")
                    sim_placeholder = st.empty()
                    item_image_np = np.array(image)
                    
                    # Run the loop
                    run_conveyor_sim(
                        item_image_np=item_image_np,
                        metal_pct=metal_pct,
                        is_good=is_good,
                        pass_threshold=pass_threshold,
                        sim_placeholder=sim_placeholder
                    )

#DATASET
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



#MODEL PERFORMANCE
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
