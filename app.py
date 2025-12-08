import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import time

# ---------------------------------------------------------
# 1. GLOBAL SIM STATE PERSISTENCE
# ---------------------------------------------------------
if "good_bin" not in st.session_state:
    st.session_state["good_bin"] = 0

if "bad_bin" not in st.session_state:
    st.session_state["bad_bin"] = 0

if "sim_state" not in st.session_state:
    st.session_state["sim_state"] = {
        "conveyor_moving": True,
        "items": [],
        "pusher_position": 0,
        "forklift_active": False,
        "scan_timer": 0,
    }

# ---------------------------------------------------------
# 2. LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# ---------------------------------------------------------
# 3. LOAD ENVIRONMENT IMAGES
# ---------------------------------------------------------
@st.cache_resource
def load_env_images():
    try:
        floor = plt.imread('floor.jpg')
        belt = plt.imread('conveyor_belt.png')
        return floor, belt
    except:
        return None, None

# ---------------------------------------------------------
# 4. METAL PIXEL CALCULATION
# ---------------------------------------------------------
def compute_pixel_percentage(results, model):
    r = results[0]

    if r.masks is None:
        return 0, 0, False

    total_metal = 0
    total_nonmetal = 0
    class_names = model.names

    for mask, cls in zip(r.masks.data, r.boxes.cls):
        mask_np = mask.cpu().numpy()
        pixel_count = np.sum(mask_np)
        label = class_names[int(cls)]

        if "metal" in label.lower():
            total_metal += pixel_count
        else:
            total_nonmetal += pixel_count

    total_pixels = total_metal + total_nonmetal

    if total_pixels == 0:
        return 0, 0, False

    metal_pct = (total_metal / total_pixels) * 100
    return metal_pct, 100 - metal_pct, True

# ---------------------------------------------------------
# 5. CONVEYOR SIMULATION
# ---------------------------------------------------------
def run_conveyor_sim(item_image_np, metal_pct, is_good, pass_threshold, sim_placeholder):
    state = st.session_state["sim_state"]
    img_floor, img_belt = load_env_images()

    if img_floor is None:
        sim_placeholder.error("Missing floor.jpg and conveyor_belt.png")
        return

    BIN_CAPACITY = 5
    PUSHER_STROKE = 20
    SPEED = 8

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax_feed = fig.add_axes([0.05, 0.60, 0.25, 0.25])
    ax_feed.set_xticks([])
    ax_feed.set_yticks([])

    class Item:
        def __init__(self, img, metal_pct, flag):
            self.x = 0
            self.y = 42
            self.status = "new"
            self.image = img
            self.metal_content = round(float(metal_pct), 2)
            self.is_good = bool(flag)

    if len(state["items"]) == 0:
        state["items"].append(Item(item_image_np, metal_pct, is_good))

    total_frames = 100

    for frame in range(total_frames):
        ax.clear()
        ax.axis('off')

        ax.imshow(img_floor, extent=[0, 100, 0, 100])
        ax.imshow(img_belt, extent=[0, 80, 40, 52])

        ax_feed.clear()
        ax_feed.set_facecolor('black')
        ax_feed.imshow(img_belt, extent=[60, 90, 40, 52])

        for item in state["items"]:
            ax.imshow(item.image, extent=[item.x - 7, item.x + 7, item.y - 7, item.y + 7])

            if state["conveyor_moving"] and item.status == "new":
                item.x += SPEED
                if item.x >= 75:
                    state["conveyor_moving"] = False
                    item.status = "scanning"
                    state["scan_timer"] = 0

            if item.status == "scanning":
                ax_feed.imshow(item.image, extent=[67, 83, 38, 54])
                ax_feed.text(66, 52, "⚫ LIVE SENSOR FEED", color='red')

                state["scan_timer"] += 1
                if state["scan_timer"] > 5:
                    item.status = "decision"

            elif item.status == "decision":
                if item.is_good:
                    item.x += 6
                    if item.x > 80:
                        item.y -= 6
                    if item.y < 28:
                        st.session_state["good_bin"] += 1
                        state["items"].remove(item)
                        state["conveyor_moving"] = True
                else:
                    if state["pusher_position"] < PUSHER_STROKE:
                        state["pusher_position"] += 10
                        item.y -= 6
                        item.x += 2
                    else:
                        st.session_state["bad_bin"] += 1
                        state["pusher_position"] = 0
                        state["items"].remove(item)
                        state["conveyor_moving"] = True

        good = st.session_state["good_bin"]
        bad = st.session_state["bad_bin"]

        ax.text(85, 40, f"GOOD:\n{good}/{BIN_CAPACITY}", color="lime", fontsize=10)
        ax.text(85, 30, f"BAD:\n{bad}", color="red", fontsize=10)

        sim_placeholder.pyplot(fig)
        time.sleep(0.02)

# ---------------------------------------------------------
# 6. UI TABS
# ---------------------------------------------------------
st.title("Orlan's Junkshop Scrap Cleaner")
tab1, tab2, tab3 = st.tabs(["🔍 Checker + Conveyor", "📂 Dataset", "📈 Performance"])

with tab1:
    conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)
    pass_threshold = st.sidebar.slider("Pass %", 0, 100, 95)

    col_left, col_right = st.columns(2)

    with col_right:
        sim_placeholder = st.empty()
        st.caption("Live Sorting Simulation")

    with col_left:
        upload = st.file_uploader("Upload...", type=["jpg", "png"])
        camera = st.camera_input("Camera")

        img_src = upload if upload else camera
        if img_src:
            img = Image.open(img_src).convert("RGB")
            st.image(img, caption="Uploaded")

            if st.button("Detect & Sort"):
                with st.spinner("Processing..."):
                    res = model.predict(img, conf=conf_threshold)
                    st.image(res[0].plot(), caption="Detection")

                    metal_pct, nonmetal, ok = compute_pixel_percentage(res, model)
                    if not ok:
                        st.warning("No segmentation detected")
                    else:
                        is_good = metal_pct >= pass_threshold
                        if is_good:
                            st.success(f"PASS {metal_pct:.1f}% metal")
                        else:
                            st.error(f"FAIL {metal_pct:.1f}% metal")

                        run_conveyor_sim(np.array(img), metal_pct, is_good, pass_threshold, sim_placeholder)

with tab2:
    st.write("Dataset Info…")

with tab3:
    st.write("Model Performance…")
