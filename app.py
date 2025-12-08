import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import time

# ==============================
# 0️⃣ GLOBAL STATE (BIN COUNTERS)
# ==============================
if "good_bin" not in st.session_state:
    st.session_state["good_bin"] = 0

if "bad_bin" not in st.session_state:
    st.session_state["bad_bin"] = 0

# ==============================
# 1️⃣ MODEL LOADING
# ==============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ==============================
# 2️⃣ SIMULATION BACKGROUND IMAGES
# ==============================
@st.cache_resource
def load_env_images():
    try:
        floor = plt.imread("floor.jpg")
        belt = plt.imread("conveyor_belt.png")
        return floor, belt
    except:
        return None, None

# ==============================
# 3️⃣ METAL PIXEL CALCULATION
# ==============================
def compute_pixel_percentage(results, model):
    r = results[0]

    if r.masks is None:
        return 0, 0, False

    class_names = model.names
    total_metal = 0
    total_nonmetal = 0

    for mask, cls in zip(r.masks.data, r.boxes.cls):
        mask_np = mask.cpu().numpy()
        pixels = np.sum(mask_np)
        label = class_names[int(cls)]
        if "metal" in label.lower():
            total_metal += pixels
        else:
            total_nonmetal += pixels

    total = total_metal + total_nonmetal
    if total == 0:
        return 0, 0, False

    metal_pct = (total_metal / total) * 100
    return metal_pct, 100 - metal_pct, True

# ==============================
# 4️⃣ CONVEYOR + SORTING SIM
# ==============================
def run_conveyor_sim(item_image_np, metal_pct, is_good, pass_threshold, sim_placeholder):

    img_floor, img_belt = load_env_images()
    if img_floor is None:
        sim_placeholder.error("Missing floor.jpg or conveyor_belt.png")
        return

    BIN_CAPACITY = 5
    SPEED = 8
    PUSHER_MAX = 20

    state = {
        "conveyor_moving": True,
        "items": [],
        "pusher": 0,
        "scan_timer": 0,
        "good": st.session_state["good_bin"],
        "bad": st.session_state["bad_bin"],
        "forklift_active": False
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax_feed = fig.add_axes([0.05, 0.60, 0.25, 0.25])
    ax_feed.set_xticks([])
    ax_feed.set_yticks([])
    ax_feed.set_facecolor("black")

    class Item:
        def __init__(self, img, pct, flag):
            self.x = 0
            self.y = 42
            self.status = "new"
            self.image = img
            self.metal_pct = round(float(pct), 2)
            self.is_good = flag

    if len(state["items"]) == 0:
        state["items"].append(Item(item_image_np, metal_pct, is_good))

    def draw_bin(x, y, w, h, color, text):
        body = [c * 0.7 for c in matplotlib.colors.to_rgb(color)]
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=body, edgecolor="black"))
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=9, color="white",
                weight="bold")

    for frame in range(120):
        ax.clear()
        ax.axis("off")

        # Background
        ax.imshow(img_floor, extent=[0, 100, 0, 100])
        ax.imshow(img_belt, extent=[0, 80, 40, 52])

        # LIVE FEED
        ax_feed.clear()
        ax_feed.set_facecolor("black")
        ax_feed.imshow(img_belt, extent=[60, 90, 40, 52])
        ax_feed.text(66, 52, "⚫ FEED", color="red")

        remove = []

        for item in state["items"]:
            # Draw item
            ax.imshow(item.image, extent=[item.x - 7, item.x + 7, item.y - 7, item.y + 7], zorder=3)

            # Move + scan logic
            if state["conveyor_moving"] and item.status == "new":
                item.x += SPEED
                if item.x >= 75:
                    item.x = 75
                    state["conveyor_moving"] = False
                    item.status = "scanning"
                    state["scan_timer"] = 0

            if item.status == "scanning":
                ax_feed.imshow(item.image, extent=[67, 83, 38, 54])
                color = "lime" if item.metal_pct >= pass_threshold else "red"
                ax_feed.text(66, 37, f"{item.metal_pct:.1f}%", color=color)

                state["scan_timer"] += 1
                if state["scan_timer"] > 5:
                    item.status = "decision"

            elif item.status == "decision":
                if item.is_good:  # DROP left
                    item.x += 8
                    if item.x > 80:
                        item.y -= 8
                    if item.y < 28:
                        state["good"] += 1
                        remove.append(item)
                        state["conveyor_moving"] = True
                else:  # PUSH right
                    if state["pusher"] < PUSHER_MAX:
                        state["pusher"] += 10
                        item.y -= 8
                        item.x += 2
                    else:
                        state["bad"] += 1
                        remove.append(item)
                        state["pusher"] = 0
                        state["conveyor_moving"] = True

        for i in remove:
            state["items"].remove(i)

        # Bins
        draw_bin(85, 15, 18, 22, "green", f"GOOD\n{state['good']}/{BIN_CAPACITY}")
        draw_bin(70, 5, 18, 22, "red", f"BAD\n{state['bad']}")

        sim_placeholder.pyplot(fig)
        time.sleep(0.03)

    st.session_state["good_bin"] = state["good"]
    st.session_state["bad_bin"] = state["bad"]

# ==============================
# 5️⃣ USER INTERFACE
# ==============================
st.title("Orlan's Junkshop Scrap Cleaner")
tab1, tab2, tab3 = st.tabs(["🔍 Sort + Sim", "📂 Dataset", "📈 Performance"])

with tab1:
    conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)
    pass_threshold = st.sidebar.slider("PASS % Metal", 0, 100, 95)

    col_left, col_right = st.columns(2)

    with col_right:
        sim_placeholder = st.empty()
        st.caption("Live Sorting Simulation")

    with col_left:
        upload = st.file_uploader("Upload", type=["jpg", "png"])
        camera = st.camera_input("Camera")
        img_src = upload if upload else camera

        if img_src:
            img = Image.open(img_src).convert("RGB")
            st.image(img, caption="Uploaded")

            if st.button("Detect & Sort"):
                with st.spinner("Analyzing..."):
                    res = model.predict(img, conf=conf)
                    st.image(res[0].plot(), caption="Detection")

                    metal_pct, nonmetal_pct, valid = compute_pixel_percentage(res, model)

                    if not valid:
                        st.warning("No segmentation detected")
                    else:
                        if metal_pct >= pass_threshold:
                            st.success(f"PASS — {metal_pct:.1f}% metal")
                            is_good = True
                        else:
                            st.error(f"FAIL — {metal_pct:.1f}% metal")
                            is_good = False

                        run_conveyor_sim(np.array(img),
                                         metal_pct,
                                         is_good,
                                         pass_threshold,
                                         sim_placeholder)

with tab2:
    st.write("Dataset Overview Coming Soon...")

with tab3:
    st.write("Model Performance Coming Soon...")
