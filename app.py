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
# 0. GLOBAL SIM STATE (BIN COUNTS ACROSS RUNS)
# ---------------------------------------------------------
if "good_bin" not in st.session_state:
    st.session_state["good_bin"] = 0
if "bad_bin" not in st.session_state:
    st.session_state["bad_bin"] = 0

# ---------------------------------------------------------
# 1. LOAD MODEL (With Caching)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO('best.pt')   # <-- your trained model

model = load_model()

# ---------------------------------------------------------
# 2. LOAD ENVIRONMENT IMAGES FOR SIMULATION
# ---------------------------------------------------------
@st.cache_resource
def load_env_images():
    try:
        floor = plt.imread('floor.jpg')
        belt = plt.imread('conveyor_belt.png')  # use your PNG belt
        return floor, belt
    except FileNotFoundError:
        return None, None

# ---------------------------------------------------------
# 3. PIXEL PERCENTAGE CALCULATOR
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 4. CONVEYOR SIMULATION (USES YOLO RESULT)
# ---------------------------------------------------------
def run_conveyor_sim(item_image_np, metal_pct, is_good, pass_threshold, sim_placeholder):
    """
    item_image_np : numpy array of the uploaded image
    metal_pct     : float, metal percentage
    is_good       : bool, True if PASS
    pass_threshold: current slider threshold (for HUD color)
    sim_placeholder: st.empty() in the right column
    """
    img_floor, img_belt = load_env_images()
    if img_floor is None or img_belt is None:
        sim_placeholder.error(
            "Environment images not found. Please add 'floor.jpg' and 'conveyor_belt.png' "
            "in the same folder as this app."
        )
        return

    # --- CONFIGURATION ---
    PUSHER_STROKE = 20
    BIN_CAPACITY = 5        # physical capacity in GOOD bin
    CONVEYOR_SPEED = 8.0

    # --- SIMULATION STATE ---
    # Start bin counts from stored session values so they persist across images
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

    # --- MATPLOTLIB FIGURE SETUP ---
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#222222')
    ax.set_facecolor('#222222')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.axis('off')

    # Inset LIVE FEED
    ax_feed = fig.add_axes([0.05, 0.60, 0.25, 0.25])
    ax_feed.set_facecolor('black')
    ax_feed.set_xticks([])
    ax_feed.set_yticks([])

    # --- VISUAL HELPERS (CLOSURES BOUND TO ax / ax_feed / state) ---
    def draw_image_main(img, x, y, width, height, z):
        ax.imshow(img, extent=[x, x + width, y, y + height], zorder=z)

    def draw_detailed_bin(x, y, width, height, color, label, text_color='white'):
        z_bin = 4
        body_color = matplotlib.colors.to_rgb(color)
        body_color = [c * 0.8 for c in body_color]
        ax.add_patch(
            patches.Rectangle(
                (x, y), width, height,
                facecolor=body_color,
                edgecolor='black',
                linewidth=2,
                zorder=z_bin
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (x - 1, y + height - 3), width + 2, 4,
                facecolor=color,
                edgecolor='black',
                linewidth=2,
                zorder=z_bin + 1
            )
        )
        rib_color = 'black'
        rib_width = 0.6
        for i in range(1, 4):
            rib_x = x + (width / 4) * i
            ax.add_patch(
                patches.Rectangle(
                    (rib_x - rib_width / 2, y + 1),
                    rib_width, height - 5,
                    facecolor=rib_color,
                    edgecolor=None,
                    zorder=z_bin + 1,
                    alpha=0.3
                )
            )
        ax.text(
            x + width / 2, y + height / 2,
            label,
            ha='center', va='center',
            color=text_color,
            weight='bold',
            fontsize=9,
            zorder=z_bin + 2
        )

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
            txt = ax.text(
                item.x, item.y + 12,
                status_text,
                color=stamp_color,
                fontsize=16,
                weight='bold',
                ha='center',
                zorder=z + 5
            )
            txt.set_path_effects(
                [path_effects.withStroke(linewidth=3, foreground='black')]
            )

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

        # Pneumatics
        ax.add_patch(
            patches.Rectangle((72, 55), 6, 10,
                              facecolor='#444444',
                              edgecolor='black',
                              zorder=1)
        )
        piston_extension = (state["pusher_position"] / PUSHER_STROKE) * 18
        ax.add_patch(
            patches.Rectangle((74.5, 55 - piston_extension), 1, piston_extension,
                              facecolor='#c0c0c0',
                              zorder=1)
        )
        head_y = 55 - piston_extension - 3
        ax.add_patch(
            patches.Rectangle((71, head_y), 8, 3,
                              facecolor='#ff0000',
                              edgecolor='black',
                              zorder=4)
        )

        # Camera Stand
        ax.add_patch(
            patches.Rectangle((72, 35), 6, 4,
                              facecolor='#222',
                              edgecolor='cyan',
                              zorder=6)
        )
        ax.plot([75, 75], [39, 50], color='grey', lw=2, zorder=1)

        # Bins
        draw_detailed_bin(
            85, 15, 18, 22, '#228B22',
            f"GOOD:\n{state['good_bin_count']}/{BIN_CAPACITY}"
        )
        draw_detailed_bin(
            70, 5, 18, 22, '#DC143C',
            f"REJECTS:\n{state['bad_bin_count']}"
        )

        if state["forklift_active"]:
            bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="red", ec="black", lw=2)
            ax.text(
                50, 20,
                "BIN FULL: FORKLIFT DISPATCHED",
                ha='center',
                color='white',
                weight='bold',
                fontsize=12,
                bbox=bbox_props,
                zorder=25
            )

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
        ax_feed.text(66, 52, "⚫ LIVE SENSOR FEED", color='red', fontsize=8, weight='bold')

        # Spawn the single YOLO item once at start
        if len(state["items"]) == 0 and state["conveyor_moving"] and frame == 0:
            state["items"].append(Item(item_image_np, metal_pct, is_good))

        # Movement
        if state["conveyor_moving"]:
            for item in state["items"]:
                if item.status == "new":
                    item.x += CONVEYOR_SPEED
                    if item.x >= 75:
                        item.x = 75
                        state["conveyor_moving"] = False
                        item.status = "scanning"
                        state["scan_timer"] = 0

        # Processing
        items_to_remove = []

        for item in state["items"][
