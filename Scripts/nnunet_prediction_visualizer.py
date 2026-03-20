# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:01:59 2025

@author: Damian
"""
# === IMPORT === # 
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm

# === File Paths ===
#path to image subfolder
#example: image_dir = r".../nnUNet_Project/Prediction_NIfTI_Images/{FOLDER_NAME}/images"
image_dir = r"    "

#path to predictions folder
#example: pred-dir = r".../nnUNet_Project/Prediction_NIfTI_Images/{FOLDER_NAME}/predictions"
pred_dir = r"   "
mapping_file = os.path.join(image_dir, "frame_mapping.txt")  # Contains frame headers

# === Load Frame Header Mapping ===
frame_title_map = {}
if os.path.exists(mapping_file):
   with open(mapping_file, "r") as f:
    for line in f:
        if line.strip():
            fname, header = line.strip().split(",", 1)
            base_id = fname.replace(".nii.gz", "")
            frame_title_map[base_id] = header

# === Label Color Map ===
label_map = {
    1: ("Primary Protons", "#00BFFF"),
    2: ("Low Energy Electrons", "#FF3030"),
    3: ("Gammas", "#FFD700"),
    4: ("Scattered Protons", "#9370DB"),
    5: ("Heavy Particles", "#7CFC00"),
    6: ("δ Electrons", "#FF8C00"),
}



def plot_image_and_prediction(image_path, prediction_path, title):

    image_nii = nib.load(image_path)
    pred_nii = nib.load(prediction_path)

    image = nib.as_closest_canonical(image_nii)
    prediction = nib.as_closest_canonical(pred_nii)

    image_data = image.get_fdata()
    pred_data = prediction.get_fdata()

    if image_data.ndim == 3:
        mid_slice = image_data.shape[2] // 2
        image_slice = image_data[:, :, mid_slice]
        pred_slice = pred_data[:, :, mid_slice]
    else:
        image_slice = image_data
        pred_slice = pred_data

    # === Create Overlay Mask ===
    rgb_segmentation = np.zeros((*pred_slice.shape, 3), dtype=np.uint8)
    for label_value, (label_name, color) in label_map.items():
        mask = pred_slice == label_value
        rgb = np.array(mcolors.to_rgb(color)) * 255
        rgb_segmentation[mask] = rgb.astype(np.uint8)

    # === Grayscale normalization ===
    vmin = image_slice[image_slice > 0].min() if np.any(image_slice > 0) else 1
    vmax = image_slice.max() if np.any(image_slice > 0) else 2
    cmap = plt.get_cmap('gray').copy()

    cmap.set_under('black')
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle(title, fontsize=14)

    # --- Input image ---
    axes[0].imshow(image_slice, cmap='gray', norm=norm, origin='lower')
    axes[0].set_title("Input Image", color='black')

    # --- Prediction overlay ---

    axes[1].imshow(image_slice, cmap='gray', norm=norm, origin='lower')
    axes[1].imshow(rgb_segmentation, alpha=0.8, origin='lower')

    axes[1].set_title("Prediction Overlay", color='black')

    # --- Axis formatting ---
    for ax in axes:
        ax.set_facecolor('black')
        ax.set_xticks(np.arange(0, image_slice.shape[1]+1, 20))
        ax.set_yticks(np.arange(0, image_slice.shape[0]+1, 20))
        ax.set_xlim(0, image_slice.shape[1])
        ax.set_ylim(0, image_slice.shape[0])
        ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.3, alpha=0.6)
        ax.tick_params(colors='black')
        ax.set_xlabel("X Coordinate (px)", color='black')
        ax.set_ylabel("Y Coordinate (px)", color='black')

    # --- Legend ---
    used_labels = np.unique(pred_slice)
    handles = []
    for label_value, (label_name, color) in label_map.items():
        if label_value in used_labels:
            handles.append(Patch(color=color, label=label_name))
    if handles:
        legend = axes[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                                fontsize=12, markerscale=2)
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts():
            text.set_color("white")

    plt.tight_layout()
    plt.show()

# === Iterate Over Files ===
image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith("_0000.nii.gz")])

for filename in image_filenames[:100]:
    case_id = filename.replace("_0000.nii.gz", "")
    image_path = os.path.join(image_dir, filename)
    prediction_path = os.path.join(pred_dir, f"{case_id}.nii.gz")

    if os.path.exists(prediction_path):
        title = frame_title_map.get(case_id, f"Frame: {case_id}")
        print(f"🔍 Visualizing: {title}")
        plot_image_and_prediction(image_path, prediction_path, title)