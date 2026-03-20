#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:16:03 2025

@author: liyong
"""

import os
import numpy as np
import nibabel as nib
import json

# === Input files (merged with multiple particle types inside) ===
# set merged_clog_files = your training .clog file
#Example: merged_clog_files = [".../nnUNet_Project/nnUNet_Data/Training_Data/{FILENAME}.clog"]
merged_clog_files = [
    "   "
]

# === Output Directories ===
# set output_base = your dataset folder
# Example: output_base = ".../nnUNet_Project/nnUNet_Data/nnUNet_raw/Dataset###_name"
output_base = "   "
imagesTr_dir = os.path.join(output_base, 'imagesTr')
labelsTr_dir = os.path.join(output_base, 'labelsTr')
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)




# === Particle ID Mapping ===
class_to_index = {
    0: "background",
    1: "Primary Protons",
    2: "Low energy Electrons",
    3: "Photons",
    4: "Scattered Protons",
    5: "Heavy Particles",
    6: "δ Electrons"
}

valid_labels = set(class_to_index.keys())


# === Parse the merged .clog files ===
frames_points = {}

for file_idx, filepath in enumerate(merged_clog_files):
    with open(filepath, 'r') as f:
        current_frame = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Frame"):
                current_frame = int(line.split()[1])
                key = f"{file_idx}__Frame{current_frame}"
                frames_points.setdefault(key, [])
            else:
                coords = line.replace('[', '').split(']')
                for coord in coords:
                    coord = coord.strip(", ")
                    if coord:
                        try:
                            x, y, intensity, label = map(int, coord.split(','))
                            if intensity == 16383:
                                continue  # Skip overflow
                            if label not in valid_labels:
                                continue  # Skip unknown label
                            frames_points[key].append((x, y, intensity, label))
                        except ValueError:
                            continue  # skip malformed lines

# === Determine image size ===
max_x = max_y = 0
for pts in frames_points.values():
    for (x, y, _, _) in pts:
        max_x = max(max_x, x)
        max_y = max(max_y, y)
img_width, img_height = max_x + 1, max_y + 1

# === Save frames ===
case_id = 0
for key, points in frames_points.items():
    if not points:
        continue

    image = np.zeros((img_height, img_width), dtype=np.uint8)
    label = np.zeros((img_height, img_width), dtype=np.uint8)

    for (x, y, intensity, particle_label) in points:
        image[y, x] = min(255, intensity)
        label[y, x] = particle_label

    affine = np.eye(4)
    case_str = f'PC_{case_id:04d}'
    nib.save(nib.Nifti1Image(image, affine), os.path.join(imagesTr_dir, f"{case_str}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(label.astype(np.uint8), affine), os.path.join(labelsTr_dir, f"{case_str}.nii.gz"))
    print(f"✅ Saved {case_str} from {key} — labels: {np.unique(label)}")
    case_id += 1

print(f"\n🎉 Finished generating {case_id} training samples.")

# === Create dataset.json ===
num_training = len([f for f in os.listdir(imagesTr_dir) if f.endswith("_0000.nii.gz")])

dataset_json = {
    "channel_names": {
        "0": "grayscale"
    },
    "labels": {name: idx for idx, name in class_to_index.items()},
    "numTraining": num_training,
    "file_ending": ".nii.gz"
}

json_path = os.path.join(output_base, "dataset.json")
with open(json_path, "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"✅ dataset.json written to: {json_path}")
print(f"📦 Training samples counted: {num_training}")