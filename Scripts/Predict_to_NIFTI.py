# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:28:35 2025

@author: Damian
"""

import os
import numpy as np
import nibabel as nib

# === Input Parameters ===
#path to your prediction .clog file
#example: clog_filepath = ".../nnUNet_Project/Prediction_Data/{FILENAME}.clog"
clog_filepath = "    "

#path to output folder with a /images subfolder
#example: output_dir = ".../nnUNet_Project/Prediction_NIfTI_Images/{FOLDER_NAME}/images"
output_dir = "     "

mapping_file = os.path.join(output_dir, "frame_mapping.txt")
os.makedirs(output_dir, exist_ok=True)

# === Load and Parse CLOG File ===
frames_points = []
frame_headers = []
true_frame_indices = []

with open(clog_filepath, 'r') as f:
    current_header = None
    current_points = []
    current_index = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Frame"):
            if current_points:
                frames_points.append(current_points)
                frame_headers.append(current_header)
                true_frame_indices.append(current_index)
                current_points = []
            current_header = line
            current_index = int(line.split()[1])
        else:
            coords = line.replace('[', '').split(']')
            for coord in coords:
                coord = coord.strip(", ")
                if coord:
                    # x, y, intensity, label, energy = map(int, coord.split(','))
                    parts = coord.split(',')
                    x, y, intensity, label = map(int, parts[:4])
                    # energy = float(parts[4])
                    if intensity == 16383:
                        continue
                    current_points.append((x, y, intensity, label))
    # Add final frame
    if current_points:
        frames_points.append(current_points)
        frame_headers.append(current_header)
        true_frame_indices.append(current_index)

# === Determine image size ===
max_x = max_y = 0
for pts in frames_points:
    for (x, y, _, _) in pts:
        max_x = max(max_x, x)
        max_y = max(max_y, y)
img_width, img_height = max_x + 1, max_y + 1

# === Save NIfTI images + frame mapping ===
affine = np.eye(4)
mapping_entries = []

for i, points in enumerate(frames_points):
    if not points:
        continue

    image = np.zeros((img_height, img_width), dtype=np.uint8)
    for (x, y, intensity, label) in points:
        image[y, x] = min(255, intensity)

    case_str = f'PC_{i:04d}'
    nib.save(nib.Nifti1Image(image, affine), os.path.join(output_dir, f"{case_str}_0000.nii.gz"))
   
    # Save mapping: PC_0000.nii.gz, Frame 42 (timestamp, etc.)
    mapping_entries.append(f"{case_str}.nii.gz,{frame_headers[i]}")
    print(f"✅ Saved {case_str} from {frame_headers[i]}")

# Write mapping file
with open(mapping_file, "w") as f:
    for line in mapping_entries:
        f.write(f"{line}\n")

print(f"\n🎉 Wrote {len(mapping_entries)} frames and saved mapping to: {mapping_file}")