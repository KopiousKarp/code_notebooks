import cv2
import os
import re
import numpy as np
from glob import glob
from tqdm import tqdm
image_dir = "/work/growth_chamber/images/"
mask_dir = "/work/growth_chamber/v4_segmentations/"
out_path = "/work/growth_chamber/timelapse_overlay.mp4"

fps = 4

# All original images (sorted)
image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))

# Load first frame to get dimensions
first = cv2.imread(image_files[0])
h, w = first.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

# Precompile regex to extract plant ID
mask_files = glob(os.path.join(mask_dir, "*.jpg"))
id_regex = re.compile(r".*_(\d+)\.jpg$")  # matches e.g. foo_0003.jpg
plant_ids = sorted({
    int(id_regex.match(f).group(1))
    for f in mask_files
    if id_regex.match(f)
})

num_plants = len(plant_ids)
print("Detected plant IDs:", plant_ids)
print("Generating", num_plants, "colors...")

def generate_color_gradient(n):
    colors = []
    for i in range(n):
        hue = int(179 * (i / n))
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors.append(tuple(int(c) for c in bgr[0,0]))
    return colors

gradient_colors = generate_color_gradient(num_plants)
plant_to_color = {pid: gradient_colors[i] for i, pid in enumerate(plant_ids)}

for img_path in tqdm(image_files):
    frame = cv2.imread(img_path)
    base = os.path.basename(img_path)

    # Find all mask files for this frame
    # Example: frame = "TS_000120.jpg", masks = "TS_000120_1.jpg", "TS_000120_2.jpg"
    mask_pattern = os.path.splitext(base)[0] + "_*.jpg"
    mask_paths = glob(os.path.join(mask_dir, mask_pattern))

    overlay = frame.copy()

    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        m = id_regex.match(mask_path)
        if not m:
            continue
        # print(mask_path, np.unique(mask), np.mean(mask))
        plant_id = int(m.group(1))
        color = plant_to_color[plant_id]    
        # --- PROPER BINARY MASK ---
        # Option A: If masks have per-plant integer labels
        # binary = (mask == plant_id)

        # Option B: If masks are 0/255 only
        binary = mask > 127

        # --- SAFETY CHECK (optional) ---
        # print(mask_path, np.unique(mask), np.sum(binary))

        if np.sum(binary) == 0:
            continue  # nothing to draw

        # Create a color layer for this mask
        color_layer = np.zeros_like(frame)
        color_layer[:] = color  # broadcast color to whole image

        # Create binary mask
        binary = mask > 0

        # Alpha blending
        alpha = 0.4
        overlay[binary] = (overlay[binary] * (1 - alpha) +
                           color_layer[binary] * alpha).astype(np.uint8)

    video.write(overlay)

video.release()
print("✅ Video saved:", out_path)
