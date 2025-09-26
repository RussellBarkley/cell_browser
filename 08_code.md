---
Chapter 9
---

# Preliminaries

## Autoencoder model trained on the MNIST dataset

```{code} python
:label: mnist-ae

import os
import numpy as np
import tensorflow as tf
import h5py
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    AveragePooling2D,
    Flatten,
    Dense,
    Reshape,
    ReLU,
)
from tensorflow.keras.callbacks import CSVLogger, Callback
from tensorflow.keras.optimizers import Adam

# ────────────────────────────────────────
# 0) GPU memory growth
# ────────────────────────────────────────
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

# ────────────────────────────────────────
# 1) Output directories
# ────────────────────────────────────────
BASE_RESULTS_DIR = r"D:\Results\091925_mnist_ae"
TRAIN_RECON_DIR  = os.path.join(BASE_RESULTS_DIR, "recon_train")
VAL_RECON_DIR    = os.path.join(BASE_RESULTS_DIR, "recon_val")
os.makedirs(TRAIN_RECON_DIR, exist_ok=True)
os.makedirs(VAL_RECON_DIR,   exist_ok=True)

# ────────────────────────────────────────
# 2) Load & preprocess MNIST
# ────────────────────────────────────────
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
x_val   = np.expand_dims(x_val.astype("float32")   / 255.0, -1)

# ────────────────────────────────────────
# 3) Hyperparameters
# ────────────────────────────────────────
n_layers      = 2
base_filters  = 16
latent_dim    = 64
learning_rate = 3e-4
batch_size    = 32
epochs        = 100

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    tf.data.Dataset
      .from_tensor_slices((x_train, x_train))
      .shuffle(10_000)
      .batch(batch_size)
      .prefetch(AUTOTUNE)
)
val_ds = (
    tf.data.Dataset
      .from_tensor_slices((x_val, x_val))
      .batch(batch_size)
      .prefetch(AUTOTUNE)
)

# ────────────────────────────────────────
# 4) Build conv autoencoder with single 3×3 conv per block
# ────────────────────────────────────────
# Encoder
inp = Input((28,28,1), name="encoder_input")
x = inp
for i in range(n_layers):
    f = base_filters * (2**i)
    x = Conv2D(f, 3, padding="same")(x)
    x = ReLU()(x)
    x = AveragePooling2D()(x)

flat = Flatten()(x)
z    = Dense(latent_dim, name="z")(flat)
encoder = Model(inp, z, name="encoder")

# Decoder
latent_in = Input((latent_dim,), name="z_sampling")
spatial = 28 // (2**n_layers)
channels = base_filters * (2**(n_layers-1))
x = Dense(spatial * spatial * channels)(latent_in)
x = Reshape((spatial, spatial, channels))(x)

for i in reversed(range(n_layers)):
    f = base_filters * (2**i)
    x = Conv2DTranspose(f, 3, strides=(2,2), padding="same")(x)
    x = ReLU()(x)

decoded = Conv2D(1, 3, padding="same", activation="sigmoid", name="decoder_output")(x)
decoder = Model(latent_in, decoded, name="decoder")

ae = Model(inp, decoder(encoder(inp)), name="autoencoder")
ae.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="mse",
    metrics=["mse"],
)

# ────────────────────────────────────────
# 5) Callback to save originals & reconstructions
# ────────────────────────────────────────
class ReconMNISTCallback(Callback):
    def __init__(self, data, num=100, save_dir=None):
        super().__init__()
        self.data     = data
        self.num      = num
        self.save_dir = save_dir
        self.idx      = np.random.choice(len(data), num, replace=False)

    def on_epoch_end(self, epoch, logs=None):
        imgs   = self.data[self.idx]
        recons = self.model.predict(imgs, verbose=0)

        orig_dir  = os.path.join(self.save_dir, "originals")
        recon_dir = os.path.join(self.save_dir, "reconstructions")
        os.makedirs(orig_dir,  exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

        ep = epoch + 1
        for i in range(self.num):
            orig_img  = (imgs[i,...,0] * 255).astype(np.uint8)
            recon_img = (recons[i,...,0] * 255).astype(np.uint8)

            Image.fromarray(orig_img).save(
                os.path.join(orig_dir, f"orig_epoch{ep:03d}_{i:03d}.png")
            )
            Image.fromarray(recon_img).save(
                os.path.join(recon_dir, f"recon_epoch{ep:03d}_{i:03d}.png")
            )

recon_train_cb = ReconMNISTCallback(x_train, num=100, save_dir=TRAIN_RECON_DIR)
recon_val_cb   = ReconMNISTCallback(x_val,   num=100, save_dir=VAL_RECON_DIR)

# ────────────────────────────────────────
# 6) Train
# ────────────────────────────────────────
csv_logger = CSVLogger(os.path.join(BASE_RESULTS_DIR, "training_log.csv"), append=False)
ae.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[
        csv_logger,
        recon_train_cb,
        recon_val_cb,
    ]
)

# ────────────────────────────────────────
# 7) Save weights & extract latents
# ────────────────────────────────────────
encoder.save_weights(os.path.join(BASE_RESULTS_DIR, "encoder_weights.h5"))
decoder.save_weights(os.path.join(BASE_RESULTS_DIR, "decoder_weights.h5"))

all_images = np.concatenate([x_train, x_val], axis=0)
all_labels = np.concatenate([y_train, y_val], axis=0)
n_total    = all_images.shape[0]

inf_ds = tf.data.Dataset.from_tensor_slices(all_images).batch(batch_size)

h5_path = os.path.join(BASE_RESULTS_DIR, "latents.h5")
with h5py.File(h5_path, "w") as hf:
    hf.create_dataset("z",      shape=(n_total, latent_dim), dtype="f4")
    hf.create_dataset("labels", shape=(n_total,),             dtype="i8")
    idx = 0
    for batch in inf_ds:
        z_batch = encoder.predict(batch, verbose=0)
        b = z_batch.shape[0]
        hf["z"][idx:idx+b, :]   = z_batch
        hf["labels"][idx:idx+b] = all_labels[idx:idx+b]
        idx += b

print(f"[DONE] Latents + labels saved to {h5_path}")

```

# Methods

## 1. Sort tiles into quarters

```{code} python
:label: sort-quarters
:caption: Sort images into four quarters and rename them to stitch 25x25 tile grids.

import os
import glob
import shutil

# ---- Configuration ----
# Set the path to your dataset directory (where the .tif files are located)
dataset_dir = "D:/Confocal_imaging_nuclei_tif/Run53"  # <-- update this path

# Define the grid dimensions (number of rows and columns of the full image)
num_rows = 50  # adjust as needed
num_cols = 50   # adjust as needed

# Define output directories for each quarter relative to the dataset directory
output_dirs = {
    "top_left": os.path.join(dataset_dir, "quarter_top_left"),
    "top_right": os.path.join(dataset_dir, "quarter_top_right"),
    "bottom_left": os.path.join(dataset_dir, "quarter_bottom_left"),
    "bottom_right": os.path.join(dataset_dir, "quarter_bottom_right")
}

# Create output directories if they don't already exist
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# ---- Helper Functions ----
def get_tile_position(index, cols):
    """
    Compute the (row, col) position in the full grid for a given tile index,
    using snake pattern ordering.
    For even rows (0-indexed), the tile order is left-to-right.
    For odd rows, the order is reversed (right-to-left).
    """
    row = index // cols
    col_in_row = index % cols
    if row % 2 == 1:
        # Reverse the order for odd rows.
        col = (cols - 1) - col_in_row
    else:
        col = col_in_row
    return row, col

def determine_quarter(row, col, total_rows, total_cols):
    """
    Determine which quarter the tile belongs to based on its full image position.
    """
    mid_row = total_rows // 2
    mid_col = total_cols // 2

    if row < mid_row:
        if col < mid_col:
            return "top_left"
        else:
            return "top_right"
    else:
        if col < mid_col:
            return "bottom_left"
        else:
            return "bottom_right"

def get_quarter_offset(quarter, total_rows, total_cols):
    """
    Returns the (row_offset, col_offset) for a given quarter.
    """
    mid_row = total_rows // 2
    mid_col = total_cols // 2

    if quarter == "top_left":
        return 0, 0
    elif quarter == "top_right":
        return 0, mid_col
    elif quarter == "bottom_left":
        return mid_row, 0
    elif quarter == "bottom_right":
        return mid_row, mid_col

def get_quarter_dimensions(quarter, total_rows, total_cols):
    """
    Returns the (num_rows, num_cols) dimensions for the given quarter.
    """
    mid_row = total_rows // 2
    mid_col = total_cols // 2

    if quarter == "top_left":
        return mid_row, mid_col
    elif quarter == "top_right":
        return mid_row, total_cols - mid_col
    elif quarter == "bottom_left":
        return total_rows - mid_row, mid_col
    elif quarter == "bottom_right":
        return total_rows - mid_row, total_cols - mid_col

def quarter_sort_key(tile_info):
    _, full_row, full_col, quarter = tile_info
    offset_row, offset_col = get_quarter_offset(quarter, num_rows, num_cols)
    # Compute the local coordinates within the quarter.
    local_row = full_row - offset_row
    local_col = full_col - offset_col
    # Get the number of columns in this quarter.
    _, quarter_cols = get_quarter_dimensions(quarter, num_rows, num_cols)
    # For a snake pattern in the quarter, reverse the order of columns on odd local rows.
    if local_row % 2 == 1:
        adjusted_local_col = (quarter_cols - 1) - local_col
    else:
        adjusted_local_col = local_col
    return (local_row, adjusted_local_col)

# ---- Main Processing ----
def sort_and_rename_tiles():
    # Gather all .tif files from the dataset directory.
    file_pattern = os.path.join(dataset_dir, "*.tif")
    tile_files = sorted(glob.glob(file_pattern))

    total_tiles_expected = num_rows * num_cols
    if len(tile_files) != total_tiles_expected:
        print(f"Warning: Expected {total_tiles_expected} tiles but found {len(tile_files)}.")

    # Prepare a dictionary to hold tiles for each quarter.
    quarters = {
        "top_left": [],
        "top_right": [],
        "bottom_left": [],
        "bottom_right": []
    }

    # Compute full image positions and assign tiles to quarters.
    for i, tile_path in enumerate(tile_files):
        full_row, full_col = get_tile_position(i, num_cols)
        quarter = determine_quarter(full_row, full_col, num_rows, num_cols)
        # Append tuple: (tile_path, full_row, full_col, quarter)
        quarters[quarter].append((tile_path, full_row, full_col, quarter))
        print(f"Tile {os.path.basename(tile_path)}: full (row={full_row}, col={full_col}) -> {quarter}")

    # For each quarter, sort the tiles in the snake pattern order relative to that quarter,
    # then rename and copy the files into the corresponding output folder.
    for quarter, tiles in quarters.items():
        # Sort the tiles based on their local (row, col) using the snake pattern.
        sorted_tiles = sorted(tiles, key=quarter_sort_key)
        dest_dir = output_dirs[quarter]
        print(f"\nProcessing {quarter} with {len(sorted_tiles)} tiles:")

        for new_index, tile_info in enumerate(sorted_tiles):
            tile_path, full_row, full_col, _ = tile_info
            new_filename = f"{new_index:04d}.tif"
            dest_path = os.path.join(dest_dir, new_filename)

            shutil.copy(tile_path, dest_path)
            print(f"  {os.path.basename(tile_path)} (full row={full_row}, col={full_col}) -> {new_filename}")

if __name__ == "__main__":
    sort_and_rename_tiles()

```

## 2. Crop stitched quarters

```{code} python
:label: crop-quarters
:caption: Crop the stitched quarters into quarters, resulting in sixteen stitched images.

from PIL import Image
import os
import glob

# Disable the decompression bomb check (use with caution)
Image.MAX_IMAGE_PIXELS = None

# Define the input directory
input_folder = r'D:\Confocal_imaging_nuclei_tif\MIST_Fused_Images\Quarters'

# Define the output folder as a subdirectory within the input folder
output_folder = os.path.join(input_folder, "Tiled")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Search for all .tif files in the input folder
tif_files = glob.glob(os.path.join(input_folder, '*.tif'))

# Process each .tif file
for tif_path in tif_files:
    try:
        # Open the image using Pillow
        image = Image.open(tif_path)
        width, height = image.size

        # Define coordinate boxes for the four quarters of the image
        boxes = {
            "top_left": (0, 0, width // 2, height // 2),
            "top_right": (width // 2, 0, width, height // 2),
            "bottom_left": (0, height // 2, width // 2, height),
            "bottom_right": (width // 2, height // 2, width, height),
        }

        # Get the base name of the file to use in the output file name
        base_filename = os.path.splitext(os.path.basename(tif_path))[0]

        # Crop the image into four parts and save each to the output folder
        for label, box in boxes.items():
            cropped_img = image.crop(box)
            output_file = os.path.join(output_folder, f"{base_filename}_{label}.tif")
            cropped_img.save(output_file)

        print(f"Processed {tif_path}")
    except Exception as e:
        print(f"Error processing {tif_path}: {e}")

print("Tiling complete!")
```

## 3. Align and crop ROIs

```{code} python
:label: crop-rois
:caption: Center, pre-align and crop the masked ROIs.

import os
import numpy as np
import cv2
from PIL import Image
from tifffile import imread, imwrite
import concurrent.futures

# Suppress decompression bomb warnings
Image.MAX_IMAGE_PIXELS = None  # Allow processing of large images without warnings

# Parameters for cropping and bounding box sizes
crop_size = 256  # Final crop size
bbox_size = 364  # Size of the bounding box around the ROI
half_bbox = bbox_size // 2
half_crop = crop_size // 2

# Define the directory containing all .tif and .png files
image_dir = r"D:\Confocal_imaging_nuclei_tif\MIST_Fused_Images\Quarters\Tiled"
# Define output directory for ROIs
output_dir = os.path.join(image_dir, "ROIs")
os.makedirs(output_dir, exist_ok=True)

def process_tif_file(args):
    tif_file, image_dir, output_dir = args
    image_path = os.path.join(image_dir, tif_file)
    mask_path = os.path.join(image_dir, tif_file.replace('.tif', '_cp_masks.png'))

    if not os.path.exists(mask_path):
        print(f"Mask for {tif_file} not found. Skipping.")
        return

    # Load the stitched image and mask
    try:
        stitched_image = imread(image_path)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return

    try:
        mask = np.array(Image.open(mask_path))
    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return

    # Get unique ROI values from the mask (excluding background 0)
    roi_values = np.unique(mask)
    roi_values = roi_values[roi_values != 0]

    print(f"Processing {tif_file}: Found {len(roi_values)} ROIs.")

    for roi in roi_values:
        roi_mask = (mask == roi).astype(np.uint8)
        y_idx, x_idx = np.where(roi_mask)
        if y_idx.size == 0 or x_idx.size == 0:
            continue

        # Skip if ROI touches the image border
        if (y_idx.min() == 0 or x_idx.min() == 0 or
                y_idx.max() == (mask.shape[0] - 1) or x_idx.max() == (mask.shape[1] - 1)):
            print(f"Skipping ROI {roi} in {tif_file} because it touches the image border.")
            continue

        # Calculate the center of the ROI
        x_center = (x_idx.min() + x_idx.max()) // 2
        y_center = (y_idx.min() + y_idx.max()) // 2

        # Define bounding box around the ROI center
        x_start = max(0, x_center - half_bbox)
        y_start = max(0, y_center - half_bbox)
        x_end = min(stitched_image.shape[1], x_center + half_bbox)
        y_end = min(stitched_image.shape[0], y_center + half_bbox)

        if (x_end - x_start) < bbox_size or (y_end - y_start) < bbox_size:
            print(f"Skipping ROI {roi} in {tif_file} due to insufficient area for a {bbox_size}x{bbox_size} crop.")
            continue

        sub_image = stitched_image[y_start:y_end, x_start:x_end]
        sub_mask = roi_mask[y_start:y_end, x_start:x_end]

        # Find contours for orientation
        contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        (center_rect, (w_rect, h_rect), angle) = rect
        if w_rect < h_rect:
            angle += 90

        (h_sub, w_sub) = sub_mask.shape
        center_sub = (w_sub // 2, h_sub // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center_sub, angle, 1.0)
        rotated_sub_image = cv2.warpAffine(sub_image, rotation_matrix, (w_sub, h_sub))
        rotated_sub_mask = cv2.warpAffine(sub_mask, rotation_matrix, (w_sub, h_sub))

        # Determine final crop area based on rotated mask
        y_coords, x_coords = np.where(rotated_sub_mask > 0)
        if y_coords.size == 0 or x_coords.size == 0:
            print(f"Skipping ROI {roi} in {tif_file} due to invalid mask after rotation.")
            continue

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        x_center_rot = (x_min + x_max) // 2
        y_center_rot = (y_min + y_max) // 2

        crop_x_start = max(0, x_center_rot - half_crop)
        crop_y_start = max(0, y_center_rot - half_crop)
        crop_x_end = crop_x_start + crop_size
        crop_y_end = crop_y_start + crop_size

        # Adjust if crop exceeds image boundaries
        if crop_x_end > rotated_sub_image.shape[1]:
            crop_x_start = rotated_sub_image.shape[1] - crop_size
            crop_x_end = rotated_sub_image.shape[1]
        if crop_y_end > rotated_sub_image.shape[0]:
            crop_y_start = rotated_sub_image.shape[0] - crop_size
            crop_y_end = rotated_sub_image.shape[0]

        cropped_image = rotated_sub_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        cropped_mask = rotated_sub_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        final_cropped_image = cropped_image.copy()
        final_cropped_image[cropped_mask == 0] = 0

        output_filename = f"{os.path.splitext(tif_file)[0]}_ROI_{roi}.tif"
        output_path = os.path.join(output_dir, output_filename)
        try:
            imwrite(output_path, final_cropped_image)
            print(f"Saved cropped image for ROI {roi} from {tif_file} at {output_path}.")
        except Exception as e:
            print(f"Failed to save {output_path}: {e}")

if __name__ == '__main__':
    tif_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    print(f"Found {len(tif_files)} .tif files to process.")
    args_list = [(tif_file, image_dir, output_dir) for tif_file in tif_files]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_tif_file, args_list)

    print("Cropping completed!")
```

# Dataset

## Measure number and mean intensity of ROIs in stitched images

```{code} python
:label: scatter-code

import os
import glob
import csv
import warnings

import numpy as np
from tifffile import imread as tif_read
from PIL import Image
from PIL.Image import DecompressionBombWarning, DecompressionBombError

# -----------------------------------------------------------------------------
# DISABLE THE WARNING AND PREPARE FOR ERROR-BYPASS
# -----------------------------------------------------------------------------
# Suppress the warning
warnings.simplefilter('ignore', DecompressionBombWarning)
# Remove any size limit
Image.MAX_IMAGE_PIXELS = None

def safe_open(path):
    """
    Try Image.open; if a DecompressionBombError is raised,
    reset MAX_IMAGE_PIXELS and retry once.
    """
    try:
        return Image.open(path)
    except DecompressionBombError:
        Image.MAX_IMAGE_PIXELS = None
        return Image.open(path)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
input_dir  = r'D:\Confocal_imaging_nuclei_tif\MIST_Fused_Images'
output_csv = os.path.join(input_dir, 'results.csv')
# -----------------------------------------------------------------------------

def process_image_pair(tif_path, mask_path):
    """
    Returns (num_rois, mean_intensity_all_rois, std_intensity_all_rois)
    """
    img = tif_read(tif_path).astype(np.float32)
    mask = np.array(safe_open(mask_path))

    labels   = np.unique(mask)
    roi_ids  = labels[labels > 0]
    num_rois = roi_ids.size

    if num_rois == 0:
        return 0, float('nan'), float('nan')

    pixels = img[mask > 0]
    mean_i = float(np.mean(pixels))
    std_i  = float(np.std(pixels))

    return num_rois, mean_i, std_i

def main():
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'num_rois', 'mean_intensity', 'std_intensity'])

        for tif_path in sorted(glob.glob(os.path.join(input_dir, '*.tif'))):
            base = os.path.splitext(os.path.basename(tif_path))[0]
            mask_name = base + '_cp_masks.png'
            mask_path = os.path.join(input_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"Skipping {base}: no mask named '{mask_name}'")
                continue

            n, mean_i, std_i = process_image_pair(tif_path, mask_path)
            writer.writerow([base + '.tif', n, f"{mean_i:.4f}", f"{std_i:.4f}"])
            print(f"Processed {base}: ROIs={n}, mean={mean_i:.2f}, std={std_i:.2f}")

    print(f"\nResults written to: {output_csv}")

if __name__ == '__main__':
    main()
```

```{code} python
:label: ten-stitched-images
:caption: Prints ten stitched images closest to the centroid by z-score normalized euclidean distance.

#!/usr/bin/env python3
# Find 10 images closest to the mean centroid with equal weighting of features.
# Also print the per-image std (if a std-like column exists).

import os, sys
import pandas as pd
import numpy as np

CSV_PATH = r"D:\Results\Interactive_figures\Stitched_sixteenth\results.csv"

def find_col(cols, names):
    lc = {c.lower(): c for c in cols}
    for n in names:
        n = n.lower()
        if n in lc:
            return lc[n]
        for k, v in lc.items():
            if n in k:
                return v
    return None

df = pd.read_csv(CSV_PATH)

img_col   = find_col(df.columns, ["image","filename","file","img"])
nroi_col  = find_col(df.columns, ["num_rois","n_rois","roi_count"])
mean_col  = find_col(df.columns, ["mean_intensity","mean_inte","avg_intensity"])
std_col   = find_col(df.columns, ["std","std_intensity","intensity_std","stdev","stddev","sigma"])

if None in (img_col, nroi_col, mean_col):
    sys.exit(f"Missing required columns. Found: {list(df.columns)}")

use_cols = [img_col, nroi_col, mean_col] + ([std_col] if std_col is not None else [])
df = df[use_cols].dropna().copy()

# Centroid in ORIGINAL units (for reporting)
mu_rois = df[nroi_col].mean()
mu_mean = df[mean_col].mean()

# ---- Equal-weighted distance (z-score each feature) ----
eps = 1e-9
std_rois = df[nroi_col].std(ddof=0) + eps
std_mean = df[mean_col].std(ddof=0) + eps

z_rois = (df[nroi_col] - mu_rois) / std_rois
z_mean = (df[mean_col] - mu_mean) / std_mean

df["distance_eqw"] = np.sqrt(z_rois**2 + z_mean**2)

top = df.sort_values("distance_eqw").head(10)

print(f"Centroid (original units) over {len(df)} images:")
print(f"  num_rois       = {mu_rois:.4f}")
print(f"  mean_intensity = {mu_mean:.4f}\n")

if std_col is None:
    print("[i] No per-image std column found (searched for: std, std_intensity, intensity_std, stdev, stddev, sigma).")
    print("10 filenames closest to centroid (equal-weighted by z-scoring):")
    for i, r in enumerate(top.itertuples(index=False), 1):
        print(f"{i:2d}. {getattr(r, img_col)}  "
              f"(num_rois={getattr(r, nroi_col)}, mean_intensity={getattr(r, mean_col):.4f}, "
              f"zDist={getattr(r, 'distance_eqw'):.4f})")
else:
    print("10 filenames closest to centroid (equal-weighted by z-scoring):")
    for i, r in enumerate(top.itertuples(index=False), 1):
        print(f"{i:2d}. {getattr(r, img_col)}  "
              f"(num_rois={getattr(r, nroi_col)}, mean_intensity={getattr(r, mean_col):.4f}, "
              f"std={getattr(r, std_col):.4f}, zDist={getattr(r, 'distance_eqw'):.4f})")
```

```{code} python
:label: TFRecord-shards
:caption: Splits the dataset into one-hundred TFRecord shards.

import os
import hashlib
import numpy as np
import tensorflow as tf
from tifffile import imread

# ------------ Configuration ------------
input_dir   = r'D:\Confocal_imaging_nuclei_tif\MIST_Fused_Images\ROIs'
output_dir  = r'D:/Results/TFRecords'
num_shards  = 100
# ---------------------------------------

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def image_example(image_bytes, filename_relpath):
    return tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_bytes),
        'filename':  _bytes_feature(filename_relpath.encode('utf-8')),  # store RELATIVE path
    }))

os.makedirs(output_dir, exist_ok=True)
writers = [
    tf.io.TFRecordWriter(os.path.join(output_dir, f"data_shard_{i:03d}.tfrecords"))
    for i in range(num_shards)
]

print(f"Writing {num_shards} shards to {output_dir} from {input_dir}")
count = 0

# Deterministic iteration (sort by name or by full path)
entries = sorted(
    [e for e in os.scandir(input_dir) if e.is_file() and e.name.lower().endswith('.tif')],
    key=lambda e: e.name  # or e.path if you prefer
)

for entry in entries:
    img = imread(entry.path)  # expect (256, 256), uint8

    if img.shape != (256, 256):
        raise ValueError(f"Unexpected image shape {img.shape} for {entry.name}")

    # ensure uint8, preserve raw values (no per-image normalization)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8, copy=False)

    img_bytes = img.tobytes(order='C')

    # Stable shard index from RELATIVE PATH (reproducible across runs/machines)
    relpath  = os.path.relpath(entry.path, input_dir)
    hval     = int(hashlib.md5(relpath.encode('utf-8')).hexdigest(), 16)
    shard_idx = hval % num_shards

    example = image_example(img_bytes, relpath)
    writers[shard_idx].write(example.SerializeToString())

    count += 1
    if count % 10000 == 0:
        print(f"Processed {count} images...")

for w in writers:
    w.close()

print(f"TFRecord sharding complete. Total images: {count}")
```

```{code} python
:label: nucleus_ae
:caption: Convolutional autoencoder model trained on the nucleus dataset.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, AveragePooling2D,
    Flatten, Dense, Reshape, ReLU,
)
from tensorflow.keras.callbacks import (
    Callback, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tifffile import imread, imwrite
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────────────────────────────
# 0) GPU memory growth
# ────────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# ────────────────────────────────────────────────────────────────────────
# 1) Paths & TFRecord shards
# ────────────────────────────────────────────────────────────────────────
shards_dir  = r'D:\Results\TFRecords'        # your uint8 TFRecords folder
results_dir = r'D:/Results/09052025_AE1M_Conv2DTranspose'
os.makedirs(results_dir, exist_ok=True)

all_shards = sorted([
    os.path.join(shards_dir, f)
    for f in os.listdir(shards_dir)
    if f.endswith('.tfrecords')
])
train_shards, val_shards = train_test_split(all_shards, test_size=0.2, random_state=42)

batch_size = 32
tfrecord_compression = None   # 'GZIP' only if you wrote with gzip

# Input pipeline knobs (kept modest to avoid OOM)
SHUFFLE_BUFFER     = 2000   # was 10000
NUM_PARALLEL_READS = 4      # was AUTOTUNE
PREFETCH_BUFS      = 1      # was AUTOTUNE

# ────────────────────────────────────────────────────────────────────────
# 2) TFRecord parsing for raw uint8 → float32 [0,1]
# ────────────────────────────────────────────────────────────────────────
feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'filename':  tf.io.FixedLenFeature([], tf.string),  # kept for reference
}

def _parse_example_uint8(example_proto):
    data = tf.io.parse_single_example(example_proto, feature_description)
    img  = tf.io.decode_raw(data['image_raw'], tf.uint8)      # uint8 on disk
    img  = tf.reshape(img, [256, 256, 1])                     # grayscale
    img  = tf.image.convert_image_dtype(img, tf.float32)      # [0,255] -> [0,1]
    return img, img

# ────────────────────────────────────────────────────────────────────────
# 3) Build datasets (small, predictable memory footprint)
# ────────────────────────────────────────────────────────────────────────
def make_dataset(shard_list, shuffle_buffer=SHUFFLE_BUFFER, training=True):
    ds = tf.data.TFRecordDataset(
        shard_list,
        num_parallel_reads=NUM_PARALLEL_READS,   # ← capped
        compression_type=tfrecord_compression
    )
    # Shuffle BEFORE parsing so we shuffle compact serialized records, not tensors
    if training:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        opts = tf.data.Options()
        opts.experimental_deterministic = False  # allow non-deterministic order for speed
        ds = ds.with_options(opts)

    ds = ds.map(_parse_example_uint8, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)  # static shapes help performance/memory
    ds = ds.prefetch(PREFETCH_BUFS)                 # ← small prefetch to avoid spikes
    return ds

train_ds = make_dataset(train_shards, training=True)
val_ds   = make_dataset(val_shards,   training=False)

# ────────────────────────────────────────────────────────────────────────
# 4) Autoencoder (unchanged)
# ────────────────────────────────────────────────────────────────────────
latent_dim    = 512
learning_rate = 3e-4

inp = Input((256,256,1), name='encoder_input')
x = inp
for filters in [16, 32, 64, 128]:
    x = Conv2D(filters, 3, padding='same')(x); x = ReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x); x = ReLU()(x)
    x = AveragePooling2D()(x)

flat = Flatten()(x)
z = Dense(latent_dim, name='z')(flat)
encoder = Model(inp, z, name='encoder')

latent_in = Input((latent_dim,), name='z_sampling')
x = Dense(16 * 16 * 128)(latent_in)
x = Reshape((16, 16, 128))(x)
for filters in [128, 64, 32, 16]:
    x = Conv2DTranspose(filters, 3, strides=2, padding='same')(x); x = ReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x); x = ReLU()(x)

decoded = Conv2D(1, 3, padding='same', activation='sigmoid', name='decoder_output')(x)
decoder = Model(latent_in, decoded, name='decoder')

ae = Model(inp, decoder(encoder(inp)), name='autoencoder')
ae.compile(optimizer=Adam(learning_rate), loss='mse', metrics=['mse'])

# ────────────────────────────────────────────────────────────────────────
# 5) ReconCallback (same logic; predict in small batches to avoid spikes)
# ────────────────────────────────────────────────────────────────────────
class ReconCallback(tf.keras.callbacks.Callback):
    def __init__(self, files, num=100, save_dir=None, pred_batch=8):
        super().__init__()
        self.files      = files
        self.num        = min(num, len(files))
        self.save_dir   = save_dir
        self.pred_batch = pred_batch
        self.idx        = np.random.choice(len(files), self.num, replace=False)

    def on_epoch_end(self, epoch, logs=None):
        sel_files = [self.files[i] for i in self.idx]
        raw_stack = [imread(f).astype(np.float32) for f in sel_files]
        imgs_norm = np.stack([r / 255.0 for r in raw_stack])[..., np.newaxis]

        # ← small predict batch to keep peak GPU usage low
        recon_norm = self.model.predict(imgs_norm, batch_size=self.pred_batch, verbose=0)[..., 0]

        orig_dir = os.path.join(self.save_dir, 'originals')
        recon_dir = os.path.join(self.save_dir, 'reconstructions')
        os.makedirs(orig_dir, exist_ok=True); os.makedirs(recon_dir, exist_ok=True)

        for i, raw in enumerate(raw_stack):
            imwrite(os.path.join(orig_dir, f'orig_epoch{epoch+1:03d}_{i:03d}.png'),
                    raw.astype(np.uint8))
        for i, recon in enumerate(recon_norm):
            imwrite(os.path.join(recon_dir, f'recon_epoch{epoch+1:03d}_{i:03d}.png'),
                    (recon * 255).astype(np.uint8))

# ────────────────────────────────────────────────────────────────────────
# 6) Collect TIFFs (for ReconCallback) and split lists
# ────────────────────────────────────────────────────────────────────────
all_orig_files = [
    os.path.join(r'D:/Confocal_imaging_nuclei_tif/MIST_Fused_Images/ROIs', f)
    for f in os.listdir(r'D:/Confocal_imaging_nuclei_tif/MIST_Fused_Images/ROIs')
    if f.lower().endswith('.tif')
]
train_orig_files, val_orig_files = train_test_split(all_orig_files, test_size=0.2, random_state=42)

# ────────────────────────────────────────────────────────────────────────
# 7) Train
# ────────────────────────────────────────────────────────────────────────
csv_cb = CSVLogger(os.path.join(results_dir, 'training_log.csv'), separator=',', append=False)

ae.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[
        csv_cb,
        ReconCallback(train_orig_files, 100, os.path.join(results_dir, 'reconstructions_train'), pred_batch=8),
        ReconCallback(val_orig_files,   100, os.path.join(results_dir, 'reconstructions_val'),   pred_batch=8),
    ]
)

# ────────────────────────────────────────────────────────────────────────
# 8) Save weights
# ────────────────────────────────────────────────────────────────────────
encoder.save_weights(os.path.join(results_dir, 'encoder_weights.h5'))
decoder.save_weights(os.path.join(results_dir, 'decoder_weights.h5'))

```

# Results

## Embed latents

```{code} python
:label: embed_latents

#!/usr/bin/env python3
"""
Embed TFRecord dataset images using a saved encoder and write latents to HDF5.

Outputs:
  - latents.h5 with datasets:
      /z          -> float32, shape (N, latent_dim)
      /filenames  -> utf-8 strings, shape (N,)
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import h5py

# ────────────────────────────────────────────────────────────────────────
# 0) Configuration
# ────────────────────────────────────────────────────────────────────────
# Paths (update if needed)
SHARDS_DIR   = r'D:\Results\TFRecords'                      # same TFRecords you trained on
RESULTS_DIR  = r'D:/Results/09052025_AE1M_Conv2DTranspose'  # where encoder_weights.h5 lives
WEIGHTS_PATH = os.path.join(RESULTS_DIR, 'encoder_weights.h5')
OUT_H5_PATH  = os.path.join(RESULTS_DIR, 'latents.h5')

# TFRecord compression type: None or 'GZIP' (must match how shards were written)
TFRECORD_COMPRESSION = None

# Pipeline + model knobs
BATCH_SIZE          = 64           # inference batch; raise if you have headroom
NUM_PARALLEL_READS  = 4            # keep modest to avoid I/O bursts
PREFETCH_BUFS       = 1
LATENT_DIM          = 512
INPUT_SHAPE         = (256, 256, 1)

# HDF5 chunking (append granularity); larger chunks = fewer resizes
H5_CHUNK_ROWS       = 8192

# ────────────────────────────────────────────────────────────────────────
# 1) Safety checks
# ────────────────────────────────────────────────────────────────────────
if not os.path.isdir(SHARDS_DIR):
    print(f"[!] SHARDS_DIR not found: {SHARDS_DIR}", file=sys.stderr); sys.exit(1)

if not os.path.isfile(WEIGHTS_PATH):
    print(f"[!] Encoder weights not found: {WEIGHTS_PATH}", file=sys.stderr); sys.exit(1)

# Collect shards
all_shards = sorted(
    os.path.join(SHARDS_DIR, f)
    for f in os.listdir(SHARDS_DIR)
    if f.endswith('.tfrecords')
)
if not all_shards:
    print(f"[!] No .tfrecords found in {SHARDS_DIR}", file=sys.stderr); sys.exit(1)

print(f"[i] Found {len(all_shards)} shard(s).")
print(f"[i] Weights: {WEIGHTS_PATH}")
print(f"[i] Output : {OUT_H5_PATH}")

# ────────────────────────────────────────────────────────────────────────
# 2) GPU memory growth (safe if no GPU as well)
# ────────────────────────────────────────────────────────────────────────
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if gpus:
        print(f"[i] GPUs: {[d.name for d in gpus]} (memory growth enabled)")
except Exception as e:
    print(f"[!] Could not set GPU memory growth: {e}")

# ────────────────────────────────────────────────────────────────────────
# 3) TFRecord parsing: uint8 → float32 [0,1], keep filename
# ────────────────────────────────────────────────────────────────────────
FEATURES = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'filename':  tf.io.FixedLenFeature([], tf.string),
}

def parse_example(example_proto):
    data = tf.io.parse_single_example(example_proto, FEATURES)
    img  = tf.io.decode_raw(data['image_raw'], tf.uint8)
    img  = tf.reshape(img, INPUT_SHAPE)
    img  = tf.image.convert_image_dtype(img, tf.float32)  # [0,255] -> [0,1]
    fname = data['filename']                              # tf.string scalar
    return img, fname

def make_dataset(shards):
    ds = tf.data.TFRecordDataset(
        shards,
        num_parallel_reads=NUM_PARALLEL_READS,
        compression_type=TFRECORD_COMPRESSION
    )
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(PREFETCH_BUFS)
    return ds

dataset = make_dataset(all_shards)

# ────────────────────────────────────────────────────────────────────────
# 4) Rebuild encoder architecture (must match training)
# ────────────────────────────────────────────────────────────────────────
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, ReLU

def build_encoder(input_shape=(256, 256, 1), latent_dim=512):
    inp = Input(input_shape, name='encoder_input')
    x = inp
    for filters in [16, 32, 64, 128]:
        x = Conv2D(filters, 3, padding='same')(x); x = ReLU()(x)
        x = Conv2D(filters, 3, padding='same')(x); x = ReLU()(x)
        # Explicit pool size/stride to mirror training
        x = AveragePooling2D(pool_size=2, strides=2, padding='valid')(x)

    # After four 2×2 pools: 256→128→64→32→16 (with 128 channels)
    flat = Flatten()(x)
    z = Dense(latent_dim, name='z')(flat)
    return Model(inp, z, name='encoder')

encoder = build_encoder()
encoder.load_weights(WEIGHTS_PATH)
encoder.trainable = False
print("[i] Encoder rebuilt and weights loaded.")

# ────────────────────────────────────────────────────────────────────────
# 5) Prepare HDF5 with resizable datasets
# ────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_H5_PATH), exist_ok=True)
if os.path.exists(OUT_H5_PATH):
    print(f"[!] Removing existing output: {OUT_H5_PATH}")
    os.remove(OUT_H5_PATH)

str_dt = h5py.string_dtype(encoding='utf-8')
with h5py.File(OUT_H5_PATH, 'w') as h5:
    z_ds  = h5.create_dataset(
        'z',
        shape=(0, LATENT_DIM),
        maxshape=(None, LATENT_DIM),
        dtype='float32',
        chunks=(H5_CHUNK_ROWS, LATENT_DIM)
    )
    fn_ds = h5.create_dataset(
        'filenames',
        shape=(0,),
        maxshape=(None,),
        dtype=str_dt,
        chunks=(H5_CHUNK_ROWS,)
    )

    # Bytes variant (also NumPy 2.x safe)
    h5.attrs['created_utc'] = np.bytes_(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    h5.attrs['input_shape'] = np.bytes_(str(INPUT_SHAPE))
    h5.attrs['weights_path'] = np.bytes_(WEIGHTS_PATH)
    h5.attrs['tfrecords_dir'] = np.bytes_(SHARDS_DIR)
    h5.attrs['compression_type'] = np.bytes_(str(TFRECORD_COMPRESSION))
    h5.attrs['latent_dim'] = LATENT_DIM

    # ────────────────────────────────────────────────────────────────────
    # 6) Stream, embed, and append to HDF5
    # ────────────────────────────────────────────────────────────────────
    cursor = 0
    t0 = time.time()
    batches_done = 0

    # Using .as_numpy_iterator() to easily get numpy arrays (filenames as bytes)
    for imgs, fnames in dataset.as_numpy_iterator():
        # Encode
        z_batch = encoder.predict(imgs, batch_size=BATCH_SIZE, verbose=0)  # (B, 512)
        n = z_batch.shape[0]

        # Resize and write
        z_ds.resize(cursor + n, axis=0)
        fn_ds.resize(cursor + n, axis=0)

        z_ds[cursor:cursor+n] = z_batch
        # Decode filenames from bytes → str
        fn_list = [f.decode('utf-8') if isinstance(f, (bytes, bytearray)) else str(f) for f in fnames]
        fn_ds[cursor:cursor+n] = np.array(fn_list, dtype=object)

        cursor += n
        batches_done += 1

        # Light progress feedback every ~100 batches
        if batches_done % 100 == 0:
            elapsed = time.time() - t0
            print(f"[i] {cursor:,} embeddings written ({batches_done} batches) in {elapsed/60:.1f} min.")

    total = cursor
    elapsed = time.time() - t0
    print(f"[✓] Done. Wrote {total:,} embeddings to {OUT_H5_PATH} in {elapsed/60:.1f} min.")
```

## t-SNE, UMAP and PCA embeddings

```{code} python
:label: embeddings

#!/usr/bin/env python3
"""
Compute PCA (top-5 pairwise 2D), and UMAP/t-SNE with both cosine and euclidean
metrics from RESULTS_DIR/latents.h5. No attrs; saves next to latents.h5.

Output files (in RESULTS_DIR):
  - pca_embeddings.h5
      /scores_top5
      /PC1_PC2, /PC1_PC3, ... (pairwise from top-5 PCs)

  - umap_embeddings.h5
      /umap_cosine
      /umap_euclidean
      /filenames

  - tsne_embeddings.h5
      /tsne_cosine
      /tsne_euclidean
      /filenames
"""

import os
import time
import h5py
import numpy as np

# --- threads politely ---
logical_cores = os.cpu_count() or 1
os.environ['LOKY_MAX_CPU_COUNT']   = str(logical_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(logical_cores)
os.environ['OMP_NUM_THREADS']      = str(logical_cores)
os.environ['MKL_NUM_THREADS']      = str(logical_cores)
os.environ['NUMEXPR_NUM_THREADS']  = str(logical_cores)

from sklearn.decomposition import PCA
import umap
from openTSNE import TSNE

# -----------------------------
# CONFIG — EDIT THIS
# -----------------------------
RESULTS_DIR = r'D:/Results/09052025_AE1M_Conv2DTranspose'
LATENTS_H5  = os.path.join(RESULTS_DIR, 'latents.h5')

OUT_PCA_H5  = os.path.join(RESULTS_DIR, 'pca_embeddings.h5')
OUT_UMAP_H5 = os.path.join(RESULTS_DIR, 'umap_embeddings.h5')
OUT_TSNE_H5 = os.path.join(RESULTS_DIR, 'tsne_embeddings.h5')

# Base params (metric overridden per run)
UMAP_PARAMS_BASE = dict(
    n_components=2,
    random_state=42,
    init="pca",
    low_memory=True,
    verbose=True,
    n_jobs=logical_cores,  # keep for your environment
)

TSNE_PARAMS_BASE = dict(
    n_components=2,
    initialization="pca",
    learning_rate="auto",
    random_state=42,
    n_jobs=logical_cores,
)

# Optional compression (set to 'gzip' to shrink files)
H5_COMPRESSION      = None
H5_COMPRESSION_OPTS = 4


def _create_ds(h5, name, shape, dtype="float32"):
    kwargs = dict(dtype=dtype)
    if H5_COMPRESSION:
        kwargs.update(compression=H5_COMPRESSION, compression_opts=H5_COMPRESSION_OPTS)
    return h5.create_dataset(name, shape=shape, **kwargs)


def main():
    t0 = time.time()
    if not os.path.isfile(LATENTS_H5):
        raise FileNotFoundError(f"[!] latents.h5 not found at: {LATENTS_H5}")

    # 1) Load latents + filenames
    print(f"[i] Reading latents: {LATENTS_H5}")
    with h5py.File(LATENTS_H5, "r") as hf:
        z = np.asarray(hf["z"][...], dtype=np.float32)           # (N, D)
        fn_raw = hf["filenames"][...]
        filenames = [f.decode("utf-8") if isinstance(f, (bytes, bytearray)) else str(f) for f in fn_raw]

    N, latent_dim = z.shape
    print(f"[i] Loaded z: {z.shape} (float32). N={N:,}, latent_dim={latent_dim}")

    # 2) PCA (scores + pairwise 2D for PC1..PC5)
    print("[i] PCA: fit top-5 (or fewer)...")
    k_keep = min(5, latent_dim)
    pca_model = PCA(n_components=k_keep, random_state=42)
    scores_top5 = pca_model.fit_transform(z).astype(np.float32)  # (N, k_keep)

    if os.path.exists(OUT_PCA_H5):
        print(f"[!] Removing existing: {OUT_PCA_H5}")
        os.remove(OUT_PCA_H5)

    print(f"[i] Writing PCA → {OUT_PCA_H5}")
    with h5py.File(OUT_PCA_H5, "w") as h5:
        _create_ds(h5, "scores_top5", scores_top5.shape)[:] = scores_top5
        # flat datasets named PCi_PCj
        for i in range(k_keep):
            for j in range(i + 1, k_keep):
                emb_2d = np.stack([scores_top5[:, i], scores_top5[:, j]], axis=1)
                _create_ds(h5, f"PC{i+1}_PC{j+1}", emb_2d.shape)[:] = emb_2d.astype(np.float32)

    # 3) UMAP — cosine & euclidean
    umap_results = {}
    for metric in ("cosine", "euclidean"):
        params = dict(UMAP_PARAMS_BASE, metric=metric)
        print(f"[i] UMAP (metric={metric}) ...")
        umap_model = umap.UMAP(**params)
        umap_results[metric] = umap_model.fit_transform(z).astype(np.float32)

    if os.path.exists(OUT_UMAP_H5):
        print(f"[!] Removing existing: {OUT_UMAP_H5}")
        os.remove(OUT_UMAP_H5)

    print(f"[i] Writing UMAP → {OUT_UMAP_H5}")
    with h5py.File(OUT_UMAP_H5, "w") as h5:
        _create_ds(h5, "umap_cosine",    umap_results["cosine"].shape)[:]    = umap_results["cosine"]
        _create_ds(h5, "umap_euclidean", umap_results["euclidean"].shape)[:] = umap_results["euclidean"]
        # filenames alongside for quick plotting
        str_dt = h5py.string_dtype(encoding="utf-8")
        d = h5.create_dataset("filenames", shape=(N,), dtype=str_dt)
        d[:] = np.array(filenames, dtype=object)

    # 4) t-SNE — cosine & euclidean
    tsne_results = {}
    for metric in ("cosine", "euclidean"):
        params = dict(TSNE_PARAMS_BASE, metric=metric)
        print(f"[i] t-SNE (metric={metric}) ...")
        tsne_results[metric] = TSNE(**params).fit(z).astype(np.float32)

    if os.path.exists(OUT_TSNE_H5):
        print(f"[!] Removing existing: {OUT_TSNE_H5}")
        os.remove(OUT_TSNE_H5)

    print(f"[i] Writing t-SNE → {OUT_TSNE_H5}")
    with h5py.File(OUT_TSNE_H5, "w") as h5:
        _create_ds(h5, "tsne_cosine",    tsne_results["cosine"].shape)[:]    = tsne_results["cosine"]
        _create_ds(h5, "tsne_euclidean", tsne_results["euclidean"].shape)[:] = tsne_results["euclidean"]
        # filenames
        str_dt = h5py.string_dtype(encoding="utf-8")
        d = h5.create_dataset("filenames", shape=(N,), dtype=str_dt)
        d[:] = np.array(filenames, dtype=object)

    mins = (time.time() - t0) / 60.0
    print(f"[✓] Done. Wrote:\n  {OUT_PCA_H5}\n  {OUT_UMAP_H5}\n  {OUT_TSNE_H5}\n[~] Elapsed: {mins:.1f} min")


if __name__ == "__main__":
    main()
```

## Anomaly detection

```{code} python
:label: mse-per-image

#!/usr/bin/env python3
import os, csv, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, AveragePooling2D, Flatten, Dense, Reshape, ReLU
from tensorflow.keras.optimizers import Adam

# ────────────────────────────────────────────────────────────────────────
# 0) Paths & knobs  ← adjust if needed
# ────────────────────────────────────────────────────────────────────────
shards_dir            = r'D:\Results\TFRecords'        # your uint8 TFRecords folder
results_dir           = r'D:/Results/09052025_AE1M_Conv2DTranspose'  # where weights were saved
encoder_weights_path  = os.path.join(results_dir, 'encoder_weights.h5')
decoder_weights_path  = os.path.join(results_dir, 'decoder_weights.h5')
output_csv            = os.path.join(results_dir, 'mse_per_image.csv')

# TFRecord settings
tfrecord_compression  = None   # set to 'GZIP' if shards were gzipped
BATCH_SIZE            = 32
NUM_PARALLEL_READS    = 4
PREFETCH_BUFS         = 1

# Model settings (must match training)
img_h, img_w, img_c   = 256, 256, 1
latent_dim            = 512
learning_rate         = 3e-4

# ────────────────────────────────────────────────────────────────────────
# 1) GPU memory growth (safe on CPU too)
# ────────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ────────────────────────────────────────────────────────────────────────
# 2) Collect shards (no shuffle for deterministic pass)
# ────────────────────────────────────────────────────────────────────────
all_shards = sorted([
    os.path.join(shards_dir, f)
    for f in os.listdir(shards_dir)
    if f.endswith('.tfrecords')
])
if not all_shards:
    raise FileNotFoundError(f"No .tfrecords found in: {shards_dir}")

print(f"[Info] Found {len(all_shards)} shards")

# ────────────────────────────────────────────────────────────────────────
# 3) TFRecord parsing (uint8 → float32 [0,1], keep filename alongside)
# ────────────────────────────────────────────────────────────────────────
feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'filename':  tf.io.FixedLenFeature([], tf.string),
}

def _parse_for_inference(example_proto):
    data = tf.io.parse_single_example(example_proto, feature_description)
    img  = tf.io.decode_raw(data['image_raw'], tf.uint8)
    img  = tf.reshape(img, [img_h, img_w, img_c])
    img  = tf.image.convert_image_dtype(img, tf.float32)  # [0,255] → [0,1]
    fname = data['filename']
    return img, fname

def make_infer_dataset(shard_list):
    ds = tf.data.TFRecordDataset(
        shard_list,
        num_parallel_reads=NUM_PARALLEL_READS,
        compression_type=tfrecord_compression
    )
    ds = ds.map(_parse_for_inference, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(PREFETCH_BUFS)
    return ds

infer_ds = make_infer_dataset(all_shards)

# ────────────────────────────────────────────────────────────────────────
# 4) Rebuild model exactly as trained
# ────────────────────────────────────────────────────────────────────────
inp = Input((img_h, img_w, img_c), name='encoder_input')
x = inp
for filters in [16, 32, 64, 128]:
    x = Conv2D(filters, 3, padding='same')(x); x = ReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x); x = ReLU()(x)
    x = AveragePooling2D(pool_size=2)(x)  # same as (2, 2)

flat = Flatten()(x)
z = Dense(latent_dim, name='z')(flat)
encoder = Model(inp, z, name='encoder')

latent_in = Input((latent_dim,), name='z_sampling')
x = Dense(16 * 16 * 128)(latent_in)
x = Reshape((16, 16, 128))(x)
for filters in [128, 64, 32, 16]:
    x = Conv2DTranspose(filters, 3, strides=2, padding='same')(x); x = ReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x); x = ReLU()(x)

decoded = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='decoder_output')(x)
decoder = Model(latent_in, decoded, name='decoder')

ae = Model(inp, decoder(encoder(inp)), name='autoencoder')
ae.compile(optimizer=Adam(learning_rate), loss='mse', metrics=['mse'])  # compile not strictly needed for predict

# Load weights
if not (os.path.isfile(encoder_weights_path) and os.path.isfile(decoder_weights_path)):
    raise FileNotFoundError("Encoder/decoder weights not found. Check paths:\n"
                            f"  {encoder_weights_path}\n  {decoder_weights_path}")
encoder.load_weights(encoder_weights_path)
decoder.load_weights(decoder_weights_path)
print("[Info] Weights loaded.")

# ────────────────────────────────────────────────────────────────────────
# 5) Iterate, predict, compute per-image MSE, write CSV
# ────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
total = 0
start = time.time()

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'mse'])  # header

    for step, (imgs, fnames) in enumerate(infer_ds, start=1):
        # Forward pass
        recons = ae.predict(imgs, batch_size=BATCH_SIZE, verbose=0)
        # Per-image MSE over (H,W,C) in [0,1] domain (matches training loss)
        mse_batch = tf.reduce_mean(tf.square(imgs - recons), axis=[1, 2, 3]).numpy()

        # Decode filenames and write rows
        fn_batch = [b.decode('utf-8') for b in fnames.numpy().tolist()]
        for name, mse in zip(fn_batch, mse_batch):
            writer.writerow([name, f"{mse:.8f}"])

        total += len(fn_batch)
        if step % 200 == 0:  # progress every ~200 batches
            elapsed = time.time() - start
            print(f"[{step:>6} batches] {total} images processed | {elapsed/60:.1f} min elapsed")

elapsed = time.time() - start
print(f"[Done] Wrote per-image MSE for {total} images → {output_csv}  |  {elapsed/60:.1f} min")
```
