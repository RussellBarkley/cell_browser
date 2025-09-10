---
Chapter 9
---
# MNIST example
## Literature reproduction
```{code} python
:label: lit-repro
:caption: Sorting images into four quarters and renaming them to stitch 25x25 tile grids.

```
# Data collection
## 1. Sort tiles into quarters
```{code} python
:label: sort-quarters
:caption: Sorting images into four quarters and renaming them to stitch 25x25 tile grids.
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
:caption: Cropping the stitched quarters into quarters, resulting in sixteen stitched images per coverslip.

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
:caption: 

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

## Sub-sample 50,000 images

```{code} python
:label: fivepercentsample
:caption: 

#!/usr/bin/env python3
"""
Randomly sample 50,000 TIFF images from a very large folder (streamed, memory-safe),
save the selected source paths to a .txt file, and copy them to a new folder.

Source:  D:\\Confocal_imaging_nuclei_tif\\MIST_Fused_Images\\ROIs
Target:  D:\\FivePercentSample
List:    D:\\FivePercentSample\\sample_paths.txt

- Uses reservoir sampling (no giant file list in RAM).
- Filters for .tif/.tiff (case-insensitive).
- Saves the chosen file paths to a .txt file (one per line).
- Avoids filename collisions in the destination by appending __N.
- Optional: set RANDOM_SEED for reproducibility.
"""

import os
import random
import shutil
from typing import Iterable, List, Optional

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = r"D:\Confocal_imaging_nuclei_tif\MIST_Fused_Images\ROIs"
OUTPUT_DIR  = r"D:\FivePercentSample"
PATHS_TXT   = r"D:\FivePercentSample\sample_paths.txt"
SAMPLE_SIZE = 50_000
RANDOM_SEED: Optional[int] = 42  # set to None for non-deterministic sampling
RECURSIVE   = False              # set True if you want to include subfolders
# ──────────────────────────────────────────────────────────────────────────────


def iter_image_paths(root: str, recursive: bool = False) -> Iterable[str]:
    """Yield full paths to .tif/.tiff files under root (optionally recursive)."""
    exts = {".tif", ".tiff"}
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in exts:
                    yield os.path.join(dirpath, name)
    else:
        with os.scandir(root) as it:
            for entry in it:
                if entry.is_file():
                    if os.path.splitext(entry.name)[1].lower() in exts:
                        yield entry.path


def reservoir_sample(paths: Iterable[str], k: int, seed: Optional[int] = None) -> List[str]:
    """
    Reservoir sample k items from an iterator of unknown/huge length.
    Returns a list of selected paths (len <= k if fewer available).
    """
    rng = random.Random(seed) if seed is not None else random
    reservoir: List[str] = []
    n = 0

    for p in paths:
        if n < k:
            reservoir.append(p)
        else:
            j = rng.randint(0, n)  # inclusive
            if j < k:
                reservoir[j] = p
        n += 1
        if n % 200_000 == 0:
            print(f"[SCAN] Seen {n:,} files so far...")

    print(f"[DONE SCAN] Total images seen: {n:,}")
    if n < k:
        print(f"[WARN] Only {n:,} images available; sampling {n:,} instead of {k:,}.")
    return reservoir


def safe_copy(src: str, dst_dir: str) -> str:
    """
    Copy src into dst_dir. If a file with the same name exists, append __N before the extension.
    Returns the final destination path.
    """
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    dst = os.path.join(dst_dir, base)

    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        return dst

    # Collision: append __1, __2, ...
    n = 1
    while True:
        candidate = os.path.join(dst_dir, f"{name}__{n}{ext}")
        if not os.path.exists(candidate):
            shutil.copy2(src, candidate)
            return candidate
        n += 1


def main():
    print("[START] Streaming and sampling...")
    sample = reservoir_sample(iter_image_paths(DATASET_DIR, RECURSIVE), SAMPLE_SIZE, RANDOM_SEED)
    print(f"[SAMPLE] Selected {len(sample):,} images.")

    # Ensure output directory exists before writing the list
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write the selected source paths to a .txt file (one per line)
    with open(PATHS_TXT, "w", encoding="utf-8") as f:
        for p in sample:
            f.write(p + "\n")
    print(f"[WRITE] Wrote selected paths to: {PATHS_TXT}")

    # Copy files
    print(f"[COPY] Beginning copy to: {OUTPUT_DIR}")
    for i, src in enumerate(sample, start=1):
        safe_copy(src, OUTPUT_DIR)
        if i % 1_000 == 0:
            print(f"[COPY] {i:,}/{len(sample):,} copied...")

    print(f"[DONE] Copied {len(sample):,} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

# Nucleus Dataset

## AE1M

```{code} python
:label: AE1M
:caption: AE trained on the nucleus dataset

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

## Embed latents

```{code} python
:label: embed_latents
:caption: 

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