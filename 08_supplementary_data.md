---
title: Supplementary Data
numbering:
  enumerator: 9.%s
label : supplementary_page
---

# NucleusNet

:::{figure} #scatter-plot
:label: sfig1a
:placeholder: ./figures/sfig1a.png
The number and brightness of ROIs in the stitched images was measured using the associated cellpose mask files. A pre-computed centroid 694.5 ROIs and 79.0 mean intensity was marked.
:::

```{list-table} Representative stitched images ranked by distance to centroid: 694.5 ROIs, 79.0 mean intensity
:label: table3
:header-rows: 1

* - Filename
  - Number of ROIs
  - Mean intensity
  - Distance*
* - Run72TR_bottom_left
  - 685
  - 78.6
  - 0.046
* - Run63TR_top_right
  - 709
  - 78.6
  - 0.066
* - Run72BR_top_right
  - 687
  - 80.3
  - 0.074
* - Run68TR_top_right
  - 711
  - 79.6
  - 0.076
* - Run25BR_top_right
  - 698
  - 77.5
  - 0.079
* - Run25BR_bottom_left
  - 681
  - 78.0
  - 0.080
* - Run72BR_top_left
  - 712
  - 79.5
  - 0.080
* - Run102TR_top_left
  - 716
  - 78.3
  - 0.099
* - Run50TL_top_right
  - 714
  - 77.8
  - 0.104
* - Run72BL_top_right
  - 720
  - 79.1
  - 0.109
```
Z-score-normalized euclidean distance was used for equal weighting because the features had different scales.

# MNIST autoencoder

## Loss plot

:::{figure} #sfig2a_data
:label: sfig2a
:placeholder: ./figures/sfig2a.png
Autoencoder reconstruction loss over one-hundred epochs of training on the MNIST dataset.
:::

# Replication study

Barkley noted the type of data in each figure and supplementary figure, excluding videos, in order of presentation.
For data type, there were four possible options; light microscopy image, electron microscopy image, correlative light-electron microscopy image, quantitation.
Quantitation was considered any analysis with numbers, while the other three data types were representative images.
Some figure panels had multiple images.
If the images were of different samples, then they were counted as unique images (>1 images).
Others were multi-channel or time-lapse images, which was considered one image with multiple frames.

```{list-table} Assessment of the types of data in the original study [@doi:10.1128/mBio.01253-18].
:label: table4
:header-rows: 1
* - Figure
  - Data type
  - Images
  - Frames
  - Frame type
* - 1A
  - Light microscopy image
  - 1
  - 1
  -
* - 1B
  - Light microscopy image
  - 1
  - 1
  -
* - 1C
  - Electron microscopy image
  - 1
  - 1
  -
* - 1D
  - Correlative light-electron microscopy image
  - 1
  - 1
  -
* - 1E
  - Correlative light-electron microscopy image
  - 1
  - 1
  -
* - 1F
  - Electron microscopy image
  - 1
  - 1
  -
* - S1A
  - Light microscopy image
  - 2
  - 1
  -
* - S1B
  - Electron microscopy image
  - 4
  - 1
  -
* - 2A
  - Electron microscopy image
  - 1
  - 1
  -
* - 2B
  - Electron microscopy image
  - 1
  - 1
  -
* - 2C
  - Electron microscopy image
  - 1
  - 1
  -
* - 2D
  - Electron microscopy image
  - 1
  - 1
  -
* - 2E
  - Electron microscopy image
  - 1
  - 1
  -
* - 2F
  - Electron microscopy image
  - 1
  - 1
  -
* - 2G
  - Electron microscopy image
  - 1
  - 1
  -
* - 2H
  - Electron microscopy image
  - 4
  - 1
  -
* - 3
  - Electron microscopy image
  - 1
  - 1
  -
* - 4A
  - Light microscopy image
  - 1
  - 4
  - Time
* - 4B
  - Light microscopy image
  - 1
  - 2
  - Channel
* - 4C
  - Light microscopy image
  - 1
  - 3
  - Channel
* - S2A
  - Light microscopy image
  - 1
  - 3
  - Time
* - S2B
  - Light microscopy image
  - 1
  - 3
  - Time
* - 5A
  - Light microscopy image
  - 4
  - 1
  -
* - 5B
  - Light microscopy image
  - 1
  - 3
  - Channel
* - 5C
  - Light microscopy image
  - 1
  - 3
  - Channel
* - 5D
  - Electron microscopy image
  - 1
  - 1
  -
* - 5E
  - Electron microscopy image
  - 1
  - 1
  -
* - S3A
  - Light microscopy image
  - 1
  - 3
  - Channel
* - S3B
  - Light microscopy image
  - 1
  - 3
  - Channel
* - S3C
  - Light microscopy image
  - 1
  - 3
  - Channel
* - S3D
  - Light microscopy image
  - 1
  - 3
  - Channel
* - S3E
  - Light microscopy image
  - 1
  - 3
  - Channel
* - S3F
  - Light microscopy image
  - 1
  - 3
  - Channel
* - S3G
  - Light microscopy image
  - 1
  - 3
  - Channel
* - S3H
  - Light microscopy image
  - 1
  - 3
  - Channel
* - S3I
  - Light microscopy image
  - 1
  - 3
  - Channel
* - 6A
  - Quantitation
  - 0
  - 0
  -
* - 6C
  - Quantitation
  - 0
  - 0
  -
```

# Stereotypical images from NucleusNet-10K

```{image} ./figures/anaphase.png
:label: anaphase
```
```{image} ./figures/telophase.png
:label: telophase
```

