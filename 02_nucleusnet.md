---
title: NucleusNet
numbering:
  enumerator: 1.%s
label : nucleusnet_page
---

# NucleusNet: one million single-cell microscopy images

## 1600 tiled panoramas

Artificial intelligence models train better with more data, which motivated Barkley to create a large collection of 256*256 pixel uint-8 TIF files of the [cell nucleus](<wiki:Cell_nucleus>), referred to as NucleusNet.
NucleusNet is the sum of one-hundred automated imaging experiments that sampled around 10,000 nuclei per coverslip at high-magnification with a motorized confocal microscope.
This dataset consists of 1,061,277 single-cell images extracted from 1600 stitched panoramas covering an area of roughly 37.25cm² of cultured cells.
Images of cell monolayers were collected over twenty-one passages of the same cell line.
[CV-1 cells](https://www.atcc.org/products/ccl-70) were seeded at varying densities then were fixed in paraformaldehyde (PFA) after at least one day of incubation, so the populations had asynchronous cell cycles.
The nuclei in fixed permeabilized cells were stained with DAPI which binds to DNA.
As expected, we observed variation in cell confluence and signal intensity between experiments, and even between regions of the same coverslip ([](#sfig1a)).
Ten stitched images closest to the centroid were selected to represent those with an average number and brightness of ROIs ([](#table1)).

## 1,061,277 single-cell images

Nuclei were masked in the 1600 stitched images with a custom [cellpose](https://github.com/MouseLand/cellpose) segmentation model [@doi:10.1038/s41592-020-01018-x] that output masks used to generate the single-cell dataset.
Cellpose was suitable for segmentation because cell division created instances where two nuclear bodies were separated in space but should count as one ROI.
Specifically, we considered pairs of mitotic figures in [anaphase](#anaphase) or [telophase](#telophase) to be one ROI and we preferred if [micronuclei](<wiki:Micronucleus>) and nuclei were masked together.
There were also instances where two or more distinct nuclei made contact but should be segmented as separate masks.
Barkley assessed that Cellpose handled these challenges well after tuning their model to our data which included manual masks.
Browse the ten representative stitched images ([](#table3)) from [](#sfig1a) to evaluate the cellpose segmentation masks as an overlay.
Unique masks were assigned random colours to help differentiate ROIs.
False-positive detections by cellpose were rare but expected as a consequence of high-throughput automation.

**Figure 1A: Embed OpenSeadragon viewer with this [image](https://russellbarkley.github.io/cellpose_masks/)**

## NucleusNet-10K

A small version (N=10,000) of the single-cell dataset was created with one-hundred random images chosen from each imaging session.
NucleusNet-10K was deposited to Hugging Face Datasets to stream examples from the cloud to this cell browser [](#fig1b).
Barkley assessed that nuclei were properly centered and the cellpose segmentation model and orientation pre-alignment were satisfactory.
Pixels were set to zero outside of the mask in the cropped images, leaving background signal around the nucleus.
This observation was notable because it emerged later in our reconstructions from the latent space of the autoencoder.
Otherwise, the cropped images in our version of NucleusNet were not processed further.

:::{figure} #nucleusnet10k_cellbrowser
:label: fig1b
:placeholder: ./figures/cellbrowser.png
Figure legend.
:::

With collections of images in the literature, it is common to see several random examples presented in a static figure.
Barkley fused NucleusNet-10K into a 100x100 tile grid and embedded the stitched image (Figure 1C).
Experiments were reported in chronological order by row from top to bottom, and it is clear that there was variation in signal intensity between experiments.
This observation warrants image pre-processing techniques that normalize or standardize pixel intensities.
Interestingly, there is another [example](https://www.allencell.org/deep-cell-zoom.html) of this visualization strategy used to present 200,000 single-cell images of human-induced pluripotent stem cells.

**Figure 1C: Embed OpenSeadragon viewer with NucleusNet-10K**