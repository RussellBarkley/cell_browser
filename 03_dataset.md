---
Chapter 3
---

# NucleusNet: a microscopy image collection of cell nuclei

```{figure} ./figures/fig5.png
:label: fig5
:align: center
:width: 100%
:enumerator: 5
A stitched image (25x25 tile grid) of the bottom left quarter of experiment 25. [Click here](https://russellbarkley.github.io/full_panorama/) or scan the QR code to browse a zoomable stitched image (50x50 tile grid) using the EMMA method.
```

## 1600 stitched images

We generated a large dataset of un-labelled single-cell confocal microscopy images of DAPI-stained CV-1 cell nuclei fixed on glass coverslips. It is the sum of one-hundred automated imaging experiments that sampled around ten-thousand nuclei per coverslip. [Figure 5](#fig5) illustrates one quarter of the sampled area from one experiment. Scan the QR code or [click here](https://russellbarkley.github.io/full_panorama/) to browse an expanded view [@doi:10.1242/jcs.262198] of the experiment that shows the full recorded area with a side length of 50 microscope fields. Images of cell monolayers were collected over twenty-one passages of the same CV-1 cell culture. The cells were seeded at varying densities then were fixed in paraformaldehyde after at least one day of incubation, so the populations had an asynchronous cell cycle. As expected, we observed variation in cell confluence and signal intensity between experiments, and even between regions of the same coverslip ([Figure 6a](#fig6a)). NucleusNet consists of 1,061,277 single-cell images extracted from 1600 stitched panoramas covering an area of roughly 37.25cm². Ten stitched images closest to the centroid in ([Figure 6a](#fig6a)) were selected to represent those with an average number and brightness of ROIs ([Table 3](#table3)). 

:::{figure} #scatter-plot
:label: fig6a
:placeholder: ./figures/fig6a.png
:enumerator: 6a
The number and brightness of ROIs in the stitched images was measured using the associated cellpose mask files ({ref}`scatter-code`). A pre-computed centroid 694.5 ROIs and 79.0 mean intensity was marked.
:::

```{list-table} Representative stitched images ranked by distance to centroid: 694.5 ROIs, 79.0 mean intensity
:label: table3
:header-rows: 1
:enumerator: 3
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

Z-score-normalized euclidean distance for equal weighting because the features had different scales ({ref}`ten-stitched-images`).*

## 1,061,277 cropped images

Nuclei were masked in the stitched images with a custom [cellpose](https://github.com/MouseLand/cellpose) segmentation model [@doi:10.1038/s41592-020-01018-x] that output mask files used to generate the single-cell dataset ({ref}`crop-rois`). Cellpose was suitable to mask the nuclei because cell division created instances where two nuclear bodies were separate on the coverslip but count as one ROI. Specifically, we considered pairs of mitotic figures in [anaphase](#anaphase) or [telophase](#telophase) to be one ROI and we preferred if [micronuclei](<wiki:Micronucleus>) and nuclei were masked together. There were also instances where two or more distinct nuclei made contact and should be separate masks. Cellpose handled these challenges after tuning the model to our data. [Click here](https://russellbarkley.github.io/cellpose_masks/) to browse the ten representative stitched images ([Table 3](#table3)) and overlay the cellpose masks to evaluate our segmentation model. Unique masks were assigned random colours to help differentiate ROIs in the overlay. [Figure 6b](#fig6b) shows one of these ten representative stitched images, specifically Run72TR_bottom_left which was closest to the centroid in ([Table 3](#table3)). This particular example had an unslightly foreign object on the coverslip that was masked by cellpose. False-positive detections by cellpose were rare but expected as a consequence of high-throughput automation.
 
```{figure} ./figures/fig6b.png
:label: fig6b
:align: center
:width: 100%
:enumerator: 6b
Ten representative stitched images with cellpose masks. Run72TR_bottom_left is shown in the static figure. [Click here](https://russellbarkley.github.io/cellpose_masks/) or scan the QR code to view these stitched images in a dynamic format using the EMMA method with a dynamic scalebar [@doi:10.1242/jcs.262198]. Toggle the overlay to evaluate the cellpose masks. Unique ROIs were randomly coloured. 
```

A random sample of 1,000 images is drawn from NucleusNet for each run of the interactive browser ([Figure 6c](#fig6c)). Nuclei were generally centered, well-segmented and pre-aligned. Pixels were set to zero outside of the mask in the cropped images, so there was background signal around each nucleus which became apparent by adjusting the image contrast. This observation was notable because it emerged later in our reconstructions from the latent space of an autoencoder.

:::{figure} #nucleusnet10k_cellbrowser
:label: fig6c
:placeholder: ./figures/fig6c.png
:enumerator: 6c
Figure legend.
:::

(data-availability)=
# Data availability:

NucleusNet was [deposited](#huggingface-upload) to Hugging Face to facilitate data sharing and the repository was streamed to present images in interactive figures. The full collection of 250,000 microscopy fields, 1,600 stitched images and corresponding mask files, as well as the cropped images were zipped and archived at Zenodo.