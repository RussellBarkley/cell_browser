---
Chapter 2
---

# Materials and Methods

## Cell culture

The CV-1 cell line is derived from the kidney of an African green monkey. We cultured CV-1 cells in Earle's minimal essential medium supplemented with fetal bovine serum (10% volume/volume) at 37 degrees celsius with 5 percent environmental carbon dioxide to buffer pH. At each passage, six glass-bottom cell culture dishes with a 20mm diameter coverslip (Mattek Corporation, P35G-1.5-20-C) were seeded with varied densities of CV-1 cells (1e4-5e4 cells/well) to promote heterogeneous cell confluence.

## Sample preparation

The samples were fixed with 4% paraformaldehyde for twenty minutes after one to three days of incubation based on confluence. After fixation, the samples were washed with phosphate buffered saline (PBS) then were refrigerated at 4 degrees celsius for short-term storage. Nuclei were labelled with Nucblue Fixed Cell ReadyProbes Reagent (DAPI) (ThermoFisher Scientific, R37606) in 1 milliliter of PBS for at least 30 minutes then were imaged in the staining solution at room temperature.

## Image acquisition

We sampled large regions of no. 1.5 glass coverslips using a motorized stage on a laser scanning confocal microscope (Olympus Fluoview FV3000) with an oil immersion objective lens at 100X magnification (Olympus, model UPLAPO100X, NA 1.50). The pinhole diameter was opened to 800μm to maximize the depth of field. N=2500 fields were imaged in a 50x50 tile grid (snake pattern by rows: right down left down) with 10% overlap. Z-drift compensation was active to maintain focus and the confocal microscope was on a pressurized CleanBench isolation laboratory table (TMC Vibration Control) to dampen vibrations.

```{list-table} Illumination and image processing controls.
:label: table1
:enumerator: 1
* - Excitation wavelength
  - 405nm
* - Emission wavelength
  - 418nm-496nm
* - Laser intensity
  - 0.12%-0.15%
* - Detector sensitivity
  - 500V
* - Gain
  - 1.0X
* - Offset
  - 3%
* - Field resolution
  - 1024x1024 pixels
* - Pixel size
  - 8.0453 pixels/micrometer
* - Pixel type
  - uint16
* - Bits per pixel
  - 12
```

---

# Data pre-processing

```{figure} ./figures/fig4.png
:label: fig4
:align: center
:width: 100%
:enumerator: 4

An overview of data collection and pre-processing. A) Tiles were imaged in a 50x50 grid. B) Then stitched into 25x25 tile grids (quarters). C) Quarters were cropped into four stitched images.
```

## 1) Conversion to 8-bit .TIF format

The raw 12-bit files are in Olympus' .oir file format. The raw .oir files were converted to 8-bit .TIF format using FIJI [@doi:10.1038/nmeth.2019] and were renamed 0-2500.tif corresponding to the order that the tiles were imaged.

## 2) Tiles stitched with Microscopy Image Stitching Tool (MIST)

The size of a full stitched image approaches the maximum value that a 32-bit integer can hold, therefore the N=2500 .tif files were divided into sub-folders containing n=625 images corresponding to four quarters ({ref}`sort-quarters`) ([Figure 4b](#fig4)). The images in these sub-folders were stitched into 25x25 tile panoramas using the MIST plugin [@doi:10.1038/s41598-017-04567-y] in FIJI. MIST was useful to correct for inaccurate step sizes that sometimes occurred with the motorized stage.

```{list-table} MIST plugin settings.
:label: table2
:enumerator: 2
* - Filename pattern type
  - Sequential
* - Starting point
  - Upper left
* - Direction
  - Horizontal continuous
* - Grid width
  - 25
* - Grid height
  - 25
* - Grid start tile
  - 0
* - Timeslices
  - 0
* - Filename pattern
  - {pppp}.tif
* - Blending mode
  - Linear
* - Compression mode 
  - Uncompressed
* - Pixel size metadata
  - Mirometer X 8.0453 Y 8.0453
```

The stitched quarter was displayed and saved, named by run and position. For example, Run53TL was from the top left quarter of the 53rd imaging experiment. The stitched quarters were then cropped into four quarters ({ref}`crop-quarters`), yielding n=16 stitched images per coverslip and N=1600 stitched images from all one-hundred experiments.

## 3) Segmentation with cellpose

Nuclei were masked in the stitched images using a custom [cellpose](https://github.com/MouseLand/cellpose) model (CP_20250418_Nuclei_1Kmasks) [@doi:10.1038/s41592-020-01018-x]. We re-trained the nucleus model on N=125000 fields from our dataset, including n=1000 fields with manually-segmented nuclei. Advanced parameters in the graphical user interface were adjusted to flow_threshold: 0.5, cellprob_threshold: -2.0, diameter (pixels): 152.91. The cellpose model segmented the nuclei in all stitched images and the mask files were saved as .png files where each region of interest (ROI) is defined by a unique pixel value.

```{admonition} The data processing methods are biased
:class: warning
The detection of nuclei and the accuracy of the masks depended on the cellpose model. Our version of the cropped single-cell dataset was trained on fields including manual masks. Therefore, the population sampled in our single-cell dataset was defined by our semi-supervised cellpose model that was trained using manual masks. There are undesirable artifacts like blank images and inaccurate masks in the dataset, and sampling was likely inconsistent between stitched images due to variations in signal intensity. These limitations could be improved with better model training or more accurate manual segmentation within cellpose.
```

## 4) Cropped single cell masked dataset: NucleusNet

The orientation of a cell is known to confound the vector embedding of autoencoder models trained on single-cell microscopy data, motivating the development of orientation-invariant autoencoder models [@doi:10.1038/s41467-024-45362-4]. Similarly, a multi-encoder variational autoencoder model controlled for several transformational features like orientation that were 'uninformative' in single-cell analyses [@doi:10.1038/s42003-022-03218-x]. We [pre-aligned](https://github.com/jmhb0/o2vae/tree/master/prealignment) and center-cropped nuclei by fitting and rotating a minimal area rectangle to the cellpose mask ({ref}`crop-rois`). All values outside of the mask were set to zero in the cropped images. The dimensions of the uncompressed cropped images are 256x256 pixels, which is one-quarter the size of the 1024x1024 pixel microscopy fields.

---

# Deep zoom image display

Electron Microscopy Map (EMMA) is a visualization technique inspired by digital maps to present large high-resolution transmission electron microscopy data. EMMA uses an image tile pyramid to enable seamless transitions between magnifications, which was ideal to view the large image files that were produced in this study. See the primary literature for guidance on implementing the EMMA method [@doi:10.1242/jcs.262198]. We used free and open-source options, like [VIPS](https://www.libvips.org/) to generate the image tile pyramid and [OpenSeadragon](https://openseadragon.github.io/) v4.1.0 to view the zoomify tiles. OpenSeadragon v5.0.0+ had unstable performance on mobile devices.
