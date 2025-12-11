---
title: Materials and Methods
numbering:
  enumerator: 8.%s
label : methods_page
---

# NucleusNet

## Cell culture

The CV-1 cell line is fibroblast-like and derived from the kidney of an African green monkey.
CV-1 cells were cultured in Eagle's minimal essential medium supplemented with fetal bovine serum (10% volume/volume) at 37 degrees celsius with 5 percent environmental carbon dioxide to buffer pH.
At each passage, six glass-bottom cell culture dishes with a 20mm diameter coverslip (Mattek Corporation, P35G-1.5-20-C) were seeded with varied densities of CV-1 cells (~1e4-5e4 cells) to promote heterogeneous cell confluence.

## Sample preparation

The samples were fixed with 4% paraformaldehyde for twenty minutes after one to three days of incubation when the cells approached confluence.
After fixation, the samples were washed with phosphate buffered saline (PBS) then were refrigerated at 4 degrees celsius for storage.
Nuclei were labelled with Nucblue Fixed Cell ReadyProbes Reagent (DAPI) (ThermoFisher Scientific, R37606) in 1 milliliter of PBS for at least 30 minutes then were imaged in the staining solution at room temperature.

## Image acquisition

Large regions of no. 1.5 glass coverslips were sampled using a motorized stage (IX3-SSU) on a laser scanning confocal microscope (Olympus Fluoview FV3000) with an oil immersion objective lens at 100X magnification (Olympus, model UPLAPO100X, NA 1.50).
The pinhole diameter was opened to 800μm to maximize the depth of field.
N=2500 fields were imaged in a 50x50 tile grid (snake pattern by rows: right down left down) with 5% overlap at each edge.
Z-drift compensation was active to maintain autofocus.
The confocal microscope was on a pressurized CleanBench isolation laboratory table (TMC Vibration Control).

```{list-table} Illumination and image processing controls on the Olympus Fluoview FV3000 microscope.
:label: table1

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

## Data pre-processing

```{figure} ./figures/fig4.png
:label: fig4
:align: center
:width: 100%

An overview of data collection and pre-processing. 
A) Tiles were imaged in a 50x50 grid. 
B) Then stitched into 25x25 tile grids (quarters). 
C) Quarters were cropped into four stitched images.
```

1. Conversion to 8-bit .TIF format

Raw 12-bit files are in the proprietary Olympus .oir file format with metadata.
The raw were converted to 8-bit TIF format using FIJI [@doi:10.1038/nmeth.2019] and were renamed 0-2500.tif corresponding to the order that the tiles were imaged.

2. Tiles stitched with Microscopy Image Stitching Tool (MIST)

The size of a full stitched image approaches the maximum value that a 32-bit integer can hold, therefore the N=2500 .tif files were divided into sub-folders containing n=625 images corresponding to four quarters ([](#fig4)).
The images in these sub-folders were stitched into 25x25 tile panoramas using the MIST plugin [@doi:10.1038/s41598-017-04567-y] in FIJI.
MIST was useful to correct for inaccurate step sizes that sometimes occurred with the motorized stage.

```{list-table} MIST plugin settings.
:label: table2

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

The stitched quarter was displayed and saved, named by run and position.
For example, Run53TL was from the top left quarter of the 53rd imaging experiment.
The stitched quarters were then cropped into four quarters, yielding n=16 stitched images per coverslip ([](#fig4)) and N=1600 stitched images from all one-hundred experiments.

(cellpose-segmentation)=
3. Segmentation with cellpose

Nuclei were masked in the stitched images using a custom [cellpose](https://github.com/MouseLand/cellpose) model [@doi:10.1038/s41592-020-01018-x].
The pre-trained cellpose nucleus model was re-trained on N=125000 fields from NucleusNet, including n=1000 fields with manually-segmented nuclei.
Weights were saved as CP_20250418_Nuclei_1Kmasks.
Advanced parameters in the graphical user interface were adjusted to flow_threshold: 0.5, cellprob_threshold: -2.0, diameter (pixels): 152.91.
The cellpose model segmented the nuclei in all stitched images and the mask files were saved as PNG files where each region of interest (ROI) is defined by a unique pixel value.

```{admonition} Bias in NucleusNet from data processing methods
:class: warning
The detection of nuclei and the accuracy of the masks relied on the cellpose model. There are undesirable artifacts like blank images and inaccurate masks in the dataset, and sampling was likely inconsistent between stitched images due to variations in signal intensity. The cellpose model was re-trained and hand-tuned by a human. Training was also semi-supervised with manual masks. Thus, there are human and model biases.
```

4. Cropped single cell masked dataset

The orientation of a cell is known to confound the vector embedding of autoencoder models trained on single-cell microscopy data, motivating the development of orientation-invariant autoencoder models [@doi:10.1038/s41467-024-45362-4].
Similarly, a multi-encoder variational autoencoder model controlled for several transformational features like orientation that were 'uninformative' in single-cell analyses [@doi:10.1038/s42003-022-03218-x].
Barkley [pre-aligned](https://github.com/jmhb0/o2vae/tree/master/prealignment) and by fitting and rotating a minimal area rectangle to the cellpose mask.
Nuclei were center-cropped and all values outside of the mask were set to zero in the cropped images.

## Deep zoom microscopy maps

Microscopy maps are a novel visualization strategy to present large high-magnification microscopy images as zoomable digital maps.
There are free and open-source options to create microscopy maps [@doi:10.1242/jcs.262198].
[VIPS](https://www.libvips.org/) was used to generate the Deep Zoom image (.dzi) image tile pyramid and [OpenSeadragon](https://openseadragon.github.io/) v4.1.0 viewed the tiles.
OpenSeadragon v5.0.0+ had unstable performance on mobile devices.
Code and data was hosted on Github Pages.

# Replication Study

## Cell culture

HeLa CCL-2 cells were cultured in Dulbecco’s modification of Eagle’s Medium (DMEM) (50-003-PB, Corning) which was supplemented with 10% fetal bovine serum (FS-0500-AD, Atlas).
Antibiotics were not added to the medium in the replication study, although in the original study the authors used 100 U/ml penicillin G, 100ug/ml streptomycin and 0.25ug/ml amphotericin B.

## Immunofluorescence

Cells were seeded on 12mm diameter glass coverslips (1.0 glass) (1254580, Fisher) in 12-well plates (665180, Cellstar).
To fix the cells, media was aspirated and 4% PFA solution was applied without washing.
4% PFA was incubated for twenty minutes at room temperature, followed by three PBS washes.
After fixation the cells were permeabilized and blocked for twenty minutes in PBS with Triton X-100 and bovine serum albumin.
Antibodies were diluted in this buffer and were incubated for one hour at room temperature in the dark, both followed by three PBS washes.
Nuclei were labelled for 15 minutes in Nucblue Fixed Cell Stain ReadyProbes reagent (containing DAPI) (R37606, Fisher) diluted in PBS.
Coverslips were mounted (P36980, Fisher) on glass slides (1255015, Fisher) and cured overnight at room temperature in the dark.
Slides were gently cleaned then stored at four degrees celsius.

## Antibodies

The original study used a polyclonal serum for calreticulin but did not include the product number, so it was substituted for a monoclonal antibody.
Mouse calreticulin monoclonal antibody (MA5-15382, Thermo) was stored at -20C in 5ul aliquots to be used at 1:1000 dilution.
Chicken anti-muNS serum was previously made in the Parker lab (1:1000 dilution).
Guinea pig anti-muNS serum was made by the Stanifer lab (1:5000 dilution).
Donkey anti-mouse antibodies conjugated to AlexaFluor488 (715-545-150), donkey anti-chicken antibodies conjugated to AlexaFluor647 (703-605-155) and donkey anti-guinea pig conjugated to AlexaFluor594 (706-585-148) were stored diluted in 1:1 glycerol and were used at 1:800 dilutions.

## Transfection

Lipofectamine 3000 (L3000015, Invitrogen) was used with 12-well plates; 62.5ul Opti-MEM (31985-062, Gibco) was mixed with 2.5ul P3000 reagent and 1.25ug DNA, and another tube of 62.5ul Opti-MEM was mixed with 1.875ul lipofectamine 3000 reagent. The tubes were mixed, incubated for 20 minutes, then added dropwise to the samples.
TransIT-2020 Reagent (MIR 5400, Mirus) was used with 12-well plates; 125ul Opti-MEM was mixed with 1.25ug DNA, mixed gently, then 3.75ul TransIT-2020 was added and mixed, followed by 20 minutes of incubation and dropwise addition to the samples.

## Plasmids

The mCherry-KDEL construct was the same as the original study; mCherry-ER-3 plasmid (55041, Addgene), which retained mCherry in the lumen of the ER.
Barkley included mCherry-CytERM-N-17 (55024, Addgene) to retain mCherry on the cytoplasmic side of the ER.
mCherry-CytERM was considered a positive control for ER remodelling because of the propensity of fluorescent proteins to cause organized smooth endoplasmic reticulum (OSER) whorls [@doi:10.1083/jcb.200306020].
Two additional constructs were included with ER proteins fused to mCherry; mCh-Climp63 (136293, Addgene) and mCh-Sec61 beta (49155, Addgene).
mCherry-Climp63 was a control for ER remodelling, given its role in determining ER morphology [@doi:10.1016/j.cell.2010.11.007], while mCherry-Sec61β was considered a general ER marker [@doi:10.1126/science.aay7108].

## Virus infection

The study used a mutant mammalian orthoreovirus strain Type 1 Lang (T1L) M1-P208S, which makes globular viral factories through the loss of interaction with microtubules [@doi:10.1128/JVI.76.9.4483-4496.2002].
No rationale was provided for using the P208S mutant, nor did our lab have it, so wildtype mammalian orthoreovirus strains T1L and Type 3 Dearing (T3D) were used in the replication study.
The important distinction is that the morphology of viral factories between strains T1L and T3D differ.
T1L tends to form filamentous viral factories, and mutant reovirus M1-P208S used the T1L genetic backbone but formed globular viral factories.
A stock of unpurified reovirus T3D was used in one experiment to compare strains.
Stocks of purified reovirus T1L were titrated in L-929 cells to estimate the concentration of plaque-forming units (PFU).
L-929, or L cells are permissive to reovirus infection and were used for virus titration in the original study, so there was consistency in calculating the multiplicity of infection (MOI).
To clarify, MOI was PFU/cell, and PFU was estimated on L-929 cells in both the original study and the replication study.

For infection of cells on coverslips in 12-well plates, purified reovirus was diluted in a volume of 100ul of serum-free DMEM.
Media was aspirated from the cells, replaced with 100ul serum-free DMEM to wet the coverslip, then the virus-containing medium was added.
Reovirus was adsorbed to the cells for one hour at room temperature and the plate was rocked every ten minutes to redistribute the inoculum.
Thereafter, one milliliter of pre-warmed serum-containing DMEM was added to the cells, and the plate was incubated at 37C for 24h before PFA fixation.

## Imaging

Samples were imaged on an Olympus Fluoview FV3000 laser-scanning confocal microscope with a 100X objective lens using immersion oil. 
The pinhole was opened for all channels for the widest depth of field. 
Z-draft compensation was used to autofocus the samples during acquisition. 
Each channel was excited separately and care was taken to avoid spectral overlap. 
The coordinates for a 10x10 grid with 5% overlap was set and the one hundred fields were captured using the multi-area time lapse protocol.