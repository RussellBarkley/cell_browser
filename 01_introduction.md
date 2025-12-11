---
title: Introduction
numbering:
  enumerator: 1.%s
label : introduction_page
---

Representative microscopy images are visual communication tools used by microscopists to present the results of research to other scientists.
The earliest representative microscopy images were hand drawings in [Micrographia](<wiki:Micrographia>) in 1665 by Robert Hooke.
Nowadays, roughly three-quarters of publications in biomedical journals report at least one microscopy image [@doi:10.7554/eLife.55133].
In scientific journals, microscopists choose stereotypical images of phenomena to show in static figures.
An un-spoken rule in the scientific community is that representative images are trusted to be honest and accurate.
However, microscopy image selection and quantitative analyses thereof are vulnerable to bias and deception [@doi:10.1242/jcs.261567].
Aside from research integrity, scientists sincerely misinterpret observations.
The problem is that it is difficult for readers to critically evaluate representative images in static figures.
Furthermore, archival and review of primary microscopy data is disorganized and time-consuming if at all available.

Virtual nanoscopy was an innovative visualization strategy for unbiased presentation of transmission electron microscopy data [@doi:10.1083/jcb.201201140].
It was described as Google Earth for cell biologists, which was fitting because it used technology developed for satellite images.
With automated acquisition, high-magnification tiles are stitched and processed into a multi-resolution [pyramid](<wiki:Pyramid_(image_processing)>) for tile-based interactive viewers.
In practice, it is easy to implement [@doi:10.1242/jcs.262198] but difficult to publish due to a lack of support from scientific journals.
This project introduced embedded microscopy maps into the literature as a proof of principle to share immunofluorescence bioimaging data without external links or software.

Microscopists contribute to the [reproducibility crisis](<wiki:Replication_crisis>) in science, which is being adressed through protocol standardization [@doi:10.1111/jmi.13041].
From the perspective of most biomedical researchers, the perceived cause of the reproducibility crisis is the pressure to publish [@doi:10.1371/journal.pbio.3002870].
Large-scale replicability studies performed by scientists at Amgen and Bayer assessed that 11% [@doi:10.1038/483531a] and 25% [@doi:10.1038/nrd3439-c1] of key findings from preclinical cancer research papers were robust and reproducible.
Another reproducibility project evaluated replication success with various criteria, though the 'lowest bar' determined a replication rate of 79% for representative images [@doi:10.7554/eLife.71601]. 
Notably, 26% of claimed effects were evidenced by a representative image without quantitative analysis.
Consider that representative images cannot describe population variation and the uncertainty of an observation.

A study critical of the reproducibility of representative microscopy images focused on methods of objective image selection to limit bias (https://doi.org/10.1016/s0006-3495(99)77379-0).
The authors implemented a web server that chose typical images from uploaded data, but it is now unsupported and there is no modern equivalent.
Our work started by revisiting this idea of objective image selection, as Barkley and Parker thought it was an interesting use case for artificial intelligence.
Barkley automated the process of sample collection and image selection from a collection of one million single-cell images of the [cell nucleus](<wiki:Cell_nucleus>), called _NucleusNet_.
Images were presented in interactive figures, including embedded maps based on [Deep Zoom](<wiki:Deep_Zoom>) and [OpenSeadragon](https://openseadragon.github.io/) which was free and open-source [@doi:10.1242/jcs.262198].
Representative images were then selected by the classic [autoencoder](<wiki:Autoencoder>) model near the centroid of the embedding.
Autoencoders are unsupervised deep learning models that compress and reconstruct images through a vector bottleneck referred to as _latent space_.
The structure of latent space is a [black box](<wiki:Black_box>), though it can be shaped to be more useful with the art of representation learning [@doi:10.1109/TPAMI.2013.50].
Barkley found that it would be irresponsible to use an uninterpretable embedding for the task of image selection. 
To define what is representative, both the subjective determinations made by humans and the objective determinations made by autoencoders are opaque.
Perhaps the limitation is that static figures require image selection.
As a proof of principle, the projected shifted towards demonstrations of virtual nanoscopy as an alternative to representative image selection.
Another application for virtual nanoscopy was to test the replicability of representative images.
To show this, Barkley challenged the claim of reovirus-induced endoplasmic reticulum (ER) organelle remodelling.
[OrganelleBox](https://organellebox.sf.czbiohub.org/), was used for cross-validation and to expand the investigation to study sixteen organelles during reovirus infection.

---

## Contributions

1. **A single-cell confocal microscopy image dataset of 1,061,277 unique cell nuclei.**

2. **Demonstration of a novel use case for autoencoders: typical image selection.**

3. **Embedded microscopy maps for unbiased data sharing with virtual nanoscopy**

4. **An image atlas of cellular organelles during mammalian orthoreovirus infection.**

---

## Definitions

_Virtual nanoscopy_ was a term used to describe the process of automated image acquisition, followed by unbiased presentation in an interactive tile-based viewer. [@doi:10.1083/jcb.201201140]

_Microscopy map_ is the interactive figure used to view virtual nanoscopy data. [@doi:10.1083/jcb.201201140]
The original term '_electron_ microscopy map' was redefined to fit a broader use case because our study used confocal microscopy data.

A _representative image_ is defined as a visual communication tool used by microscopists to communicate research findings to other scientists.
This definition is agnostic to the research context and it emphasizes the use of pictures as scientific evidence.
Parker suggested that _stereotypical image_ would be an apt name for the current use of representative images.

A _grayscale image_ is defined as a two-dimensional matrix where each pixel is a single intensity value ranging from $0-1$ that represents the amount of light or intensity information at a specific point [@doi:10.1109/BIP60195.2023.10379342].

A _latent space_ is a collection of vectors that form a reduced-dimensionality embedding of the data, fit by a machine learning model [@doi:10.1111/cgf.13672].
