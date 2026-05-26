---
title: Introduction
numbering:
  enumerator: 0.%s
---

Representative microscopy images are visual communication tools used by microscopists to present research results to other scientists.
The earliest representative microscopy images were hand-drawn in Robert Hooke's [Micrographia](<wiki:Micrographia>) in 1665.
Today, roughly three-quarters of publications in biomedical journals report at least one microscopy image [@doi:10.7554/eLife.55133].
There is an unspoken expectation that scientists rely on these images to be honest and accurate.
However, microscopy image selection and quantitative analyses thereof are vulnerable to bias and deception [@doi:10.1242/jcs.261567].
Besides cherry-picking, scientists can sincerely misinterpret observations.
The problem is that readers find it difficult to critically evaluate representative images.
Furthermore, archival and review of primary microscopy data is disorganized and time-consuming, if at all available.
These problems can be addressed with novel visualization strategies like [virtual microscopy](<wiki:Virtual_microscopy>) that transmit entire interactive datasets over the internet.
An example was the unbiased presentation of electron microscopy data to readers as interactive digital maps — or microscopy maps [@doi:10.1083/jcb.201201140].
In practice, microscopy maps are easy to share but are accessed via external links [@doi:10.1242/jcs.262198].

[Reproducibility](<wiki:Reproducibility>) is an essential principle of the [scientific method](<wiki:Scientific_method>) for acquiring knowledge, yet there is a contentious [reproducibility crisis](<wiki:Replication_crisis>) in science.
From the perspective of biomedical researchers, the perceived cause is the pressure to publish [@doi:10.1371/journal.pbio.3002870].
Repeatability within the same study by the same researchers is distinguished from replicability by independent researchers using the same methodology.
Large-scale replicability studies found that only 11% [@doi:10.1038/483531a] and 25% [@doi:10.1038/nrd3439-c1] of key findings from preclinical cancer research papers were replicable.
The replication crisis is multifaceted, but the irreproducibility of representative images deserves scrutiny in its own right.
According to one study, replication rates for representative images ranged from 27% to 90% [@doi:10.7554/eLife.71601].
This wide range reflects how difficult it is to assess the replicability of representative images without dedicated tools.
A feature of virtual microscopy is the unbiased sampling and transmission of multi-scale image data, making it useful as a method to replicate representative images. 
If results are reproducible, original claims should be readily observable in microscopy maps, which capture a broader range of variation.
Here, microscopy maps were used to reassess the claim that mammalian orthoreovirus remodels the endoplasmic reticulum (ER) of cells during infection [@doi:10.1128/mBio.01253-18].
Using the protocols described in the original paper, hundreds of overlapping tiles were sampled at high-magnification from replicate immunofluorescence experiments on a fully motorized confocal microscope.
Microscopy maps were embedded in the article using the next generation file format OME-Zarr [@doi:10.1007/s00418-023-02209-1] and custom viewers based on Viv [@doi:10.1038/s41592-022-01482-7].
