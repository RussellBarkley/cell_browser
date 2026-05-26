---
title: Approach
numbering:
  enumerator: 1.%s
---

Already adopted by some clinicians and educators, virtual microscopy could also benefit researchers as an interface for communicating and interpreting image data.
Indeed, interactive microscopy maps were first proposed for precisely this purpose, as an unbiased way to present electron microscopy data [@doi:10.1083/jcb.201201140].
Free and open-source solutions [@doi:10.1242/jcs.262198] and recent advances in file formats [@doi:10.1007/s00418-023-02209-1] now make this feasible, yet these tools have not been adopted in scientific publishing.
Our approach combined automated image acquisition, multi-scale [pyramid](<wiki:Pyramid_(image_processing)>) formats and embedded interactive viewers ([](#abstract)).

```{figure} ./figures/Virtual_microscopy.png
:label: abstract
Virtual microscopy. 1. Automated dataset collection with any suitable microscope. ZDC: Z-drift compensation. 2. Stitch tiles then convert the image to DZI or OME-Zarr, host online. 3. Image tiles or data chunks are retrieved from cloud storage as readers interact with microscopy maps.
```

Elemental Microscopy supports two interactive viewers; [Vizarr](https://github.com/hms-dbmi/vizarr) [@doi:10.1038/s41592-022-01482-7] and [OpenSeadragon](https://openseadragon.github.io/), enabling two methods of virtual microscopy based on OME-Zarr or Deep Zoom. 
Authors can also build custom viewers. 
Brodrick developed [microATLAS](https://github.com/LadInTheLab/microATLAS-widget), which was used to view OME-Zarr data in this article. 
All three viewers are readily embeddable in Elemental Microscopy. 
For Vizarr and OpenSeadragon, authors simply copy the markdown and replace the source URL with a link to their own data. 
For microATLAS, authors are encouraged to use the [widget builder](https://ladinthelab.github.io/microATLAS-widget/builder.html) to generate code.

## OME-Zarr

OME-Zarr is a standardized and flexible image format developed for use cases like virtual microscopy [@doi:10.1007/s00418-023-02209-1].
Most native microscopy file formats can be converted to OME-Zarr with preserved metadata using [Bio-Formats](https://www.openmicroscopy.org/bio-formats/).
It is a chunked file type that can support multiple resolution levels.
As an advanced implementation of microscopy maps [@doi:10.1242/jcs.262198], pyramidal OME-Zarr data can be viewed online [@doi:10.1038/s41592-022-01482-7].

### Vizarr

```{code}
:label: vizarr
:caption: Minimal markdown to embed Vizarr in Elemental Microscopy. Replace the source with your data.
:::{any:bundle} https://curvenote.github.io/widgets/widgets/vizarr-viewer.js
{
  "source": "https://your-bucket.s3.amazonaws.com/sample.zarr/",
  "height": "600px"
}
:::
```

### microATLAS

```{code}
:label: microatlas
:caption: Example markdown to embed microATLAS in Elemental Microscopy. Create your own with the [builder](https://ladinthelab.github.io/microATLAS-widget/builder.html).
:::{any:bundle} https://ladinthelab.github.io/microATLAS-widget/widget.js
{
  "source": "https://your-bucket.s3.amazonaws.com/sample.zarr/",
  "views": [
    {
      "name": "Overview",
      "zoom": -4,
      "target": [4886, 4886, 0]
    }
  ],
  "annotations": [
    { "name": "Region A", "target": [2400, 2400], "color": [100, 200, 255] }
  ],
  "scaleBar": { "maxWidth": 120, "position": "bottom-right" }
}
:::
```

```{figure} ./thumbnail.gif
:label: demo
Interacting with microATLAS: 
Use the menu (☰, top-left corner) to change the displayed channels, colours, or contrast.
Scroll (↕) to zoom into cells of interest.
```

## Deep Zoom

[Deep Zoom](<wiki:Deep_Zoom>) is a free and open-source solution, described elsewhere [@doi:10.1242/jcs.262198], now embeddable in Elemental Microscopy.
Pyramidal Deep Zoom image (DZI) data hosted on Github Pages is streamed to the [OpenSeadragon](https://openseadragon.github.io/) viewer.
By default, readers cannot control dimensions like channels or time, so it is best suited for single-channel or composite images.

### OpenSeadragon

```{code}
:label: openseadragon
:caption: Minimal markdown to embed OpenSeadragon in Elemental Microscopy. Replace the source with your DZI.
:::{any:bundle} https://curvenote.github.io/widgets/widgets/openseadragon.mjs
{
    "tileSources": "https://your-github.github.io/your-repository/sample.dzi"
}
:::
```

:::{dropdown} Imaging
Data collection was fully-motorized for unbiased sampling of large panoramas from replicate immunofluorescence experiments.
Grids of overlapping tiles were acquired with a robotic XY stage (IX3-SSU) and Z-drift compensation (IX3-ZDC).
Coverslips were imaged with an Olympus Fluoview FV3000 confocal microscope using a 100X objective lens (Olympus model UPLAPO100X, NA 1.50) in immersion oil. 
Pinholes in each channel were maximized to capture the widest depth of field.
:::