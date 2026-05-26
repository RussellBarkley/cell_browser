---
title: OSER Assay
numbering:
  enumerator: 8.%s
label : oser_page
---

Fluorescent proteins tend to oligomerize when confined to the ER, which can restructure the organelle into organized smooth ER (OSER) whorls [@doi:10.1083/jcb.200306020]. 
This phenomenon is used to assess the propensity of fluorescent proteins to oligomerize [@doi:10.1111/j.1600-0854.2012.01336.x]. 
Six fluorescent proteins were tested in an OSER assay using interactive microscopy maps (Figure 8.1).
Four of these can be cross-referenced with published estimates of the percentage of cells lacking visible OSER whorls: mEGFP (98.8±1.2), mCherry (95.0±0.8), FusionRed (91.5±3.0), and mNeonGreen (90.4±2.1) [@doi:10.1038/nmeth.3891].
Among the samples tested here, Dendra2 formed the most prominent whorls, visible in the majority of cells.

:::::{figure}
:label: figure8
::::{grid} 2 2 2 2
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/oser_assay/fusionred.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "FusionRed-CytERM",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 50,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/oser_assay/mcherry.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mCherry-CytERM",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 50,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/oser_assay/dendra2.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "Dendra2-CytERM",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 50,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/oser_assay/megfp.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mEGFP-CytERM",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 50,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/oser_assay/mneongreen.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mNeonGreen-CytERM",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 50,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/oser_assay/moxgfp.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "moxGFP-CytERM",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 50,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
::::
**Figure 8.1:** Qualitative OSER assay presented as interactive microscopy maps.
:::::

:::{dropdown} Methods
3 × 10⁵ HeLa cells were seeded on 35 mm glass-bottom dishes. 
An OSER assay was performed with six fluorescent proteins cloned into the CytERM plasmid. 
2.5 µg of each plasmid was transfected with Lipofectamine 3000 for 24 hours before PFA fixation and automated imaging of 15×15 (225 field) grids with a 100X objective lens. 
Illumination settings, including excitation and emission wavelengths, were adjusted individually. 
mEGFP, moxGFP, Dendra2, and mNeonGreen shared the same wavelengths.
:::