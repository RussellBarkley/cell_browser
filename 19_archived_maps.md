---
title: Archived Maps
numbering:
  enumerator: 17.%s
label : archival_page
---

We were interested in the finding that reovirus μNS colocalized with the ER [@doi:10.1128/mBio.01253-18], so we performed experiments with moxGFP::μNS and mCherry-ER reporters.
The results were unclear, so this research direction was abandoned.
Fortunately, overlapping tiles were recorded in an experiment dated _September 14 2022_.
This data was stitched and converted to OME-Zarr for reinterpretation as interactive microscopy maps (Figure 17.1).

:::::{figure}
:label: figure17
::::{grid} 2 2 2 2
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/archival/FLclimp.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "moxGFP::μNS(1-721) + mCherry-Climp63",
    "position": "top-center",
    "margin": 12,
    "fontSize": 14,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/archival/FLsec61b.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "moxGFP::μNS(1-721) + mCherry-Sec61β",
    "position": "top-center",
    "margin": 12,
    "fontSize": 14,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/archival/MRclimp.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "moxGFP::μNS(472-721) + mCherry-Climp63",
    "position": "top-center",
    "margin": 12,
    "fontSize": 14,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/archival/MRsec61b.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "moxGFP::μNS(472-721) + mCherry-Sec61β",
    "position": "top-center",
    "margin": 12,
    "fontSize": 14,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
::::
**Figure 17.1:** Reinterpretation of archived μNS–ER colocalization data as interactive microscopy maps.