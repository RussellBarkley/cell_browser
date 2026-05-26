---
title: Video
numbering:
  enumerator: 9.%s
label : video_page
---

::::{figure}
:label: figure9
:::{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/video.zarr",
  "width": "500px",
  "height": "500px",
  "title": {
    "text": "mNeonGreen::μNS + mCherry-KDEL",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Play by default",
      "zoom": -3.784,
      "target": [
        3431,
        3431,
        0
      ],
      "playback": {
        "playing": true,
        "fps": 24,
        "startFrame": 0
      },
      "default": true
    },
    {
      "name": "mCherry-KDEL artifacts",
      "zoom": 0.106,
      "target": [
        3718,
        5505,
        0
      ],
      "playback": {
        "playing": true,
        "fps": 24,
        "startFrame": 0
      }
    }
  ],
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 10,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
:::
Cotransfection of mNeonGreen::μNS and mCherry-KDEL in live cells.
::::

:::{dropdown} Methods
HeLa cells were seeded on 35 mm glass-bottom dishes (20 mm coverslips, no. 1.5 glass). 
pCI-neo mNeonGreen::μNS(1-721) and mCherry-KDEL plasmids were co-transfected (1:1 ratio, 1.25 µg each) for 24 hours with Lipofectamine 3000. 
The stage was heated to 37°C but lacked CO₂, so the media was supplemented with 25 mM HEPES buffer before imaging. 
A 7×7 grid of 49 fields was imaged in a serpentine pattern at 4-minute intervals for thirty frames (~2 hours total).
:::