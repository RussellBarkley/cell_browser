---
title: Plasmids
numbering:
  enumerator: 3.%s
---

## mCherry-ER reporters

mCherry-KDEL was the same construct used in the original study; mCherry-ER-3 plasmid (55041, Addgene).
mCherry-CytERM-N-17 (55024, Addgene) was included as a positive control for ER remodelling because organized smooth endoplasmic reticulum (OSER) whorls form through fluorescent protein dimerization [@doi:10.1083/jcb.200306020].
An [OSER assay](#oser_page) (Figure 8.1) was performed to compare whorl formation across different fluorescent proteins relative to mCherry-CytERM.
Two additional constructs were included with ER proteins fused to mCherry; mCh-Climp63 (136293, Addgene) and mCh-Sec61β (49155, Addgene).
An earlier experiment combining moxGFP::μNS with these constructs was inconclusive (Figure 17.1).

Cells transiently expressing mCherry-ER fusions were fixed 24 hours after transfection with TransIT-2020 and Lipofectamine 3000.
Parker identified saturated signal in the original experiment and advised Barkley to repeat it, adjusting illumination settings individually.  
The corrected maps are presented below (Figure 3.1) and the [saturated maps](#saturated_page) (Figure 15.1) and [discarded maps](#discarded_page) (Figures 16.1 to 16.5) are presented in the Supplemental.

:::::{figure}
:label: figure3
::::{grid} 2 2 2 2
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/plasmid_repeat/lf3000_kdel_Cycle_02.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mCherry-KDEL (Lipofectamine)",
    "position": "top-center",
    "margin": 12,
    "fontSize": 16,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "ER remodeling",
      "zoom": -0.798,
      "target": [
        11053,
        10028,
        0
      ],
      "appearance": {
        "channelsVisible": [
          false,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            4095
          ],
          [
            0,
            4095
          ],
          [
            0,
            2208
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      }
    }
  ],
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 12,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/plasmid_repeat/transit_KDEL_Cycle_02.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mCherry-KDEL (Trans-IT 2020)",
    "position": "top-center",
    "margin": 12,
    "fontSize": 16,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "ER remodeling",
      "zoom": -1.066,
      "target": [
        9952,
        424,
        0
      ],
      "appearance": {
        "channelsVisible": [
          false,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            4095
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      }
    }
  ],
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 12,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/plasmid_repeat/lf3000_cytER_Cycle.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mCherry-CytERM (Lipofectamine)",
    "position": "top-center",
    "margin": 12,
    "fontSize": 16,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjusted contrast",
      "description": "Calreticulin channel",
      "zoom": -5.512,
      "target": [
        7323,
        7323,
        0
      ],
      "appearance": {
        "channelsVisible": [
          true,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            3500
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
    },
    {
      "name": "ER remodeling",
      "zoom": -0.51,
      "target": [
        4018,
        9351,
        0
      ],
      "appearance": {
        "channelsVisible": [
          false,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            4095
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      }
    }
  ],
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 12,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/plasmid_repeat/transit_cyter_Cycle_03.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mCherry-CytERM (Trans-IT 2020)",
    "position": "top-center",
    "margin": 12,
    "fontSize": 16,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjusted contrast",
      "description": "Calreticulin channel",
      "zoom": -5.512,
      "target": [
        7323,
        7323,
        0
      ],
      "appearance": {
        "channelsVisible": [
          true,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            3500
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
    },
    {
      "name": "ER remodeling",
      "zoom": -0.51,
      "target": [
        10587,
        9413,
        0
      ],
      "appearance": {
        "channelsVisible": [
          false,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            4095
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      }
    }
  ],
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 12,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/plasmid_repeat/lf3000_sec61_Cycle_02.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mCherry-Sec61β (Lipofectamine)",
    "position": "top-center",
    "margin": 12,
    "fontSize": 16,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjusted contrast",
      "description": "Calreticulin channel",
      "zoom": -5.512,
      "target": [
        7323,
        7323,
        0
      ],
      "appearance": {
        "channelsVisible": [
          true,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            3000
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
    },
    {
      "name": "ER remodeling",
      "zoom": -0.51,
      "target": [
        11070,
        3818,
        0
      ],
      "appearance": {
        "channelsVisible": [
          false,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            4095
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      }
    }
  ],
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 12,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/plasmid_repeat/transit_sec_Cycle_01.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mCherry-Sec61β (Trans-IT 2020)",
    "position": "top-center",
    "margin": 12,
    "fontSize": 16,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "ER remodeling",
      "zoom": -0.51,
      "target": [
        9912,
        1666,
        0
      ],
      "appearance": {
        "channelsVisible": [
          false,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            4095
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      }
    }
  ],
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 12,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/plasmid_repeat/lf3000_climp_Cycle.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mCherry-Climp63 (Lipofectamine)",
    "position": "top-center",
    "margin": 12,
    "fontSize": 16,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "ER remodeling",
      "zoom": -1.621,
      "target": [
        11242,
        13229,
        0
      ],
      "appearance": {
        "channelsVisible": [
          false,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            4095
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      }
    }
  ],
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 12,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/plasmid_repeat/transit_climp_Cycle_01.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "mCherry-Climp63 (TransIT-2020)",
    "position": "top-center",
    "margin": 12,
    "fontSize": 16,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "ER remodeling",
      "zoom": -0.332,
      "target": [
        6544,
        6534,
        0
      ],
      "appearance": {
        "channelsVisible": [
          false,
          true,
          true
        ],
        "channelColors": [
          [
            0,
            255,
            0
          ],
          [
            0,
            0,
            255
          ],
          [
            255,
            0,
            0
          ]
        ],
        "contrastLimits": [
          [
            0,
            4095
          ],
          [
            0,
            4095
          ],
          [
            0,
            4095
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      }
    }
  ],
  "scaleBar": {
    "maxWidth": 100,
    "position": "bottom-right",
    "fontSize": 12,
    "font": "Arial",
    "color": "rgba(255,255,255,0.9)"
  }
}
```
::::
**Figure 3.1:** Cells expressing mCherry-ER fusion proteins. Select the 'ER remodeling' view from the menu to see representative fields of punctate structures suspected to be mCherry artifacts.
:::::

:::{important} Alternative hypothesis
For each of the four mCherry-ER reporters, at least one cell exhibits what Barkley interprets as ER remodeling (see the preset views). 
These artifacts were also observed in live cells ([](#figure9)), indicating that they are not a consequence of fixation. 
Since these cells were not infected, it is possible that previous investigators misinterpreted mCherry-ER reporter artifacts as an outcome of reovirus infection.
:::

:::{dropdown} Transfection protocol
3 × 10⁵ HeLa cells were seeded on 35 mm glass-bottom dishes (Mattek, P35G-1.5-20-C). 
Lipofectamine 3000 (L3000015, Invitrogen) volumes were based on those for 6-well plates: 125 µl Opti-MEM (31985-062, Gibco) was mixed with 5 µl P3000 reagent and 2.5 µg DNA, and a separate tube of 125 µl Opti-MEM was mixed with 3.75 µl Lipofectamine 3000 reagent. 
The tubes were combined, incubated for 20 minutes, then added dropwise to the samples. 
TransIT-2020 reagent (MIR 5400, Mirus) volumes were based on those for 6-well plates: 250 µl Opti-MEM was mixed with 2.5 µg DNA and mixed gently, then 7.5 µl TransIT-2020 was added and mixed, followed by 20 minutes of incubation and dropwise addition to the samples.
:::