---
title: Virus
numbering:
  enumerator: 4.%s
---

## Mammalian orthoreovirus strains

Parker described the morphology of reovirus strain Type 1 Lang (T1L) factories as filamentous and T3D factories as globular, and identified the M1(T1L)-P208S mutation that disrupted an interaction between μ2 and microtubules determining the filamentous phenotype [@doi:10.1128/JVI.76.9.4483-4496.2002]. 
The mBio study did not provide a rationale for using this M1(T1L)-P208S mutant, so wildtype T1L was used instead in the replication study. 
Wildtype T1L tends to form filamentous viral factories, whereas the M1(T1L)-P208S mutant would have formed globular factories. 
A stock of unpurified reovirus strain Type 3 Dearing (T3D), which forms globular viral factories, was included to compare these morphologies. 
Others have reported using MOI 100 due to the low permissivity of HeLa cells to reovirus [@doi:10.1128/jvi.00199-19], so ten-fold dilutions from MOI 0 to 100 were tested (Figure 4.1). 
Infected cells were visualized with α-μNS chicken serum and AlexaFluor647-conjugated secondary antibodies. 
Most HeLa cells were infected with reovirus T1L at MOI 100, so this concentration was used for replication experiments, an increase from MOI 1 used in the original study. 
At MOI 1, only a few infected cells were positively identified and pinned on the map.

:::::{figure}
:label: reovirus-moi-figure

::::{grid} 2 2 2 2
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/reovirus_moi/t1lmoi0.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "Reovirus T1L MOI 0",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjust contrast",
      "description": "Reovirus channel min 0 max 2000",
      "zoom": -4.928,
      "target": [
        4886,
        4886,
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
            255
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
            2000
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
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
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/reovirus_moi/t3dmoi0.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "Reovirus T3D MOI 0",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjust contrast",
      "description": "Reovirus channel min 0 max 2000",
      "zoom": -4.928,
      "target": [
        4886,
        4886,
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
            255
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
            2000
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
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
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/reovirus_moi/t1lmoi1.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "Reovirus T1L MOI 1",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjust contrast",
      "description": "Reovirus channel min 0 max 2000",
      "zoom": -4.928,
      "target": [
        4886,
        4886,
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
            255
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
            2000
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
    }
  ],
  "annotations": [
    {
      "name": "Infected cell",
      "target": [
        7471,
        983
      ],
      "color": [
        235,
        87,
        87
      ]
    },
    {
      "name": "Infected cell",
      "target": [
        7362,
        850
      ],
      "color": [
        235,
        87,
        87
      ]
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
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/reovirus_moi/t3dmoi1.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "Reovirus T3D MOI 1",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjust contrast",
      "description": "Reovirus channel min 0 max 2000",
      "zoom": -4.928,
      "target": [
        4886,
        4886,
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
            255
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
            2000
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
    }
  ],
  "annotations": [
    {
      "name": "Infected cell",
      "target": [
        6460,
        2626
      ],
      "color": [
        235,
        87,
        87
      ]
    },
    {
      "name": "Infected cell",
      "target": [
        5199,
        7702
      ],
      "color": [
        235,
        87,
        87
      ]
    },
    {
      "name": "Infected cell",
      "target": [
        8381,
        6188
      ],
      "color": [
        235,
        87,
        87
      ]
    },
    {
      "name": "Infected cell",
      "target": [
        9605,
        5645
      ],
      "color": [
        235,
        87,
        87
      ]
    },
    {
      "name": "Infected cell",
      "target": [
        2486,
        8493
      ],
      "color": [
        235,
        87,
        87
      ]
    },
    {
      "name": "Infected cell",
      "target": [
        7598,
        9699
      ],
      "color": [
        235,
        87,
        87
      ]
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
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/reovirus_moi/t1lmoi10.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "Reovirus T1L MOI 10",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjust contrast",
      "description": "Reovirus channel min 0 max 2000",
      "zoom": -4.928,
      "target": [
        4886,
        4886,
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
            255
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
            2000
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
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
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/reovirus_moi/t3dmoi10.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "Reovirus T3D MOI 10",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjust contrast",
      "description": "Reovirus channel min 0 max 2000",
      "zoom": -4.928,
      "target": [
        4886,
        4886,
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
            255
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
            2000
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
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
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/reovirus_moi/t1lmoi100.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "Reovirus T1L MOI 100",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjust contrast",
      "description": "Reovirus channel min 0 max 2000",
      "zoom": -4.928,
      "target": [
        4886,
        4886,
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
            255
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
            2000
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
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
```
```{any:bundle} https://LadInTheLab.github.io/microATLAS-widget/widget.js
{
  "source": "https://nyu1.osn.mghpcc.org/barkley-replication/reovirus_moi/t3dmoi100.zarr",
  "width": "323px",
  "height": "323px",
  "title": {
    "text": "Reovirus T3D MOI 100",
    "position": "bottom-left",
    "margin": 12,
    "fontSize": 20,
    "font": "Arial",
    "color": "rgba(255,255,255,0.95)",
    "style": "text"
  },
  "views": [
    {
      "name": "Adjust contrast",
      "description": "Reovirus channel min 0 max 1250",
      "zoom": -4.928,
      "target": [
        4886,
        4886,
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
            255
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
            1250
          ]
        ],
        "blendMode": "single",
        "colormap": "viridis"
      },
      "default": true
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
```
::::

**Figure 4.1:** Reovirus strains T1L (left) and T3D (right) 24 hours post infection, with increasing MOI from top to bottom (0, 1, 10, 100).
:::::


:::{tip}
Evaluate the calreticulin channel for reported reovirus-induced ER remodeling.
The uninfected control is MOI 0 and ER morphology can be compared within populations.
:::

:::{dropdown} Infection protocol
Stocks of purified reovirus T1L were titrated in L-929 (L) cells to estimate the concentration of plaque-forming units (PFU). 
L cells were also used for virus titration in the original study, so the calculation of multiplicity of infection (MOI, PFU/cell) was consistent between studies. 
For infection, purified reovirus was diluted in 100 µl of serum-free DMEM and adsorbed to cells for one hour at room temperature with agitation every ten minutes. 
Thereafter, pre-warmed serum-containing DMEM was added and samples were incubated for one day before fixation.
:::
