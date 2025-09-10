---
Chapter 3
---

# 1,000,000 high magnification images of the cell nucleus

We generated a large collection of un-annotated single-cell microscopy images of DAPI-stained cell nuclei that were fixed on glass coverslips. This dataset is the sum of one-hundred automated imaging experiments that sampled roughly ten-thousand unique nuclei per coverslip. N=100 coverslips of CV-1 cells were prepared from N=21 passages of the same cell line. The cells were seeded at varying densities then were fixed after at least one day of incubation. As expected, we observed variation of cell confluence and signal intensity between coverslips, and even between regions of the same coverslip (Figure 4A). Of the 1600 stitched images, we show average examples with respect to the number of masked nuclei and their mean signal intensity (Figure 4B).

:::{figure} #fig:scatter-plot
:name: fig4A
:placeholder: ./figures/fig4a.png
Example widget.
:::

:::{figure} #fig:stitched-image
:name: fig4B
:placeholder: ./figures/fig4b.png
Example widget.
:::

The sum of the dimensions of the 1600 stitched images is 241,083,171,855 px². At 8.0453 pixels per micron, this would be roughly 37.25cm². A square with an equivalent area would have a side length of 6.1cm, which is around two-thirds the surface area of a 10cm cell culture dish. The dimensions of the stitched images vary slightly because the MIST plugin corrected the position of tiles, but the average side length of a stitched image is ~1.5mm and the area is ~2.3mm².

Nuclei were segmented in the stitched images using cellpose [@doi:10.1038/s41592-020-01018-x], which produced mask files that were used to generate the single-cell dataset. We used cellpose because it can separate cotacting nuclei while also being able to mask mitotic figures that are apart. A random draw of 10,000 images, or roughly 1% of the dataset, is made independently for each reader in Figure 4C.

:::{figure} #fig:cell-browser
:name: fig4C
:placeholder: ./figures/fig4c.png
Example widget.
:::



