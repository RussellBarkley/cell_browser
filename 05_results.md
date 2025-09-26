---
Chapter 5
---

# Results

The dataset of grayscale images was encoded using the saved encoder weights into latent vectors ({ref}`embed_latents`), which are compressed representations of grayscale images in the lower-dimensional bottleneck of the autoencoder. The purpose of this analysis was not to interpret the latent space for scientific inquiry or biological discovery. Rather, we visualized the latent space using interactive figures to assess the structure of the learned representation fit by the vanilla autoencoder model.

## Latent space _interpolation_

A common method to evaluate the quality of latent space is interpolation, whereby mixing codes in latent space and decoding the result creates a semantically meaningful combination of the datapoints [@doi:10.48550/arXiv.1807.07543]. Interpolating with an autoencoder describes the process of using the decoder to decode a convex combination of two latent vectors [@doi:10.48550/arXiv.1807.07543]. A high-quality interpolation should have two characteristics: intermediate points along the interpolation should resemble real data and they should provide a semantically meaningful transition between the endpoints [@doi:10.48550/arXiv.1807.07543]. Interpolating between any two latent vectors of embedded nucleus images produced reasonable intermediate reconstructions from the decoder with a smooth transition between endpoints ([Figure 8b](#fig8b)). This result is consistent with the literature describing smooth interpolations with base model autoencoders [@doi:10.48550/arXiv.1807.07543]. The authors noted that intermediate points did not always resemble real data, which was true with the nucleus interpolations. For example, [unrealistic](#unrealistic_interpolation) samples were observed in interpolations between latent vectors of nuclei with different size or image features.

:::{figure} #nucleusnet10k-interpolation
:label: fig8b
:placeholder: ./figures/fig8b.png
:enumerator: 8b
Decoded latent vectors along intermediate points of interpolations between random pairs of images. Executing the code will randomly draw 1,000 images from the dataset. Shuffle 100 random pairs for browsing.
:::

## Two-dimensional projections

Dimensionality reduction techniques like t-SNE by {cite}`JMLR:v9:vandermaaten08a`, UMAP [@doi:10.21105/joss.00861] and PCA [@doi:10.1007/b98835] are frequently used to visualize the latent space of machine learning models like autoencoders [@doi:10.1111/cgf.13672]. These plots are analagous to map projections that transform the round surface of the earth onto a flat map, there is no single perfect projection. Like map projections, latent space projections can emphasize or hide certain characteristics. We pre-computed t-SNE and UMAP embeddings with default settings and PCA initialization [@doi:10.1038/nbt.4314]. An embedding was made for two distance metrics; Euclidean distance and cosine similarity, and ten PCA embeddings compared every combination of the top five prinipal components ({ref}`embeddings`). This was meant to allow for interaction to show parameters can affect the projections ([Figure 8c](#fig8c), [Figure 8e](#fig8e) and [Figure 8g](#fig8g)). We also found it helpful to present the data as image thumbnails, so we made static figures that plotted 10,000 random image thumbnails ([Figure 8d](#fig8d), [Figure 8f](#fig8f), [Figure 8h](#fig8h)) which are best viewed as EMMA maps [@doi:10.1242/jcs.262198]. Consider that only one percent of the dataset was shown in these thumbnail projections.

### t-SNE

:::{figure} #ae1m-tsne
:label: fig8c
:placeholder: ./figures/fig8c.png
:enumerator: 8c
Figure legend.
:::

```{figure} ./figures/fig8d.png
:name: fig8d
:align: center
:width: 100%
:enumerator: 8d
Figure legend.
```

### UMAP

:::{figure} #ae1m-umap
:label: fig8e
:placeholder: ./figures/fig8e.png
:enumerator: 8e
Figure legend.
:::

```{figure} ./figures/fig8f.png
:label: fig8f
:align: center
:width: 100%
:enumerator: 8f
Figure legend.
```

### PCA

:::{figure} #ae1m-pca
:label: fig8g
:placeholder: ./figures/fig8g.png
:enumerator: 8g
Figure legend.
:::

```{figure} ./figures/fig8h.png
:name: fig8h
:align: center
:width: 100%
:enumerator: 8h
Figure legend.
```

---

# Autoencoder-based determination of representative microscopy images

## 1. Computation of a theoretical image

We calculated theoretical latent vectors using measures of central tendency like the arithmetic mean, median and geometric median in latent space. Averaged latent vectors were reconstructed with the decoder to synthesize theoretical representative images of the nucleus ([Figure 9a](#fig9a)). Theoretical representative images do not necessarily look like real data [@doi:10.1109/BIP60195.2023.10379342], so these reconstructions do not represent real nuclei, but they possess some peculiar characteristics. Notably, what appeared to be background signal around the nucleus is apparent with the dynamic range around 0-12, which was a characteristic of the dataset noted earlier ([Figure 6c](#fig6c)). Otherwise, theoretical average nuclei appeared to blend image features and resembled the nucleus of a cell in interphase with little detail or variance like the spots vacated by nucleoli.

:::{figure} #ae1m-theoretical
:label: fig9a
:placeholder: ./figures/fig9a.png
:enumerator: 9a
Example widget.
:::

## 2. Determination of a prototypical image

Using the methods of information retrieval shown in Figure 7B, we encoded the theoretical latent vector reconstructions then calculated distance metrics that defined each practical image vector with respect to the theoretical image vectors (Figure 8B). Practical representative images are defined by a low euclidean distance or high cosine similarity to the theoretical representative image in latent space.

:::{figure} #ae1m-practical
:label: fig9b
:placeholder: ./figures/fig9b.png
:enumerator: 9b
Example widget.
:::

---