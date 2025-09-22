---
Chapter 5
---

# Results

The dataset of grayscale images was encoded into 512-dimensional latent vectors using the saved encoder weights ({ref}`embed_latents`). Latent vectors are compressed representations of grayscale images in the lower-dimensional bottleneck of the autoencoder. The purpose of this analysis was not to interpret the latent space for scientific inquiry or biological discovery. Instead, we aimed to show the latent codes of an unsupervised autoencoder trained on the cropped nucleus dataset to assess the structure of latent space. Thereafter, we demonstrated an autoencoder-based method of objective representative image selection using centroids of this latent space.

## Latent space _distribution_

Per-dimension histograms show the distribution of values in latent space ([Figure 8a](#fig8a)). Most dimensions approximated zero-centered gaussian distributions, though there were several skewed distributions with value deviating from zero.

:::{figure} #ae1m-distribution
:label: fig8a
:placeholder: ./figures/fig8a.png
:enumerator: 8a
Histogram of N=1,071,277 latent vectors corresponding to grayscale images of cropped nuclei. Control the dimension or percentile and bins of the distribution shown with the sliders.
:::

## Latent space _interpolation_

A common method to evaluate the quality of latent space is interpolation, whereby mixing codes in latent space and decoding the result creates a semantically meaningful combination of the datapoints [@doi:10.48550/arXiv.1807.07543]. Interpolating with an autoencoder describes the process of using the decoder to decode a convex combination of two latent vectors [@doi:10.48550/arXiv.1807.07543]. A high-quality interpolation should have two characteristics: intermediate points along the interpolation should resemble real data and they should provide a semantically meaningful transition between the endpoints [@doi:10.48550/arXiv.1807.07543]. Interpolating between any two latent vectors of embedded nucleus images produced reasonable intermediate reconstructions from the decoder with a smooth transition between endpoints ([Figure 8b](#fig8b)). This result is consistent with the literature describing smooth interpolations with base model autoencoders [@doi:10.48550/arXiv.1807.07543]. The authors noted that intermediate points did not always resemble real data, which was true with the nucleus interpolations. For example, [unrealistic](#unrealistic_interpolation) samples were observed in interpolations between latent vectors of nuclei with different size or image features.

:::{figure} #ae1m-interpolation
:label: fig8b
:placeholder: ./figures/fig8b.png
:enumerator: 8b
Decoded latent vectors along intermediate points of interpolations between random pairs of images. Executing the code will randomly draw 1,000 images from the dataset. Shuffle 100 random pairs for browsing.
:::

## Latent space _visualization_

Dimensionality reduction techniques like t-SNE by {cite}`JMLR:v9:vandermaaten08a`, UMAP [@doi:10.21105/joss.00861] and PCA [@doi:10.1007/b98835] are common methods to visualize the high-dimensional latent space of machine learning models like autoencoders [@doi:10.1111/cgf.13672]. These plots are analagous to map projections that transform the round surface of the earth onto a flat map, there is no single perfect projection. Like map projections, latent space projections can emphasize or hide certain characteristics. We pre-computed t-SNE and UMAP embeddings with default settings and PCA initialization [@doi:10.1038/nbt.4314]. An embedding was made for two distance metrics; Euclidean distance and cosine similarity, and ten PCA embeddings compared every combination of the top five prinipal components ({ref}`embeddings`). This was meant to allow for interaction to show parameters can affect the projections ([Figure 8c](#fig8c), [Figure 8e](#fig8e) and [Figure 8g](#fig8g)). We also found it helpful to present the data as image thumbnails, so we made static figures that plotted 10,000 random image thumbnails ([Figure 8d](#fig8d), [Figure 8f](#fig8f), [Figure 8h](#fig8h)) which are best viewed as EMMA maps [@doi:10.1242/jcs.262198]. Consider that only one percent of the dataset was shown in these thumbnail projections.

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

## Applications

### Anomaly detection

Autoencoder-based anomaly detection is a common application that typically evaluates the distribution of MSE reconstruction loss and images at the extreme ends of the distribution are considered anomalous. We embedded and reconstructed the dataset with the trained model and saved the MSE loss for each image ({ref}`mse-per-image`). The data was sorted by reconstruction loss to identify 10,000 images in each of the one percentile ends of the MSE distribution, what may considered anomalous [Figure 9a](#fig9a). The low end of the distribution had hundreds of what appeared to be [blank images](#cellpose_artifact) until the dynamic range was reduced to 0-2 which revealed artifacts created by the cellpose segmentation model. Interestingly, these artifacts appeared as separate clusters in the t-SNE ([Figure 8d](#fig8d)) and UMAP ([Figure 8f](#fig8f)) plots. The bottom one percentile of the MSE distribution consisted of images with low brightness, and we noted many cells in the [metaphase](#metaphase) stage of mitosis. The upper one percentile of the MSE distribution contained images of large, bright nuclei, as well as an enrichment of cells in the [prophase](#prophase) stage of mitosis with condensing chromosomes. We considered cleaning the dataset to remove the top and bottom two-and-a-half percentiles, leaving the middle ninety-five percent of the MSE distribution, which would still be more than one million images. Removing outliers would remove cellpose artifacts and it could improve model training and analysis, but at the cost of removing rare biological phenotypes which was unacceptable. Altogether, this showed that artifacts and uncommon phenotypes were overrepresented in the tails of the MSE distribution, which demonstrated unsupervised autoencoder-based anomaly detection with the nucleus dataset.

:::{figure} #ae1m-MSE-distribution
:name: fig9a
:placeholder: ./figures/fig9a.png
:enumerator: 9a
Example widget.
:::

:::{figure} #ae1m-anomaly
:name: fig9b
:placeholder: ./figures/fig9b.png
:enumerator: 9b
Example widget.
:::

### Information retrieval

Another application for autoencoders is information storage and retrieval. In Figure 7B we implemented a tool where a random subsample of 1% the dataset is shown and images can be queried for a reverse image search using the latent space. This works by encoding the query image with the trained model then the latent vector is compared to the saved latent space. Euclidean distance and cosine similarity were the two chosen distance metrics to define the query vector with respect to every image in the dataset including itself. Thus, we expect that the top result will be the query image which is indicated by a red outline in the plot. The images closest to the query image have semantic similarity, which is a notion that is problem-dependent and ill-defined [@doi:10.48550/arXiv.1807.07543]. Similar vectors in latent space share properties like orientation, shape, brightness, texture, size etc. even though these are features that were not pre-defined, which was a limitation to the objectivity of previous methods of microscopy image selection <https://doi.org/10.1016/s0006-3495(99)77379-0>.

:::{figure} #fig:ae1m-reverse-image-search
:name: fig8C
:placeholder: ./figures/fig8c.png
Example widget.
:::

---

# Autoencoder-based determination of representative microscopy images

## 1. Computation of a theoretical image

We calculated theoretical latent vectors using measures of central tendency like the [arithmetic mean](#equation_H), [median](#equation_I) and [geometric median](#equation_K) in latent space. These averaged latent vectors were reconstructed with the decoder to synthesize theoretical representative images of the nucleus (Figure 8A). Theoretical representative images do not necessarily look like real data [@doi:10.1109/BIP60195.2023.10379342], so these reconstructions do not represent real nuclei, but they possess some peculiar characteristics. Notably, what appears to be background signal around the nucleus is apparent with the dynamic range around 0-12, which is a characteristic of the dataset that we noted earlier. Otherwise, the theoretical average nuclei appeared to be a blend of image features which resembled a nucleus in interphase, though there is little detail or variance like spots vacated by nucleoli.

:::{figure} #fig:ae1m-theoretical
:name: fig9A
:placeholder: ./figures/fig9a.png
Example widget.
:::

## 2. Determination of a prototypical image

Using the methods of information retrieval shown in Figure 7B, we encoded the theoretical latent vector reconstructions then calculated distance metrics that defined each practical image vector with respect to the theoretical image vectors (Figure 8B). Practical representative images are defined by a low euclidean distance or high cosine similarity to the theoretical representative image in latent space.

:::{figure} #fig:AE1M-practical
:name: fig9B
:placeholder: ./figures/fig.png
Example widget.
:::

---