---
Chapter 5
---

# Results

We saved the latent space using {ref}`embed_latents`, which encoded the images into 512-dimensional latent vectors using the trained encoder weights. Each latent vector is a compressed representation of an original grayscale image in the bottleneck of the autoencoder.

## Latent space distribution

The distribution of values in each dimension is viewable in {Figure 6A}. Most dimensions are zero-centered gaussians, so the latent space 

:::{figure} #fig:ae1m-distribution
:name: fig6A
:placeholder: ./figures/fig6a.png
Example widget.
:::

## Latent space interpolation

A common method for evaluating the quality of latent space is with interpolation, whereby mixing codes in latent space and decoding the result creates a semantically meaningful combination of the datapoints [@doi:10.48550/arXiv.1807.07543]. Interpolating with an autoencoder describes the process of using the decoder to decode a convex combination of two latent vectors [@doi:10.48550/arXiv.1807.07543]. A high-quality interpolation should have two characteristics: intermediate points along the interpolation should resemble real data and they should provide a semantically meaningful transition between the endpoints [@doi:10.48550/arXiv.1807.07543].

Interpolating between any two vectors from the latent space produced reasonable intermediate reconstructions with a smooth transition between endpoints. This result is consistent with the literature that described smooth interpolations with base model autoencoders [@doi:10.48550/arXiv.1807.07543] and also observed that intermediate points do not always resemble real data. We see this result as well in the interpolations of the "vanilla" autoencoder trained on the nucleus dataset, especially between images with markedly different image features, where unrealistic samples are generated.

:::{figure} #fig:ae1m-interpolation
:name: fig6B
:placeholder: ./figures/fig6B.png
Example widget.
:::

## Latent space visualization

To visualize the latent space, we pre-computed dimensionality reduction techniques like t-SNE (cite), UMAP (cite) and PCA (cite) that can project the high dimensional space down to two dimensions. It is helpful to plot the datapoints as thumbnails to present the population variation of the dataset in latent space (cite). We applied t-SNE using {ref}`ae1m-tsne`, UMAP with {ref}`ae1m-umap` and PCA with {ref}`ae1m-pca` and saved the output for plotting in Figures 6C-6E. We implemented a toggle that will select 10,000 random images to visualize as a thumbnail.


so we implemented a toggle that will select 10,000 random images (or a subsample of ~1% of dataset) to visualize as a thumbnail.


The 512-dimensional latent space was projected down to two dimensions using t-SNE, UMAP, and PCA. The latent space is over 1,000,000 vectors which produced dense plots. It is helpful to see the thumbnail of a subset of the vectors to better understand how the plots distribute the images. Toggle the thumbnail overlay to view 10,000 random thumbnails, or roughly ~1% of the dataset, using your preferred two-dimensional projection of latent space.

### t-SNE

:::{figure} #fig:ae1m-t-SNE
:name: fig6C
:placeholder: ./figures/fig6C.png
Example widget.
:::

```{figure} ./figures/tsne_points_plus_10k_thumbs.png
:name: fig2
:align: center
:width: 100%

The method of automated data collection and representative image selection used in our fluorescence confocal microscopy experiments imaging the cell nucleus. A) 
```

### UMAP

:::{figure} #fig:AE1M-UMAP
:name: fig6D
:placeholder: ./figures/fig6D.png
Example widget.
:::

### PCA

:::{figure} #fig:AE1M-PCA
:name: fig6E
:placeholder: ./figures/fig6E.png
Example widget.
:::

---

## Applications

### Anomaly detection

Autoencoder-based anomaly detection is a common application that reconstructs images using a trained model to evaluate the distribution of reconstruction losses for the dataset. Images with MSE values at the extreme ends of the distribution are considered anomalous. We reconstructed the nucleus dataset with the trained model and saved the MSE loss for each image using {ref}`AE1M-MSE-loss`.

The images were sorted by MSE to define the 50,000 images with the highest or lowest reconstruction loss, which are the extreme five percentiles of the distribution. These anomalies, ordered by MSE, are shown in Figure 7A. We observed synthetic line artifacts produced by cellpose at the low end, as well as images with low signal intensity. On the high end are bright, large nuclei and a concentration of examples of the ___ phenotype (cite). This shows that data pre-processing artifacts and rare nucleus phenotypes can be found as anomalies using autoecoder-based methods that evaluate reconstruction loss.

:::{figure} #fig:AE1M-anomaly
:name: fig7A
:placeholder: ./figures/fig7A.png
Example widget.
:::

### Information retrieval

Another useful application for autoencoders is information storage and retrieval. In Figure 7B we implemented a tool where a random subsample of 1% the dataset is shown and individual images can be queried for a reverse image search. This works by encoding the query image with the trained model then the latent vector is compared to the saved latent space. Euclidean distance and cosine similarity were the two chosen distance metrics to define the query vector with respect to every image in the dataset including itself. Thus, we expect that the top result will be the query image which is indicated by a red outline in the plot. The images closest to the query image have semantic similarity, which is a notion that is problem-dependent and ill-defined [@doi:10.48550/arXiv.1807.07543]. Similar vectors in latent space share properties like orientation, shape, brightness, texture, size etc. even though these are features that were not pre-defined, which was a limitation to the objectivity of previous methods of microscopy image selection <https://doi.org/10.1016/s0006-3495(99)77379-0>.

:::{figure} #fig:AE1M-retrieval
:name: fig7B
:placeholder: ./figures/fig7B.png
Example widget.
:::

---

# Autoencoder-based determination of representative images

## Computation of a theoretical average vector

We calculated theoretical latent vectors at measures of central tendency like the [arithmetic mean](#equation_H), [median](#equation_I) and [geometric median](#equation_K) of the entire latent embedding. These averaged latent vectors were then reconstructed by the decoder to synthesize theoretical representative images of the nucleus (Figure 8A). Theoretical representative images do not necessarily look like real data [@doi:10.1109/BIP60195.2023.10379342], so these reconstructions do not represent real nuclei.

:::{figure} #fig:AE1M-theoretical
:name: fig8A
:placeholder: ./figures/fig8A.png
Example widget.
:::

## Determination of a prototypical vector

Using the methods of information retrieval shown in Figure 7B, we encoded the theoretical latent vector reconstructions then calculated distance metrics that defined each practical image vector with respect to the theoretical image vectors (Figure 8B). Practical representative images are defined by a low euclidean distance or high cosine similarity to the theoretical representative image in latent space.

:::{figure} #fig:AE1M-practical
:name: fig8B
:placeholder: ./figures/fig8B.png
Example widget.
:::

---