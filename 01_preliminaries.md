---
Chapter 4
---

# Definitions

A _grayscale image_ is defined as a matrix $M*N$ dimensions where each pixel is a single intensity value ranging from $0-1$ that represents the amount of light or intensity information at a specific point [@doi:10.1109/BIP60195.2023.10379342]. 

A _latent vector_ $z$ is a _n_-dimensional vector in the bottleneck of an autoencoder model that is the compressed representation of an image.

---

# Representative images from the MNIST collection

## Literature reproduction

Recently, objective representative image selection was demonstrated with real-world data including the [MNIST database](<wiki:MNIST_database>). [@doi:10.1109/BIP60195.2023.10379342] The authors proposed a two-step approach for objective representative image selection. First, theoretical representative images were calculated using measures of central tendency, then representative practical images were selected from the dataset based on their proximity to the theoretical images in vector space using SVD.

The MNIST dataset consists of 70,000 images of handwritten numbers that were manually annotated into ten classes corresponding to the digits 0-9. Soto-Quiros et al. tested their approach to representative image selection on a sub-set of n=720 images labelled "four". The three chosen measures of central tendency were the [arithmetic mean](#equation_A) and [median](#equation_B) which computed the average value for each pixel independently, as well as the [geometric median](#equation_C). We reproduced the theoretical and practical representative images of the MNIST digit "four" (Figures 2A and 2B) and found the results agree with the literature. (see Figures 2 and 4 in [@doi:10.1109/BIP60195.2023.10379342]).

:::{figure} #fig:mnist-four-theory
:name: fig2A
:placeholder: ./figures/fig2a.png
Example widget.
:::

:::{figure} #fig:mnist-four-practical
:name: fig2B
:placeholder: ./figures/fig2b.png
Example widget.
:::

---

## Proposed method

### Autoencoder-based representative image selection
Our approach adapted that of Soto-Quiros et al. for objective representative image selection. First, theoretical average latent vectors were calculated using measures of central tendency. Then, latent vectors of images from the dataset were ranked based on their distance to the theoretical average to find practical representative images. 

### 1. Calculate of a theoretical representative image

We trained a convolutional autoencoder model on the MNIST dataset. The source code for the model is here: {ref}`mnist-AE`. The encoder and decoder weights were saved after n=250 epochs of training. n=___ images labelled "four" were vectorized using the encoder weights, then we calculated the [arithmetic mean](#equation_H), [median](#equation_I) and [geometric median](#equation_K) of the latent vectors. These vectors were reconstructed using the decoder weights to synthesize theoretical representative images of the digit "four" (Figure 3A). 

:::{figure} #fig:mnist-ae-four-theory
:name: fig3A
:placeholder: ./figures/fig3a.png
Example widget.
:::

### 2. Define a practical representative image

The behaviour of the theoretical image generally does not correspond to a distinct image, therefore it is not considered the final representative image [@doi:10.1109/BIP60195.2023.10379342]. We encoded the theoretical average images to define practical representative images based on distance metrics in Euclidean space. The chosen distance metrics were [euclidean distance](#equation_L) and [cosine similarity](#equation_M), where representative images were defined as having the lowest euclidean distance, or highest cosine similarity to the theoretical images (Figure 3B).

:::{figure} #fig:mnist-ae-four-practical
:name: fig3B
:placeholder: ./figures/fig3b.png
Example widget.
:::

---

### Centroid vectors of the MNIST dataset (unsure about this section)

These demonstrations on a subset of the MNIST dataset relied on the label information to average grayscale images or latent vectors. Only the label "four" was shown to be consistent with the literature. Here, we show the reconstructed theoretical representative images for all ten labels using the autoencoder-based method, as well as the overall centroid that averaged all of the latent vectors in the dataset (Figure 3C).

:::{figure} #fig:mnist-ae-labels
:name: fig3C
:placeholder: ./figures/fig3c.png
Example widget.
:::

The reconstructed image of the overall centroid did not resemble any digit and appears to blend all image features. Practical representative images were defined using distance metrics that defined each image with respect to the theoretical average images using three measures of central tendency (Figure 3D). This shows that measures of central tendency can be calculated in the latent space of an autoencoder to generate theoretical representative images that resemble the associated label. The overall centroid vector failed to reconstruct a meaningful image of a digit, which made the selection of a practical image to represent the MNIST dataset questionable.