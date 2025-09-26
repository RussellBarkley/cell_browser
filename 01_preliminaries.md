---
Chapter 4
---

# Definitions

A _grayscale image_ is defined as a matrix $M*N$ dimensions where each pixel is a single intensity value ranging from $0-1$ that represents the amount of light or intensity information at a specific point [@doi:10.1109/BIP60195.2023.10379342]. 

A _latent vector_ $z$ is a _n_-dimensional vector in the bottleneck of an autoencoder model between the encoder and decoder. It represents a compressed representation of an image.

A _latent space_ is a collection of latent vectors that form a reduced-dimensionality vector embedding of the data, fit by a machine learning model [@doi:10.1111/cgf.13672].

A _representative image_ is defined as an image with the overall quality and characteristics of the dataset [@doi:10.1109/BIP60195.2023.10379342].

---

# Representative images from the MNIST collection

## Literature reproduction

Objective representative image selection was recently demonstrated with real-world data including the [MNIST database](<wiki:MNIST_database>) [@doi:10.1109/BIP60195.2023.10379342]. The authors proposed a two-step approach to objective representative image selection. First, theoretical representative images were calculated using measures of central tendency, then representative images were selected from the dataset based on distance to theoretical images in vector space using SVD.

The MNIST dataset consists of 70,000 images of handwritten numbers that were manually annotated into ten classes corresponding to the digits 0-9. Soto-Quiros et al. tested their approach to representative image selection on a sub-set of n=720 images labelled "four". The three chosen measures of central tendency were the arithmetic mean and median which computed the average value per pixel independently, as well as the geometric median. We reproduced the theoretical average images of the MNIST digit "four" with all N=6824 images labelled "four" ([Figure 2a](#fig2a)) and we found the results to be consistent with the primary literature [@doi:10.1109/BIP60195.2023.10379342], though the exemplars were not the same ([Figure 2b](#fig2b)).

:::{figure} #fig2a_data
:label: fig2a
:placeholder: ./figures/fig2a.png
:enumerator: 2a
Computation of theoretical representative images of the MNIST digit '4'. N=6824 grayscale images with the label '4' were flattened to 784-dimensional vectors to compute then reshape reconstructed images of the arithmetic mean (left), median (middle) and geometric median (right).
:::

:::{figure} #fig2b_data
:label: fig2b
:placeholder: ./figures/fig2b.png
:enumerator: 2b
Computation of practical representative images of the MNIST digit '4' using the arithmetic mean (left), median (middle) and geometric median (right).
:::

---

## Proposed method of representative image selection

We adapted the two-step method to representative image selection [@doi:10.1109/BIP60195.2023.10379342] using the latent space of an autoencoder. First, theoretical average latent vectors were calculated using measures of central tendency, like the arithmetic mean, median and geometric median. Then, practical examples were determined in latent space by ranking latent vectors by distance to the calculated centroids.

### 1. Calculation of average latent vectors

We trained a convolutional autoencoder model ({ref}`mnist-AE`) on the MNIST dataset (Supplementary [Figure 1a](#sfig1a) and [Figure 1b](#sfig1b)) and saved the encoder and decoder weights after one-hundred training epochs and we encoded a latent space. 6824 latent vectors with the label "four" were averaged by arithmetic mean, median and geometric median, then the latent vectors were reconstructed using the decoder weights to synthesize theoretical representative images of the digit "four" ([Figure 3a](#fig3a)). 

:::{figure} #fig3a_data
:name: fig3a
:placeholder: ./figures/fig3a.png
:enumerator: 3a
Decoded latent vectors: arithmetic mean (left), median (middle) and geometric median (right).
:::

### 2. Define a practical representative image

The behaviour of the theoretical image generally does not correspond to a distinct image, therefore it is not considered the final representative image [@doi:10.1109/BIP60195.2023.10379342]. However, the theoretical average latent vectors can be used to determine practical representative images from the dataset. The closest latent vector to each average was measured by Euclidean distance in the vector embedding ([Figure 3b](#fig3a)). It was found to be the same image for all three measures, and it is remarkably similar to the practical representative images chosen in [Figure 2b](#fig2b), which suggests that the methods are comparable.

:::{figure} #fig3b_data
:name: fig3b
:placeholder: ./figures/fig3b.png
:enumerator: 3b
Closest examples to the arithmetic mean (left), median (middle) and geometric median (right).
:::
