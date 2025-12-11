---
title: Autoencoders
numbering:
  enumerator: 2.%s
label : autoencoder_page
---

# Preliminaries

## Autoencoders

From the Deep Learning book by {cite}`Goodfellow-et-al-2016`;
an autoencoder is a neural network that is trained to copy its input to its output.
Autoencoders originated in the eighties and its primary application was dimensionality reduction for information storage and retrieval.
Autoencoders consist of two parts: an encoder and a decoder with hidden layers that describe the code used to represent the data.
The autoencoder is restricted in some way that it is forced to prioritize which aspects of the input to copy, so it often learns useful properties of the data.

## Model architecture

During training, the input to the model was a fixed-size $256*256$ grayscale image.
The architecture of the networks in this study followed a common configuration in autoencoder-based anomaly detection methods [@doi:10.48550/arXiv.2501.13864].
Rectified linear unit (ReLU) activation functions were used throughout the network for nonlinearity {cite}`NIPS2012_c399862d`, except with a sigmoid activation function at the final layer.
Linear activation functions were used in the fully-connected dense layer connections to and from the latent space bottleneck.
Stacked $3*3$ kernels were used throughout the network, so each layer has two convolutions without spatial pooling in between, giving an effective receptive field of $5*5$ with fewer parameters and greater nonlinearity [@doi:10.48550/arXiv.1409.1556].
Spatial pooling is carried out by average pooling layers, which is performed over a $2*2$ window with a stride of 2.
A stack of four convolutional layers is followed by a fully-connected dense layer into latent space $z$. The decoder mirrors the encoder.
Latent vectors are reshaped then upsampled with Conv2DTranspose (stride 2) and Conv2D through a stack of convolutional layers to reconstruct a $256*256$ grayscale image.
The network was trained with Adam optimizer [@doi:10.48550/arXiv.1412.6980] to minimize the mean squared error (MSE) between original and reconstructed images.
These configurations were used for all autoencoders in this study.
Models varied in the number of layers, number of filters, latent dimensions, learning rate and batch size which were parameters that were manually tuned to the dataset.
In deciding on a model, Barkley considered the importance of limiting the number of parameters in the model and decided on the vanilla model.
There are variations of autoencoders that have been compared to show the efficiencies and trade-offs of different models at image reconstruction, latent representation and accuracy at anomaly detection [@doi:10.1016/j.mlwa.2024.100572].

# Representative images from the MNIST dataset

## Literature reproduction

A method of objective representative image selection [@doi:10.1109/BIP60195.2023.10379342] was tested on real-world data like the [MNIST database](<wiki:MNIST_database>), which is a collection of 70,000 images of handwritten numbers that were manually annotated into ten classes corresponding to the digits 0-9.
Briefly, the proposed two-step approach to objective representative image selection calculated theoretical average images then measured the distance between real and theoretical images in a vector space.
Measures of central tendency to compute theoretical representative images were the arithmetic mean, median and geometric median, chosen due to their ease of computation.
Averages were calculated as a 784-dimensional vector then reshaped to reconstruct 28x28 pixel theoretical images.
One of the examples shown in the literature used a sub-set of MNIST images (n=720) labelled "four".
Though the reproduction used all MNIST "four" images (N=6824), the outcome was consistent with the primary literature ([](#fig2a)) [@doi:10.1109/BIP60195.2023.10379342].

:::{figure} #fig2a_data
:label: fig2a
:placeholder: ./figures/fig2a.png
Computation of theoretical representative images of the MNIST digit '4'.
N=6824 grayscale images with the label '4' were flattened to 784-dimensional vectors to compute then reshape reconstructed images of the arithmetic mean (left), median (middle) and geometric median (right).
:::

Chosen exemplars were not the same as the literature ([](#fig2b)) [@doi:10.1109/BIP60195.2023.10379342], though this was unsurprising because our subset of MNIST was not the same.

:::{figure} #fig2b_data
:label: fig2b
:placeholder: ./figures/fig2b.png
Computation of practical representative images of the MNIST digit '4' using the arithmetic mean (left), median (middle) and geometric median (right).
:::

## Using autoencoders to select representative MNIST digits

Barkley adopted this approach to representative image selection using the latent space of an autoencoder model.
First, theoretical average latent vectors were calculated using the same measures of central tendency.
Then, practical examples in the embedding were ranked by Euclidean distance to the calculated centroids.

### 1. Compute and decode centroid latent vectors

Barkley trained a convolutional autoencoder model on the MNIST dataset and saved the encoder and decoder weights ([](#sfig2a)). 
n=6824 latent vectors labelled "four" were averaged and reconstructed using the decoder weights to synthesize theoretical images of the digit "four" ([](#fig2c)).
Interestingly, the decoded centroid latent vectors appeared like the mean and median reconstructions calculated using independent pixel values ([](#fig2a)).

:::{figure} #fig2c_data
:name: fig2c
:placeholder: ./figures/fig2c.png
Decoded latent vectors: arithmetic mean (left), median (middle) and geometric median (right).
:::

### 2. Define practical images in Euclidean space

The behaviour of the theoretical image generally does not correspond to a distinct image, therefore it is not considered the final representative image.
However, it can be used to select representative examples from the dataset [@doi:10.1109/BIP60195.2023.10379342].
The closest latent vector to each centroid was found based on the lowest Euclidean distance in the vector embedding ([](#fig2d)).
The resulting images were remarkably similar, if not identical to the practical representative images in [](#fig2b), which suggests that these methods were comparable.

:::{figure} #fig2d_data
:name: fig2d
:placeholder: ./figures/fig2d.png
Closest examples to the arithmetic mean (left), median (middle) and geometric median (right).
:::

## Limitation to current methods of objective image selection

These examples of image selection, while clearly effective, relied on class information from an annotated dataset.
As a counterexample, both methods of image selection fell apart when reconstructing the theoretical average using the global centroid.
Again, the reconstructed images from the latent space of the autoencoder had characteristics like that of the pixel-wise average reconstructions that are difficult to describe, which sheds light on the black box.

:::{figure} #fig2e_data
:name: fig2e
:placeholder: ./figures/fig2e.png
Figure legend.
:::

:::{figure} #fig2f_data
:name: fig2f
:placeholder: ./figures/fig2f.png
Figure legend.
:::

Reconstructing the global average did not produce meaningful images given our knowledge of the digits 0-9.
This suggests that determinations on computing the global or class average will make or break this two step approach to image selection, therefore these methods are not generalizable.
Conceptually it made sense to find an average image of the digit 4, but not to find the average of all digits.
Perhaps this approach to image selection could be viable if averaging by label, but this would require an upstream classifier model or manual annotation.

# Representative images from NucleusNet

## Model training

The entire single-cell image collection was sharded into [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) to better load the dataset into memory and shuffle it during training.
Barkley trained an autoencoder model for fifty epochs on the full dataset and saved the encoder and decoder weights after training, then encoded the grayscale images into latent vectors.
The purpose of this analysis was not to interpret the latent space for scientific inquiry or biological discovery, because artificial models are black boxes.
Rather, the goal was to show that the autoencoder formed a latent space, then images were chosen near the centroid for presentation as representative images.

### Loss plot

:::{figure} #ae1m-loss-plot
:label: fig2g
:placeholder: ./figures/fig2g.png
Mean squared error (MSE) reconstruction loss (batch size: 32) plot for the training and validation datasets recorded during training.
:::

## Latent space interpolation

A method to evaluate the quality of latent space is interpolation, whereby mixing codes in latent space and decoding the result creates a semantically meaningful combination of the datapoints [@doi:10.48550/arXiv.1807.07543].
Interpolating with an autoencoder describes the process of using the decoder to decode a convex combination of two latent vectors [@doi:10.48550/arXiv.1807.07543].
A high-quality interpolation should have two characteristics: intermediate points along the interpolation should resemble real data and they should provide a semantically meaningful transition between the endpoints [@doi:10.48550/arXiv.1807.07543].
Interpolating between any two latent vectors of embedded NucleusNet images produced reasonable intermediate reconstructions from the decoder with a smooth transition between endpoints ([](#fig2h)).
This result is consistent with the literature describing smooth interpolations with base model autoencoders [@doi:10.48550/arXiv.1807.07543].
The authors noted that intermediate points did not always resemble real data, which was true with some of the NucleusNet-10K interpolations.

:::{figure} #nucleusnet10k-interpolation
:label: fig2h
:placeholder: ./figures/fig2h.png

Decoded latent vectors along intermediate points of interpolations between random pairs of images from NucleusNet-10K.
Executing the code will randomly draw fifty pairs for interpolation.
Images that you see are decoded latent vectors, with t=0.00 and t=1.00 representing latent vectors corresponding to real images.
Four intermediate interpolations at t=0.20, t=0.40, t=0.60 and t=0.80 are mixed codes.
:::

## Autoencoder-based determination of representative images

1. Computation of a theoretical image

Barkley calculated theoretical latent vectors using measures of central tendency, including the arithmetic mean, median and geometric median in latent space.
Averaged latent vectors were reconstructed with the decoder to synthesize theoretical representative images of the nucleus ([](#fig2i)).
Theoretical representative images do not necessarily look like real data [@doi:10.1109/BIP60195.2023.10379342], so these reconstructions do not represent real data.
Otherwise, theoretical average nuclei appeared to blend all image features from the dataset.
As noted earlier, there is what appears to be background signal around the theoretical nucleus.

:::{figure} #ae1m-theoretical
:label: fig2i
:placeholder: ./figures/fig2i.png
Figure legend.
:::

2. Determination of a prototypical image

Real images were selected based on distance to theoretical centroids in latent space.
Specifically, representative images were defined by the lowest euclidean distance or highest cosine similarity to theoretical averages.
One hundred representative images from NucleusNet were ranked based on each distance metric and measure of central tendency ([](#fig2j)).


:::{figure} #ae1m-practical
:label: fig2j
:placeholder: ./figures/fig2j.png
Figure legend.
:::