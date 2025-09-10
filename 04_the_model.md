---
Chapter 4
---

# Autoencoders

An autoencoder is a neural network that is trained to copy its input to its output {cite}`Goodfellow-et-al-2016`. Autoencoders originated in the eighties and its primary application was dimensionality reduction for information storage and retrieval {cite}`Goodfellow-et-al-2016`. Autoencoders consist of two parts: an encoder and a decoder with hidden layers that describe the code used to represent the data. The autoencoder is restricted in some way that it is forced to prioritize which aspects of the input to copy, so it often learns useful properties of the data {cite}`Goodfellow-et-al-2016`. 

## Model architecture

During training, the input to the model is a fixed-size $256*256$ grayscale image. The architecture of the networks in this study followed common configurations in autoencoder-based anomaly detection methods [@doi:10.48550/arXiv.2501.13864]. [Rectified linear unit](#equation_D) (ReLU) activation functions were used throughout the network for nonlinearity {cite}`NIPS2012_c399862d`, except with a [sigmoid](#equation_E) activation function at the final layer. [Linear](#equation_F) activation functions were used in the fully-connected dense layer connections to and from the latent space bottleneck. Stacked $3*3$ kernels were used throughout the network. Each layer has two $3*3$ convolutions with ReLUs without spatial pooling in between, giving an effective receptive field of $5*5$ with fewer parameters and greater nonlinearity [@doi:10.48550/arXiv.1409.1556]. Spatial pooling is carried out by average pooling layers, which is performed over a $2*2$ window with a stride of 2. A stack of four convolutional layers is followed by a fully-connected dense layer into latent space $z$. The decoder mirrors the encoder. A latent vector is reshaped then upsampled with Conv2DTranspose (stride 2) and Conv2D through a stack of convolutional layers to reconstruct a $256*256$ grayscale image. The network was trained with Adam optimizer [@doi:10.48550/arXiv.1412.6980] to minimize the [mean squared error](#equation_G) (MSE) between original and reconstructed images. These configurations were used for all autoencoders in this study. The models varied in the number of layers, number of filters, latent dimensions, learning rate and batch size which were parameters that were tuned to the dataset. In deciding on a model for objective representative image selection, we considered the importance of limiting the number of parameters in the model, thus we chose the classic autoencoder model. There are variations of the autoencoder model that have been compared to show the efficiencies and trade-offs of different models at image reconstruction, latent representation and accuracy at anomaly detection [@doi:10.1016/j.mlwa.2024.100572]. 

---

## Training on the nucleus dataset

The entire single-cell image collection is large (>70GB) so it was sharded using {ref}`tfrecord-shards` into [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) to better load the dataset into memory and shuffle it during training.

We trained this model: {ref}`AE1M` for 50 epochs on the full dataset of single-cell images and saved the encoder and decoder weights after training for inference.

### Loss plot

:::{figure} #fig:ae1m-loss-plot
:name: fig5A
:placeholder: ./figures/fig5a.png
Example widget.
:::

## Image reconstruction

:::{figure} #fig:AE1M-reconstructions
:name: fig5B
:placeholder: ./figures/fig5b.png
Example widget.
:::
