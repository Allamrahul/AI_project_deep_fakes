# DeepFakes: Masking and Unmasking Faces using Adversarial Network

# Goal
 
 The goal is to use deep learning techniques, specfically generative modelling to achieve unpaired image translation from
      a. Unmasked face domain to masked face domain
      b. Masked face domain to unmasked face domain
 
 # Dataset
 
 We use 2 different data-sets and curate them according to our use-case:
 1. Flickr-Faces-HQ (FFHQ) data-set for unmasked images
 2. MaskedFace-Net data-set for masked images

For unmasked faces, FFHQ is a high-quality image dataset of human faces, originally created as a benchmark for generative adversarial networks (GAN). The data-set consists
of 70,000 high-quality PNG images at 1024×1024 resolution with variation and diversity in terms of subjects and objects in the frame. 

For our masked face data-set, we use the MaskedFace-Net dataset, which is a dataset of human faces with a correctly or incorrectly worn mask (133,783 images) based on the Flickr-Faces-HQ (FFHQ) data-set. The masks are photo-shopped onto the faces. Although the dataset is based on FFHQ, the facemasks are incorrectly masked for most of the images.

For both of these data-sets, we curate images based on the number of faces in images, mask placement, mask clarity, and realistic effect of the mask on the image. We use a subset of 6000 masked images and 6000 unmasked images for training and 1000 test images from each of them were used for testing.

Sample image from the unmasked real domain:

<img src="https://user-images.githubusercontent.com/18104407/144974795-c633751a-5bf4-4c36-a948-5ab613684b51.png" alt="alt text" width="256" height="256">

Sample image from the masked real domain: 

<img src="https://user-images.githubusercontent.com/18104407/144974846-2a69b428-e070-48c7-984f-9b5e24ef6e5a.jpg" alt="alt text" width="256" height="256">

The curated dataset can be found in the following drive link: https://drive.google.com/drive/folders/1qKIMJx949qAPC71GlGS1cGPceBISQGma?usp=sharing

From here on, I would be referring to 
  1. The **unmasked domain** as **domain A**
  2. The **masked domain as** as **domain B**
 
# Methodology

We use the CycleGAN architecture to transform the images of unmasked faces into masked faces, and transform masked faced to unmasked faces. The above problem is made more intricate by the fact that there does not exist a pair-to-pair data-set for masked to unmasked faces. We define pair to pair data-set as the same face without the mask and with the mask.
![ai1](https://user-images.githubusercontent.com/18104407/144973798-d01f66de-1e40-47f5-9b23-00f2cc9f0c99.png)

This leads us to the task of unpaired image-to-image translation. CycleGANs have previously been used for tasks such as horse-zebra, summer-winter etc. This task can be formulated as an image from a source domain X to a target domain Y in the absence of paired examples. We define the goal as mapping G : X →Y such that the distribution of images from G(X) is indistinguishable from the distribution Y, using an adversarial loss. CycleGAN introduced another loss because this mapping is highly under-constrained. To address this, they couple it with an inverse mapping F : Y →X and introduce a cycle consistency loss to enforce F(G(X)) ≈ X and G(F(Y )) ≈ Y.

To explain GANs, Generative Adversarial Networks include two networks, a Generator G(x), and a Discriminator D(x). The generator tries to generate the data based on the underlying distribution of the training data whereas the discriminator tries to tell apart the fake images from the real ones. They play an adversarial game where the generator tries to fool the discriminator by generating data similar to those in the training set. The Discriminator is fooled when the generated fakes are so real that it cant tell them apart. Both of them are trained simultaneously on data-sets of images, videos, and audio files. The generator G(x) model generates images from random noise and then learns the data distribution of how to generate realistic images. Random noise is given to the generator which outputs the fake images and the real image from the training set is given to the discriminator that learns how to differentiate fake images from real images. The output of Discriminator D(x) is the probability that the given input is real if the output
is 1.0, and if the output is 0 the given input is identified as fake. Thus our goal is to get the output 1 (real) for all the fake images.

In a nutshell, we have 

Generator
 input - random noise
 output - translated image (fake)
Discriminator 
 input - Generated fake by the discriminator and images from domain B
 output - 1 for real / 0 for fake

If the discriminator correctly identifies the fake, the generator is penalized and the discriminator is rewarded. The input to the generator is just random noise. 
If the discriminator incorrectly identifies the fake as real, the generator is rewarded for fooling the discriminator and the discriminator is penalized.

The generator and the discriminator play this adversarial game, while improving and updatinng each other.

In essence, we have 
  a generator G_AB, a CNN that learns to translate images from domain A to domain B. 
  a discriminator D_AB, a classifier that learns to distinguish between the images from the real domain B and the fakes generated by G_AB.
  a generator G_BA, a CNN that learns to translate images from domain B to domain A. 
  a discriminator D_BA, a classifier that learns to distinguish between the images from the real domain A and the fakes generated by G_BA.

# Training
The generator network architecture consists of three convolutions, several residual blocks, two fractionally-strided convolutions with stride 1/2, and one convolution. CycleGAN uses 6 blocks for 128 × 128 images and 9 blocks for 256×256 and higher-resolution training images. For the discriminator networks we use 70 × 70 PatchGANs, which aim to classify whether 70 × 70 overlapping image patches are real or fake. To train the network we used a number of different hyper-parameter such as batch size, batch sequence, normalization type, types of optimization algorithm, change in loss from L1, log-loss, or L2 loss. The above experiments were done using 4 Nvidia Tesla V100 GPUs and took around 12 hours to run 200 epochs. 


# Evaluation 
To evaluate our work, we use a qualitative metric in the form of visual inspection of images. The visual study is similar to the likes of perceptual studies of Amazon
Mechanical Turks with participants shown a sequence of images asking them to label them as real or fake. We also use a quantitative metric in the form of FID score. The Frechet Inception Distance (FID), is a metric for evaluating the quality of generated images and is generally used to assess the performance of generative adversarial networks. FID measures the distance between the distributions of generated and real samples. Lower FID is better, meaning they are more similar to real and generated samples as measured by the distance between their distributions.

# Results

# References
  1. https://github.com/cabani/MaskedFace-Net (for face masked images) - Most of the images in this dataset are not well masked. We will be only selecting the images which are properly masked.\
  2. https://github.com/NVlabs/ffhq-dataset (for unmasked images) - Flickr-Faces-HQ (FFHQ) is a high-quality image dataset of 70,000 high-quality PNG images at 1024×1024 resolution of human faces.
  3. https://towardsdatascience.com/demystifying-gans-cc1ac011355 
  4. Cycle GAN paper : https://arxiv.org/abs/1703.10593
  5. https://towardsdatascience.com/cycle-gan-with-pytorch-ebe5db947a99
