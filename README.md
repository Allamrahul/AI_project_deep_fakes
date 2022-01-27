# DeepFakes: Masking and Unmasking Faces using Adversarial Network

The aim of this repository is to document the code and our work on our CS 534 AI term project at Worcester Polytechnic Institute (WPI), MA.

# Project Goal
 
 The goal is to use deep learning techniques, specfically generative modelling to achieve unpaired image translation from  
 1. Unmasked face domain to masked face domain  
 2. Masked face domain to unmasked face domain  

([Presentation](https://docs.google.com/presentation/d/1XSoOWRFtrMTeHUhrwKhRe_Uf_TKeRI3R/edit?usp=sharing&ouid=101007956401404885384&rtpof=true&sd=true)) ([Paper](https://drive.google.com/file/d/11DmVHPEXjDv80aXT96OrS3-N1nQVY50O/view?usp=sharing)) 
 
 # Dataset
 
 We use 2 different data-sets and curate them according to our use-case:
 1. Flickr-Faces-HQ (FFHQ) data-set for unmasked images
 2. MaskedFace-Net data-set for masked images

For unmasked faces, FFHQ is a high-quality image dataset of human faces, originally created as a benchmark for generative adversarial networks (GAN). The data-set consists
of 70,000 high-quality PNG images at 1024×1024 resolution with variation and diversity in terms of subjects and objects in the frame. 

For our masked face data-set, we use the MaskedFace-Net dataset, which is a dataset of human faces with a correctly or incorrectly worn mask (133,783 images) based on the Flickr-Faces-HQ (FFHQ) data-set. The masks are photo-shopped onto the faces. Although the dataset is based on FFHQ, the facemasks are incorrectly masked for most of the images.

For both of these data-sets, we curate images based on the number of faces in images, mask placement, mask clarity, and realistic effect of the mask on the image. We use a subset of 6000 masked images and 6000 unmasked images for training and 1000 test images from each of them were used for testing.

**Sample image from the unmasked real domain**

<img src="https://user-images.githubusercontent.com/18104407/144974795-c633751a-5bf4-4c36-a948-5ab613684b51.png" alt="alt text" width="128" height="128">

**Sample image from the masked real domain**

<img src="https://user-images.githubusercontent.com/18104407/144974846-2a69b428-e070-48c7-984f-9b5e24ef6e5a.jpg" alt="alt text" width="128" height="128">

The curated dataset can be found in the following drive link: https://drive.google.com/drive/folders/1qKIMJx949qAPC71GlGS1cGPceBISQGma?usp=sharing

From here on, I would be referring to  
  1. The **unmasked domain** as **domain A**
  2. The **masked domain as** as **domain B**
 
# Methodology

We use the CycleGAN architecture to transform the images of unmasked faces into masked faces, and transform masked faced to unmasked faces. The above problem is made more intricate by the fact that there does not exist a pair-to-pair data-set for masked to unmasked faces. We define pair to pair data-set as the same face without the mask and with the mask.

![ai1](https://user-images.githubusercontent.com/18104407/144973798-d01f66de-1e40-47f5-9b23-00f2cc9f0c99.png)

This leads us to the task of unpaired image-to-image translation. CycleGANs have previously been used for tasks such as horse-zebra, summer-winter etc. This task can be formulated as an image from a source domain X to a target domain Y in the absence of paired examples. We define the goal as mapping G : X → Y such that the distribution of images from G(X) is indistinguishable from the distribution Y, using an adversarial loss. CycleGAN introduced another loss because this mapping is highly under-constrained. To address this, they couple it with an inverse mapping F : Y → X and introduce a cycle consistency loss to enforce F(G(X)) ≈ X and G(F(Y)) ≈ Y.

To explain GANs, Generative Adversarial Networks include two networks, a Generator G(x), and a Discriminator D(x). The generator tries to generate the data based on the underlying distribution of the training data whereas the discriminator tries to tell apart the fake images from the real ones. They play an adversarial game where the generator tries to fool the discriminator by generating data similar to those in the training set. The Discriminator is fooled when the generated fakes are so real that it cant tell them apart. Both of them are trained simultaneously on data-sets of images, videos, and audio files. The generator G(x) model generates images from random noise and then learns the data distribution of how to generate realistic images. Random noise is given to the generator which outputs the fake images and the real image from the training set is given to the discriminator that learns how to differentiate fake images from real images. The output of Discriminator D(x) is the probability that the given input is real if the output
is 1.0, and if the output is 0 the given input is identified as fake. Thus our goal is to get the output 1 (real) for all the fake images.

In a nutshell, we have a Generator takes random noise as input and outputs a translated image (fake). On the otherhand, a Discriminator takes the generated fake by the generator and images from domain B as input and outputs 1 for real / 0 for fake. If the discriminator correctly identifies the fake, the generator is penalized and the discriminator is rewarded. The input to the generator is just random noise. If the discriminator incorrectly identifies the fake as real, the generator is rewarded for fooling the discriminator and the discriminator is penalized. The generator and the discriminator play this adversarial game, while improving and updatinng each other.

In essence, we have  
1. a generator **G<sub>AB</sub>**, a CNN that takes random noise as input and learns to translate images from domain A to domain B.   
2. a discriminator **D<sub>AB</sub>**, a classifier that learns to distinguish between the images from the real domain B and the fakes generated by G<sub>AB</sub>.  
3. a generator **G<sub>BA</sub>**, a CNN that takes the output of G<sub>AB</sub> as input and learns to translate images from domain B to domain A.  
4. a discriminator **D<sub>BA</sub>**, a classifier that learns to distinguish between the images from the real domain A and the fakes generated by G<sub>BA</sub>.  
  

# Training
The generator network architecture consists of three convolutions, several residual blocks, two fractionally-strided convolutions with stride 1/2, and one convolution. CycleGAN uses 6 blocks for 128 × 128 images and 9 blocks for 256×256 and higher-resolution training images. For the discriminator networks we use 70 × 70 PatchGANs, which aim to classify whether 70 × 70 overlapping image patches are real or fake. To train the network we used a number of different hyper-parameter such as batch size, batch sequence, normalization type, types of optimization algorithm, change in loss from L1, log-loss, or L2 loss. We performed the below experiments using 4 Nvidia Tesla V100 GPUs and took around 12 hours to run 200 epochs. 

<img src="https://user-images.githubusercontent.com/18104407/144979014-780363cc-2705-452b-9326-d78b2ba26077.png" alt="alt text" width="400" height="256">

In picture below, we present fakes generated through the training process, as we progress through epochs.

![Picture3](https://user-images.githubusercontent.com/18104407/144979352-d1692879-bac5-4893-9e0d-0a77d51b33e8.png)

In picture below, we present the various GAN lossess and its varaiation as a function of epochs.

<img src="https://user-images.githubusercontent.com/18104407/144979712-326ca556-8b73-4f5f-828f-9041798822ab.png" alt="alt text" width="500" height="700">


# Evaluation and Results
Firstly, we present the fakes generated by our model in both domain A and domain B.

<img src="https://user-images.githubusercontent.com/18104407/144980367-89a89b5c-b18f-4fcf-86a9-275cae4f8928.png" alt="alt text" width="800" height="512">

To evaluate our work, we use a manual evaluation in the form of visual inspection of images. 

![Picture6](https://user-images.githubusercontent.com/18104407/144981037-5f16cf27-e2c2-4840-a680-a1fef192c541.png)

We also use the qualitative metric, a visual study is similar to the likes of perceptual studies of Amazon Mechanical Turks with participants shown a sequence of images asking them to label them as real or fake. Rating and Preference Judgment is the most used qualitative method: Images are often presented in pairs and the human judge is asked which image they prefer, e.g., which image is more realistic. We conducted a survey, where the user was asked to guess and filter out the ground truth from the GAN-generated outputs
out of 22 images that were randomly presented to the user. Each correct answer fetches the user 1 point. As it stands, we received 100 responses and the following is the analysis: Of the 2200 predictions, the users wrongly guessed 881 times. This gives us an accuracy of 40.08% which implies, the generated fakes were able to dupe the user 40.08% of time, which is impressive, considering the discerning sight we humans possess, thus validating our model through visual inspection. The statistics for the survey can be seen in figure below, where theaverage score of the user is 13.24.

![Picture7](https://user-images.githubusercontent.com/18104407/144981659-c97b4323-9234-4f5d-a7e3-dfcde6f060d5.png)

We also use a quantitative metric in the form of FID score. The Frechet Inception Distance (FID), is a metric for evaluating the quality of generated images and is generally used to assess the performance of generative adversarial networks. FID measures the distance between the distributions of generated and real samples. Lower FID is better, meaning they are more similar to real and generated samples as measured by the distance between their distributions. Finally, We evaluated our generator models based on Fréchet inception distance (FID), a quantitative GAN evaluation metric. 

<img src="https://user-images.githubusercontent.com/18104407/144981879-151688d6-fda0-4392-84b4-81228f26de10.png" alt="alt text" width="600" height="256">

Through the above training trials, we achieved the least FID<sub>AB</sub> of 17.07 with a generator model G<sub>AB</sub>, which was trained with the following values of hyper-parameters: batch size 16, instance normalization, linear learning policy and lsgan optimization loss. We also observe that the FID<sub>AB</sub> score ranges between 17.07 and 32.18. The best score of 17.07, coupled with small range means that the generator network G<sub>AB</sub> is finding it relatively easier to learn how to apply a fake mask onto an unmasked face, irrespective of the variations in the hyper parameters. Similarly, we achieved the least FID<sub>BA</sub> of 48.39 with a generator model G<sub>BA</sub>, which was trained with batch size 32, instance normalization, linear learning policy and vanilla optimization loss. The best score of 48.39 implies the generator network G<sub>BA</sub> is finding it slightly difficult to generate the masked area of the face. We also observe that FID<sub>BA</sub> score ranges between values 48.39 and 261.78, which implies the performance of the network varies significantly with change
in hyper parameters.

# References
  1. https://github.com/cabani/MaskedFace-Net (for face masked images) - Most of the images in this dataset are not well masked. We will be only selecting the images which are properly masked.\
  2. https://github.com/NVlabs/ffhq-dataset (for unmasked images) - Flickr-Faces-HQ (FFHQ) is a high-quality image dataset of 70,000 high-quality PNG images at 1024×1024 resolution of human faces.
  3. https://towardsdatascience.com/demystifying-gans-cc1ac011355 
  4. Cycle GAN paper: https://arxiv.org/abs/1703.10593
  5. https://towardsdatascience.com/cycle-gan-with-pytorch-ebe5db947a99
