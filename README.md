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
 
The curated dataset can be found in the following drive link: https://drive.google.com/drive/folders/1qKIMJx949qAPC71GlGS1cGPceBISQGma?usp=sharing
 
# Methodology

We use the CycleGAN architecture to transform the images of unmasked faces into masked faces. The above problem is made more intricate by the fact that there does not exist a pair-to-pair data-set for masked to unmasked faces. We define pair to pair data-set as the same face without the mask and with the mask.
![ai1](https://user-images.githubusercontent.com/18104407/144973798-d01f66de-1e40-47f5-9b23-00f2cc9f0c99.png)


# Deadlines

  **Beta verison**   Dec 1 \
  **Final version**  Dec 10
  
# Approach/Status

We are using cycle GAN to transfer deep fake masks on unmasked faces. Cycle GAN essentially, takes in 2 domains of images: an input domain and a target domain. During training, it tries to figure out the transfer function required to convert images in input domain to the output domain. In this case, domain U can be considered unmasked and domain M can be considered as masked. Essentially, there are 2 generators G_x and G_y, and two descriminators D_x and D_y.

# Trails/Experiments
1. 128 x 128 - b 14 -l linear -e 80 -> Good results but on 128's. Diminished quality. Location: /home/rallam/pytorch-CycleGAN-and-pix2pix/checkpoints
2. 256 x 256 - b 8  -l plateau -e 130 
3. 256 x 256 - b 16 -l linear  -e 200 -> Masking model is pretty good. Unmasking is not great. Location : /home/rallam/pytorch-CycleGAN-and-pix2pix_nws/checkpoints/unmasked_2_masked_b16_serialbatchesrandom

# References
  1. https://github.com/cabani/MaskedFace-Net (for face masked images) - Most of the images in this dataset are not well masked. We will be only selecting the images which are properly masked.\
  2. https://github.com/NVlabs/ffhq-dataset (for unmasked images) - Flickr-Faces-HQ (FFHQ) is a high-quality image dataset of 70,000 high-quality PNG images at 1024×1024 resolution of human faces.
  3. https://towardsdatascience.com/demystifying-gans-cc1ac011355 
  4. Cycle GAN paper : https://arxiv.org/abs/1703.10593
  5. https://towardsdatascience.com/cycle-gan-with-pytorch-ebe5db947a99
