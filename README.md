# AI_project_deep_fakes

Goal: Deep faking of face masks onto unmasked faces.

# Methodology
1. Clean and curate the original dataset of masked and unmasked individuals to get the images for training and testing. 
2. We will be feeding pairs of masked and unmasked faces to the network so that both generator and the discriminator networks compete against each other so that the generator can learn to generate realistic face masked images. 
3. We will be GANs (will explore out a few such as SimGAN, CycleGAN, StyleGAN, etc) to create realistic pictures of faces with facemasks.
# Dataset
The curated dataset can be found in the following drive link: https://drive.google.com/drive/folders/1qKIMJx949qAPC71GlGS1cGPceBISQGma?usp=sharing

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
