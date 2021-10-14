# AI_project_deep_fakes

Goal: Deep faking of face masks onto unmasked faces.

# Methodology
1. Clean and curate the original dataset of masked and unmasked individuals to get the images for training and testing. 
2. We will be feeding pairs of masked and unmasked faces to the network so that both generator and the discriminator networks compete against each other so that the generator can learn to generate realistic face masked images. 
3. We will be GANs (will explore out a few such as SimGAN, CycleGAN, StyleGAN, etc) to create realistic pictures of faces with facemasks.
# Dataset

# Deadlines

  **Beta verison**   Dec 1 \
  **Final version**  Dec 10

# References
  1. https://github.com/cabani/MaskedFace-Net (for face masked images) - Most of the images in this dataset are not well masked. We will be only selecting the images which are properly masked.\
  2. https://github.com/NVlabs/ffhq-dataset (for unmasked images) - Flickr-Faces-HQ (FFHQ) is a high-quality image dataset of 70,000 high-quality PNG images at 1024×1024 resolution of human faces.
  3. https://towardsdatascience.com/demystifying-gans-cc1ac011355 
  4. https://arxiv.org/abs/1703.10593
