# GAN on Stanford Cars Dataset using PyTorch

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate realistic car images based on the Stanford Cars dataset.

## Dataset
- **Name:** Stanford Cars Dataset
- **Images:** 16,185 total (train/test split provided)
- **Classes:** 196 car models
- **Preprocessing:**
  - Resized all images to 64×64
  - Normalized pixel values to [-1, 1]

## Model Architectures

### Discriminator
A CNN that classifies whether an image is real or fake:
- 5 convolutional layers
- Batch normalization in deeper layers
- LeakyReLU activations with slope = 0.2
- Output: 1x1 score per image (real/fake)

### Generator
A transposed CNN that generates images from random noise vectors:
- 5 transposed convolutional layers
- Batch normalization in all but last layer
- ReLU activations (except last: Tanh)
- Output: 3×64×64 RGB image

## Training Details
- **Noise Dim (z):** 100
- **Optimizer:** Adam (`lr=0.0002`, `betas=(0.5, 0.999)`)
- **Loss:** Binary Cross-Entropy
- **Batch Size:** 128
- **Epochs:** 50
- **Regularization:** BatchNorm, label smoothing
- **Framework:** PyTorch

## Evaluation
- **Visual Inspection:** Generated samples visualized every few epochs
