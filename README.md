# Image Segmentation using U-Net Architecture

This project implements a U-Net convolutional neural network architecture for biomedical and general image segmentation tasks (such as multi-class semantic segmentation). U-Net is widely known for its encoder-decoder structure with skip connections, which helps in preserving fine-grained spatial details while learning high-level features.

## Core Components & Architecture

The project is structured around three main building blocks:

1. **Convolutional Contracting Block (`conv_block`)**:
   - Implements the contracting path (encoder) of the U-Net.
   - Consists of two consecutive 2D Convolutional layers with ReLU activations, 'same' padding, and He Normal initializers.
   - Supports optional Dropout regularization and Max Pooling for spatial dimension downsampling.

2. **Convolutional Upsampling Block (`upsampling_block`)**:
   - Implements the expansive path (decoder).
   - Uses `Conv2DTranspose` to upsample feature maps from the previous layer.
   - Concatenates the upsampled features with the corresponding contracting path output (skip connection) to recover precise localization.
   - Followed by two convolutional layers to refine features.

3. **U-Net Model (`unet_model`)**:
   - Combines the encoder and decoder paths into a fully functional end-to-end Keras Model.
   - Gradually doubles the number of filters in the contracting path and halves them in the expansive path.
   - Ends with a 1x1 Convolutional layer mapping the feature maps to the desired number of output classes using softmax/logits.

## Required Libraries

To run this model successfully, ensure you have the following packages installed:
- TensorFlow (v2.x)
- NumPy
- Matplotlib
- Keras
