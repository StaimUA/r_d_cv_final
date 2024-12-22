# Image Denoising with U-Net

This repository contains implementations of U-Net architectures for image denoising. The models are trained on a dataset of images with added Gaussian, Poisson, and salt-and-pepper noise.

## Project Overview

The main goal was to check the level of knowledge obtained during the Computer Vision course. Also, the goal was to try to build a model that could work with several types of noise.

This project explores popular deep learning architecture for image denoising:

*   **U-Net:** A U-shaped encoder-decoder network with skip connections.

## Features

*   Implementation of U-Net architecture for image denoising.
*   Data loading using CSV files with support for image resizing and normalization.
*   Generation of diverse training data by adding random Gaussian, Poisson, and salt-and-pepper noise to clean images.
*   Data augmentation with random horizontal flips, rotations, and scaling (zoom).
*   Training using Mean Absolute Error (MAE) loss.
*   Callbacks for learning rate reduction, early stopping, and model checkpointing based on the validation loss.
*   Visualization of training progress with TensorBoard.
*   Analysis of the distribution of noise parameters across training, validation, and test sets.

## Model Architectures

### U-Net

The U-Net architecture used in this project follows a standard encoder-decoder structure with skip connections. Each level in the encoder consists of two convolutional layers followed by max pooling with ReLU activations and batch normalization. The decoder consists of transposed convolutions and skip connections to combine low-level and high-level features.

## Noise Generation

*   Gaussian Noise: The standard deviation is drawn from a uniform distribution between 5 and 30. The mean is chosen between -5 and 5.
*   Poisson Noise: The scale parameter is drawn from uniform distribution between 0.5 and 2.0.
*   Salt-and-Pepper Noise: The probability of pixel corruption is sampled from bimodal distribution: lower (0.01-0.02) or higher (0.03-0.05) with equal probability.

During the training process, model training, validation, and test losses are monitored, also Mean Absolute Error (MAE) are monitored. For evaluation the PSNR and SSIM metrics can be used.

## What Can Be Improved

There are several areas where project can be improved:

*   **Data:**
    *   **Larger Datasets:** Use larger and more diverse datasets to improve the model's generalization capabilities.
    *   **Real-World Noise:** Incorporate real-world noise characteristics in dataset.
*   **Model Architectures:**
    *   **Advanced U-Net:** Explore larger versions of U-Net with more layers, or attention mechanisms.
    *   **Hybrid Models:** Experiment with combining U-Net and DnCNN architectures.
*   **Training:**
    *  **Loss function:** Consider combining MAE with other loss functions or using perceptual losses.
*   **Data Augmentation:**
    *   **More Augmentations:** Add other types of augmentations, such as contrast and brightness adjustments, elastic deformations or other types of affine transformations.
    *   **Adaptive Parameters:** Adjust the parameters of augmentations to better fit to your data.
    *   **Augment Specific Noises:** Tailor augmentations to each type of noise you're trying to remove.
* **Code**
    *  **Use TF.Dataset**: Try to implement data generators using `tf.data.Dataset` API, which improves loading speed and performance.
