import os
import re
import csv
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def load_and_preprocess_image_no_aug(img_path, resize_to=True):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    if resize_to:
        # Resize to handles different orientations
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalization
    img = img.astype(np.float32) / 255.0
    return img

def generate_augmentation_params():
    """Generates random augmentation parameters for an image."""
    params = {
        "flip": random.random() < 0.5,
        "angle": random.randint(-10, 10) if random.random() < 0.5 else 0,
        "scale": round(random.uniform(0.8, 1.2), 2) if random.random() < 0.5 else 1.0,
    }
    return params

def augment_image(img, augment_params):
    """Applies augmentation to an image given the augmentation parameters"""
    if augment_params:
        if augment_params["flip"]:
            img = cv2.flip(img, 1)  # horizontal flip

        if augment_params["angle"] != 0:
            angle = augment_params["angle"]  # rotation angle
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

        if augment_params["scale"] != 1.0:
            scale = augment_params["scale"]
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
            img = cv2.warpAffine(img, M, (cols, rows))

    return img

def generator_from_csv(csv_path, batch_size=8, shuffle=True, augment=False):
    # Read all rows from CSV
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    # shuffle once at start
    if shuffle:
        random.shuffle(rows)
    
    idx = 0
    n = len(rows)
    
    while True:
        noisy_batch = []
        original_batch = []
        
        for _ in range(batch_size):
            if idx >= n:
                idx = 0
                if shuffle:
                    random.shuffle(rows)
            
            row = rows[idx]
            noisy_path = row[0]
            original_path = row[1]

            # Load the images
            noisy_img = load_and_preprocess_image_no_aug(noisy_path)
            original_img = load_and_preprocess_image_no_aug(original_path)

            if augment:
              augment_params = generate_augmentation_params()
              noisy_img = augment_image(noisy_img, augment_params)
              original_img = augment_image(original_img, augment_params)
                       
            # If any load fails, skip
            if noisy_img is None or original_img is None:
                idx += 1
                continue
            
            noisy_batch.append(noisy_img)
            original_batch.append(original_img)
            
            idx += 1
        
        yield (np.array(noisy_batch, dtype=np.float32),
               np.array(original_batch, dtype=np.float32))


def extract_noise_parameters(csv_path):
    """
    Extracts noise parameters from a CSV file.
    """
    gaussian_means = []
    gaussian_stds = []
    poisson_scales = []
    sp_probs = []

    with open(csv_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            if len(row) < 4:
              continue

            noise_type = row[2]
            noise_params = row[3]

            if noise_type == 'gaussian':
                match = re.search(r"mean=(-?\d+),std=(\d+)", noise_params)
                if match:
                  gaussian_means.append(int(match.group(1)))
                  gaussian_stds.append(int(match.group(2)))
                
            elif noise_type == 'poisson':
                match = re.search(r"scale=(\d+\.?\d*)", noise_params)
                if match:
                  poisson_scales.append(float(match.group(1)))
            elif noise_type == 'salt_and_pepper':
                match = re.search(r"prob=(\d+\.?\d*)", noise_params)
                if match:
                  sp_probs.append(float(match.group(1)))
    
    return gaussian_means, gaussian_stds, poisson_scales, sp_probs

def plot_noise_distributions(train_params, val_params, test_params):
    """
    Plots the distributions of noise parameters from the different datasets.

    """
    
    train_gaussian_means, train_gaussian_stds, train_poisson_scales, train_sp_probs = train_params
    val_gaussian_means,   val_gaussian_stds,   val_poisson_scales,   val_sp_probs   = val_params
    test_gaussian_means,  test_gaussian_stds,  test_poisson_scales,  test_sp_probs  = test_params
    
    
    #Parameter lists and titles
    params = [
        ([train_gaussian_means,val_gaussian_means, test_gaussian_means], 'Gaussian noise Mean', 'Mean'),
        ([train_gaussian_stds, val_gaussian_stds, test_gaussian_stds], 'Gaussian noise Std', 'Standard Deviation'),
        ([train_poisson_scales, val_poisson_scales, test_poisson_scales], 'Poisson noise Scale', 'Scale'),
        ([train_sp_probs, val_sp_probs, test_sp_probs], 'Salt-and-Pepper noise Probability', 'Probability')
    ]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    for i, (data, title, xlabel) in enumerate(params):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        if not isinstance(axes, np.ndarray):
          axes = [axes] # if only one axes is present
        
        data_labels = ["Train","Validation","Test"]
        for j, dataset_data in enumerate(data):
            ax = axes[j]
            ax.hist(dataset_data, bins=20, color=colors[j], label = data_labels[j], edgecolor = 'black', alpha = 0.6)

            ax.set_title(f'{title} Distribution for {data_labels[j]}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Frequency')
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.show()


def visualize_denoising_results(model, generator, num_images=3):
    for i in range(num_images):

        noisy_batch, original_batch = next(generator)
        
        # Use model to predict the denoised image
        denoised_pred = model.predict(noisy_batch)
        
        idx = 0 

        noisy_img = noisy_batch[idx]
        original_img = original_batch[idx]
        denoised_img = denoised_pred[idx]
        denoised_img = np.clip(denoised_img, 0.0, 1.0)
        
        noisy_uint8 = (noisy_img * 255).astype(np.uint8)
        
        # Apply Gaussian Blur
        gaussian_uint8 = cv2.GaussianBlur(noisy_uint8, (3, 3), 0)
        gaussian_img = gaussian_uint8.astype(np.float32) / 255.0  # Scale back to [0,1]
        
        # Compute PSNR for both methods
        model_psnr = compute_psnr(original_img, denoised_img, data_range=1.0)
        gaussian_psnr = compute_psnr(original_img, gaussian_img, data_range=1.0)
        
        # Compute SSIM
        model_ssim = compute_ssim(original_img, denoised_img, channel_axis=2, data_range=1.0)
        gaussian_ssim = compute_ssim(original_img, gaussian_img, channel_axis=2, data_range=1.0)
        
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("Noisy Input")
        plt.imshow(noisy_img)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.title(f"Model Denoised\nPSNR: {model_psnr:.2f} dB\nSSIM: {model_ssim:.4f}")
        plt.imshow(denoised_img)
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title(f"Gaussian Filtered\nPSNR: {gaussian_psnr:.2f} dB\nSSIM: {gaussian_ssim:.4f}")
        plt.imshow(gaussian_img)
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title("Clean Ground Truth")
        plt.imshow(original_img)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Image {i+1} - DnCNN PSNR: {model_psnr:.2f} dB, SSIM: {model_ssim:.4f}")
        print(f"Image {i+1} - Gaussian PSNR: {gaussian_psnr:.2f} dB, SSIM: {gaussian_ssim:.4f}\n")
