import os
import cv2
import csv
import random
import numpy as np
from typing import List, Tuple

# params
train_ratio = 0.70
val_ratio   = 0.15
test_ratio  = 0.15

# Paths
root_folder = 'DATA'
original_folder = os.path.join(root_folder, 'original')
noisy_folder = os.path.join(root_folder, 'noisy')

# subfolders for sets
train_dir = os.path.join(noisy_folder, 'train')
val_dir  = os.path.join(noisy_folder, 'val')
test_dir  = os.path.join(noisy_folder, 'test')

# CSV annotations
train_csv_name = 'annotations_train.csv'
val_csv_name   = 'annotations_val.csv'
test_csv_name  = 'annotations_test.csv'

########################################################
# NOISE FUNCTIONS

def add_gaussian_noise(image, mean=0, std=25):
    float_img = image.astype(np.float32)
    noise = np.random.normal(mean, std, float_img.shape)
    noisy_img = float_img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def add_poisson_noise(image, scale=1.0):
    float_img = image.astype(np.float32)
    max_val = 255.0
    norm_img = float_img / max_val
    poisson_vals = np.random.poisson(norm_img * scale * max_val)
    noisy_img = (poisson_vals / (scale * max_val)) * max_val
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def add_salt_and_pepper_noise(image, prob=0.01):
    noisy_img = np.copy(image)
    random_matrix = np.random.rand(noisy_img.shape[0], noisy_img.shape[1])
    # Add pepper
    pepper_coords = random_matrix < (prob / 2)
    noisy_img[pepper_coords] = 0
    # add salt
    salt_coords = (random_matrix >= (prob / 2)) & (random_matrix < prob)
    noisy_img[salt_coords] = 255
    return noisy_img

def get_all_original_images(folder):
    extensions = ('.png', '.jpg', '.jpeg')
    all_images = []
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(extensions):
            full_path = os.path.join(folder, file_name)
            all_images.append(full_path)
    return all_images

########################################################
# Main part

def main():
   
    # Gather all clean image paths from original_folder
    original_files = get_all_original_images(original_folder)
    
    # Shuffle files before splitting
    random.shuffle(original_files)
    
    total_images = len(original_files)
    train_count = int(train_ratio * total_images)
    val_count   = int(val_ratio * total_images)
    
    # split files per ratio
    train_clean = original_files[:train_count]
    val_clean   = original_files[train_count : train_count + val_count]
    test_clean  = original_files[train_count + val_count:]
    
    print(f"Total origianl Images: {total_images}")
    print(f"Train: {len(train_clean)}, Val: {len(val_clean)}, Test: {len(test_clean)}")
    
    # write CSV
    train_csv = open(train_csv_name, 'w', newline='')
    val_csv   = open(val_csv_name,   'w', newline='')
    test_csv  = open(test_csv_name,  'w', newline='')
    
    train_writer = csv.writer(train_csv)
    val_writer   = csv.writer(val_csv)
    test_writer  = csv.writer(test_csv)
    
    # add headers to files
    header = ['noisy_image_path', 'original_image_path', 'noise_type', 'noise_params']
    train_writer.writerow(header)
    val_writer.writerow(header)
    test_writer.writerow(header)
    
    # define a function to process each set
    def process_and_save_noisy(original_path, out_dir, csv_writer):
        for path in original_path:
            # Load the original image
            img = cv2.imread(path)
            if img is None:
                continue
                      
            # Gaussian noise
            mean = random.randint(-5, 5)
            std  = random.randint(5, 30)
            gauss_img = add_gaussian_noise(img, mean=mean, std=std)
            gauss_filename = os.path.splitext(os.path.basename(path))[0] + '_gauss.jpg'
            gauss_outpath = os.path.join(out_dir, gauss_filename)
            cv2.imwrite(gauss_outpath, gauss_img)
            
            csv_writer.writerow([
                gauss_outpath,
                path,
                'gaussian',
                f"mean={mean},std={std}"
            ])
            
            # Poisson noise
            scale = round(random.uniform(0.5, 2.0), 2)
            poisson_img = add_poisson_noise(img, scale=scale)
            poisson_filename = os.path.splitext(os.path.basename(path))[0] + '_poisson.jpg'
            poisson_outpath = os.path.join(out_dir, poisson_filename)
            cv2.imwrite(poisson_outpath, poisson_img)
            
            csv_writer.writerow([
                poisson_outpath,
                path,
                'poisson',
                f"scale={scale}"
            ])
            
            # Salt & Pepper noise
            sp_prob = round(random.uniform(0.01, 0.05), 3)
            sp_img = add_salt_and_pepper_noise(img, prob=sp_prob)
            sp_filename = os.path.splitext(os.path.basename(path))[0] + '_sp.jpg'
            sp_outpath = os.path.join(out_dir, sp_filename)
            cv2.imwrite(sp_outpath, sp_img)
            
            csv_writer.writerow([
                sp_outpath,
                path,
                'salt_and_pepper',
                f"prob={sp_prob}"
            ])
    
    # Generate noisy images for each split
    print("Processing train set...")
    process_and_save_noisy(train_clean, train_dir, train_writer)
    
    print("Processing val set...")
    process_and_save_noisy(val_clean, val_dir, val_writer)
    
    print("Processing test set...")
    process_and_save_noisy(test_clean, test_dir, test_writer)
    
    # Close CSV files
    train_csv.close()
    val_csv.close()
    test_csv.close()
    
    print("Done! Noisy images generated and CSV files created!")

if __name__ == '__main__':
    main()
