import csv
import cv2
import random
import numpy as np

def load_and_preprocess_image(img_path, resize_to=True):

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


def generator_from_csv(csv_path, batch_size=8, shuffle=True):
    """
    Generator that yields batches of (noisy_images, clean_images)
    for a denoising task. Modify if your task is classification
    or something else.
    """
    # Read all rows from CSV
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row
        rows = list(reader)
    
    # Optional shuffle once at start
    if shuffle:
        random.shuffle(rows)
    
    idx = 0
    n = len(rows)
    
    while True:
        noisy_batch = []
        clean_batch = []
        
        for _ in range(batch_size):
            if idx >= n:
                idx = 0
                if shuffle:
                    random.shuffle(rows)
            
            # Example row structure:
            # noisy_image_path, clean_image_path, noise_type, noise_params
            row = rows[idx]
            noisy_path = row[0]
            clean_path = row[1]
            
            # Load the images
            noisy_img = load_and_preprocess_image(noisy_path)
            clean_img = load_and_preprocess_image(clean_path)
            
            # If any load fails, skip
            if noisy_img is None or clean_img is None:
                idx += 1
                continue
            
            noisy_batch.append(noisy_img)
            clean_batch.append(clean_img)
            
            idx += 1
        
        yield (np.array(noisy_batch, dtype=np.float32),
               np.array(clean_batch, dtype=np.float32))