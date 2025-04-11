import os
import numpy as np
from PIL import Image
import random
from tensorflow.keras.preprocessing.image import img_to_array

# Constants for image size
IMAGE_SIZE = 128  # Desired size of high-resolution image (e.g., 128x128)
LOW_RES_SIZE = 32  # Size of low-resolution image (e.g., 32x32)
SEED = 42  # Random seed for reproducibility

# Function to load and prepare images
def load_and_prepare_images(data_dir, seed=None):
    if seed:
        random.seed(seed)
    
    high_res_images = []
    low_res_images = []
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if len(image_files) == 0:
        print(f"No image files found in {data_dir}")
        return None, None

    for file_name in image_files:
        try:
            image_path = os.path.join(data_dir, file_name)
            image = Image.open(image_path).convert('RGB')

            # Skip images smaller than IMAGE_SIZE
            if image.width < IMAGE_SIZE or image.height < IMAGE_SIZE:
                print(f"Skipping {file_name}: Image too small (must be at least {IMAGE_SIZE}x{IMAGE_SIZE}).")
                continue
            
            # Resize the image to standardize size
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

            # Original high-resolution image
            high_res_img = img_to_array(image) / 255.0  # Normalize
            low_res_img = image.resize((LOW_RES_SIZE, LOW_RES_SIZE), Image.Resampling.LANCZOS)
            low_res_img = img_to_array(low_res_img) / 255.0  # Normalize

            high_res_images.append(high_res_img)
            low_res_images.append(low_res_img)

            # Data augmentation with rotations (90, 180, 270 degrees)
            for angle in range(1, 4):
                rotated_image = image.rotate(angle * 90)
                high_res_img = img_to_array(rotated_image) / 255.0
                low_res_img = rotated_image.resize((LOW_RES_SIZE, LOW_RES_SIZE), Image.Resampling.LANCZOS)
                low_res_img = img_to_array(low_res_img) / 255.0

                high_res_images.append(high_res_img)
                low_res_images.append(low_res_img)

            print(f"Processed image: {file_name}")
        
        except Exception as e:
            print(f"Error loading or processing {file_name}: {e}")
            continue

    return np.array(high_res_images), np.array(low_res_images)
