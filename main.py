import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import xgboost as xgb
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Suppress oneDNN informational logs from TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Directories
HR_DIR = "data/highRes"
LR_DIR = "data/lowRes"

# Image size and patch settings
IMG_SIZE = (256, 256, 3)  # Resize images to 256x256 for better resolution
PATCH_SIZE = 16  # Increased patch size to 16x16

# Function to load images
def load_images(directory):
    images = []
    for file_name in sorted(os.listdir(directory)):
        img_path = os.path.join(directory, file_name)
        img = load_img(img_path, target_size=IMG_SIZE[:2])  # Resize image
        img = img_to_array(img) / 255.0  # Normalize pixel values
        images.append(img)
    return np.array(images)

# Load the dataset
x_train = load_images(LR_DIR)  # Low-resolution images
y_train = load_images(HR_DIR)  # High-resolution images

# Check if the images are loaded correctly
print(f"Loaded {x_train.shape[0]} low-resolution images.")
print(f"Loaded {y_train.shape[0]} high-resolution images.")
print(f"Shape of first low-resolution image: {x_train[0].shape}")
print(f"Shape of first high-resolution image: {y_train[0].shape}")

# Function to extract patches from images
def extract_patches(image, patch_size=PATCH_SIZE):
    h, w, c = image.shape
    patches = []
    positions = []
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :].flatten()
            patches.append(patch)
            positions.append((i, j))
    return np.array(patches), positions

# Extract patches from training images
X_train_patches = []
Y_train_patches = []

for lr_img, hr_img in zip(x_train, y_train):
    lr_patches, _ = extract_patches(lr_img)
    hr_patches, _ = extract_patches(hr_img)
    
    X_train_patches.append(lr_patches)
    Y_train_patches.append(hr_patches)

X_train_patches = np.vstack(X_train_patches)
Y_train_patches = np.vstack(Y_train_patches)

# Train XGBoost model (GPU Enabled)
xgb_regressor = xgb.XGBRegressor(
    n_estimators=200,   # Increased number of trees
    learning_rate=0.05,  # Reduced learning rate for better convergence
    max_depth=8,         # Increased tree depth for capturing more complex features
    objective="reg:squarederror",
    tree_method="hist",  # Enables GPU acceleration
    device="cuda"        # Use CUDA for GPU support
)

xgb_regressor.fit(X_train_patches, Y_train_patches)
print("XGBoost model trained successfully with GPU!")

# Select an image to upscale
image_index = 0  # Change this to test different images
sample_lr_img = x_train[image_index]
sample_hr_img = y_train[image_index]  # Original high-resolution image
sample_file_name = sorted(os.listdir(LR_DIR))[image_index]

# Check if the low-resolution image is being selected
print(f"Processing image: {sample_file_name}")

# Extract patches from the test image
lr_patches, positions = extract_patches(sample_lr_img)

# Predict high-resolution patches using XGBoost
predicted_patches = xgb_regressor.predict(lr_patches)

# Reconstruct the high-resolution image
reconstructed_img = np.zeros_like(sample_lr_img)
count_img = np.zeros_like(sample_lr_img)  # To handle overlapping patches

for (i, j), patch in zip(positions, predicted_patches):
    patch = patch.reshape((PATCH_SIZE, PATCH_SIZE, 3))  # Reshape to image format
    reconstructed_img[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :] += patch
    count_img[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :] += 1

# Normalize overlapping patches
reconstructed_img /= np.maximum(count_img, 1)

# Apply sharpening filter for better quality
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
sharpened_img = cv2.filter2D(reconstructed_img, -1, kernel)

# Check if the images are being reconstructed correctly
print(f"Reconstructed image shape: {sharpened_img.shape}")

# Display results
plt.figure(figsize=(18, 6))

# Low-resolution image
plt.subplot(1, 3, 1)
plt.imshow(sample_lr_img)
plt.title(f"Low-Resolution Image #{image_index+1} ({sample_file_name})")
plt.axis("off")

# Original high-resolution image
plt.subplot(1, 3, 2)
plt.imshow(sample_hr_img)
plt.title(f"Original High-Resolution Image #{image_index+1}")
plt.axis("off")

# Predicted high-resolution image
plt.subplot(1, 3, 3)
plt.imshow(np.clip(sharpened_img, 0, 1))  # Clip values for visualization
plt.title(f"XGBoost Super-Resolved Image #{image_index+1}")
plt.axis("off")

plt.show()
