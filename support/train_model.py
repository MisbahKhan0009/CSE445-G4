import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Suppress TensorFlow info logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Paths to image directories
HR_DIR = "data/highRes"
LR_DIR = "data/lowRes"

# Image and patch config
IMG_SIZE = (256, 256, 3)
PATCH_SIZE = 16
STRIDE = 8  # For overlapping patches

# Load images from a directory
def load_images(directory):
    images = []
    for file_name in sorted(os.listdir(directory)):
        img_path = os.path.join(directory, file_name)
        img = load_img(img_path, target_size=IMG_SIZE[:2])
        img = img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)

# Extract overlapping patches
def extract_patches(image, patch_size=PATCH_SIZE, stride=STRIDE):
    h, w, c = image.shape
    patches = []
    positions = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size, :].flatten()
            patches.append(patch)
            positions.append((i, j))
    return np.array(patches), positions

# Load datasets
x_train = load_images(LR_DIR)
y_train = load_images(HR_DIR)
print(f"Loaded {x_train.shape[0]} low-res and {y_train.shape[0]} high-res images.")

# Extract and prepare training patches
X_train_patches = []
Y_train_patches = []

for lr_img, hr_img in zip(x_train, y_train):
    lr_patches, _ = extract_patches(lr_img)
    hr_patches, _ = extract_patches(hr_img)
    X_train_patches.append(lr_patches)
    Y_train_patches.append(hr_patches)

X_train_patches = np.vstack(X_train_patches)
Y_train_patches = np.vstack(Y_train_patches)

# Normalize patches
scaler_X = StandardScaler().fit(X_train_patches)
scaler_Y = StandardScaler().fit(Y_train_patches)

X_train_scaled = scaler_X.transform(X_train_patches)
Y_train_scaled = scaler_Y.transform(Y_train_patches)

# Train XGBoost model
xgb_regressor = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    objective="reg:squarederror",
    tree_method="hist",
    device="cuda"
)

xgb_regressor.fit(X_train_scaled, Y_train_scaled)
print("✅ XGBoost model trained successfully with GPU!")

# Select test image
image_index = 0
sample_lr_img = x_train[image_index]
sample_hr_img = y_train[image_index]
sample_file_name = sorted(os.listdir(LR_DIR))[image_index]
print(f"🔍 Processing image: {sample_file_name}")

# Extract and scale test patches
lr_patches, positions = extract_patches(sample_lr_img)
lr_scaled = scaler_X.transform(lr_patches)

# Predict and inverse transform
predicted_scaled = xgb_regressor.predict(lr_scaled)
predicted_patches = scaler_Y.inverse_transform(predicted_scaled)

# Reconstruct predicted image
reconstructed_img = np.zeros_like(sample_lr_img)
count_img = np.zeros_like(sample_lr_img)

for (i, j), patch in zip(positions, predicted_patches):
    patch = patch.reshape((PATCH_SIZE, PATCH_SIZE, 3))
    reconstructed_img[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :] += patch
    count_img[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :] += 1

# Normalize overlapping pixels
reconstructed_img /= np.maximum(count_img, 1)

# Sharpen image
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
sharpened_img = cv2.filter2D(reconstructed_img, -1, sharpen_kernel)
sharpened_img = np.clip(sharpened_img, 0, 1)

# Evaluation
psnr_val = psnr(sample_hr_img, sharpened_img)
ssim_val = ssim(sample_hr_img, sharpened_img, channel_axis=2)
print(f"📈 PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

# Display results
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(sample_lr_img)
plt.title(f"Low-Resolution #{image_index+1}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sample_hr_img)
plt.title("Original High-Resolution")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened_img)
plt.title(f"XGBoost Super-Resolved\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
plt.axis("off")

plt.tight_layout()
plt.show()
