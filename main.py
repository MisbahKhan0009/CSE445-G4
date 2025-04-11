import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def load_images(hr_path, lr_path):
    hr_images = []
    lr_images = []
    
    for i in range(1, 101):  # Assuming 100 images
        hr_file = os.path.join(hr_path, f'HR{i:03d}.jpg')
        lr_file = os.path.join(lr_path, f'LR{i:03d}.jpg')
        
        if os.path.exists(hr_file) and os.path.exists(lr_file):
            hr_img = np.array(Image.open(hr_file).resize((256, 256))) / 255.0
            lr_img = np.array(Image.open(lr_file).resize((64, 64))) / 255.0
            
            hr_images.append(hr_img)
            lr_images.append(lr_img)
    
    return np.array(hr_images), np.array(lr_images)

def build_sr_model():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, 9, padding='same', activation='relu'),
        layers.Conv2D(32, 5, padding='same', activation='relu'),
        layers.Conv2D(32, 5, padding='same', activation='relu'),
        layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        layers.Conv2D(3, 5, padding='same', activation='sigmoid')
    ])
    return model

def evaluate_and_plot(model, lr_img, hr_img, title="Test Image"):
    sr_img = model.predict(lr_img[np.newaxis, ...])[0]
    
    psnr_value = psnr(hr_img, sr_img)
    ssim_value = ssim(hr_img, sr_img, multichannel=True)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(lr_img)
    plt.title('Low Resolution')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(hr_img)
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sr_img)
    plt.title(f'Super Resolved\nPSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return psnr_value, ssim_value

def main():
    # Load images
    hr_path = 'data/highRes'
    lr_path = 'data/lowRes'
    hr_images, lr_images = load_images(hr_path, lr_path)
    
    # Split dataset
    lr_train, lr_test, hr_train, hr_test = train_test_split(
        lr_images, hr_images, test_size=0.2, random_state=42
    )
    
    # Create and compile model
    model = build_sr_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    history = model.fit(
        lr_train, hr_train,
        epochs=50,
        batch_size=8,
        validation_data=(lr_test, hr_test),
        verbose=1
    )
    
    # Evaluate on test images
    print("\nEvaluating test images...")
    psnr_values = []
    ssim_values = []
    
    for i in range(min(5, len(lr_test))):
        psnr_val, ssim_val = evaluate_and_plot(
            model, lr_test[i], hr_test[i], 
            f"Test Image {i+1}"
        )
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    
    print(f"\nAverage PSNR: {np.mean(psnr_values):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")

if __name__ == "__main__":
    main()