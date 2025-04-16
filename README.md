# Image Super-Resolution Project

## Overview

This project focuses on enhancing the resolution of low-quality images using various machine learning and deep learning models. The primary objective is to improve image quality by generating high-resolution outputs from low-resolution inputs, preserving the original content and structure.

## Team Members

- **Md. Misbah Khan** (ID: 2132089642)
- **Md Abdula Al Shyed** (ID: 2212592042)
- **Rakibul Islam** (ID: 2212058642)
- **Raju Ahamed Rabby** (ID: 2212592042)

## Models Implemented

1. **XGBoost**: An open-source machine learning library known for its speed and efficiency in supervised learning tasks.
2. **Random Forest**: An ensemble learning method that constructs multiple decision trees to improve predictive performance.
3. **Convolutional Neural Networks (CNNs)**: Deep learning models designed for processing structured grid data, such as images, by utilizing convolutional layers to automatically learn spatial hierarchies of features.
4. **Super-Resolution Generative Adversarial Networks (SRGANs)**: Deep learning models that enhance image resolution by generating high-quality images from low-resolution inputs.

## Dataset Preparation

- **Source**: Downloaded 100 high-resolution images from Pexels.
- **Processing**: Resized images using the Pillow library and saved them in the `data/highRes` folder with sequential filenames (HR001.jpg, HR002.jpg, etc.).
- **Downsampling**: Applied a factor of 10 using the Lanczos resampling filter to create low-resolution images, saved in the `data/lowRes` folder with corresponding filenames (LR001.jpg, LR002.jpg, etc.).

## Problem Statement

The challenge is to upscale low-resolution images to high-resolution ones. Traditional methods like K-Nearest Neighbors (KNN) and Bilinear Interpolation often result in images with larger pixels rather than true high-resolution outputs. 

According to the **Data Processing Inequality**, no algorithm can increase the information content of an image. Therefore, advanced models like CNNs and SRGANs are needed to learn patterns from data and infer the missing high-frequency details necessary for realistic upscaling.

## Approach

We implemented and evaluated the performance of the four models mentioned above. Each model was trained and tested using the same dataset to ensure a fair comparison. The evaluation metrics included:

- **Peak Signal-to-Noise Ratio (PSNR)**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
- **Structural Similarity Index Measure (SSIM)**: Measures the similarity between two images based on luminance, contrast, and structure.

## Results

The performance of each model was evaluated based on PSNR and SSIM scores. The **XGBoost** and **Random Forest** models provided decent results but lacked the high-frequency details needed for sharp outputs.

The **CNN** and **SRGAN** models performed significantly better, with SRGAN producing visually appealing high-resolution images. Detailed numerical results and visual comparisons are provided in the `results` folder.

## Requirements

- **Python**: Version 3.11 or above

### Required Libraries

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `xgboost`
- `opencv-python`
- `matplotlib`
- `Pillow`
- `scikit-image` *(optional, for SSIM)*

## Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MisbahKhan0009/CSE445-G4.git


   This command creates a local copy of the repository on your machine.

2. **Navigate to the Project Directory**:
   Change into the project directory:
   ```bash
   cd CSE445-G4
   ```
3. **Install Dependencies: You can install all required libraries using**:
   pip install -r requirements.txt
   
4. Run the Main Script:
   python main.py


And then set up your environemnt and install all the libraries. Then run the ```main.py``` file.

This is the url of our website: https://cse445-g4.pages.dev/




  
