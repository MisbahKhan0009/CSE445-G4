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

The challenge is to upscale low-resolution images to high-resolution ones. Traditional methods like K-Nearest Neighbors (KNN) and Bilinear Interpolation often result in images with larger pixels rather than true high-resolution outputs. Data Processing Inequality suggests that processing data cannot add information content, indicating the need for models that can infer missing details. Deep learning models, such as CNNs and SRGANs, are employed to address this by learning from large datasets to reconstruct high-resolution images.

## Approach

We implemented and evaluated the performance of the four models mentioned above. Each model was trained and tested using the same dataset to ensure a fair comparison. The evaluation metrics included Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) to assess image quality.

## Results

The performance of each model was evaluated based on PSNR and SSIM scores. Detailed results and comparisons are available in the `results` folder.

## Requirements

- **Python**: Version 3.11
- **Libraries**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow`
  - `Pillow`
## Installation steps: 

1. **Clone the Repository**:
   Open your terminal or command prompt and execute:
   ```bash
   git clone https://github.com/MisbahKhan0009/CSE445-G4.git
   ```


   This command creates a local copy of the repository on your machine.

2. **Navigate to the Project Directory**:
   Change into the project directory:
   ```bash
   cd CSE445-G4
   ```



And then set up your environemnt and install all the libraries. Then run the ```main.py``` file.




  
