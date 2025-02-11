# Super-Resolution Image Restoration

API docs: API docs: https://www.pexels.com/api/documentation/#photos-search__response__photos

## Overview

This project focuses on building a machine learning-based **super-resolution model** to restore low-resolution images to their original quality. The dataset consists of 100 randomly acquired images from Unsplash using `gallery-dl`. The images are artificially degraded by **blurring** or **undersampling**, and the model learns to reconstruct high-resolution versions.

## Project Workflow

1. **Dataset Collection**

   - Download 100 random images from Unsplash using `gallery-dl`.
   - Store the images in a structured dataset directory.

2. **Preprocessing**

   - Downsample images using bicubic interpolation.
   - Apply Gaussian blur to create low-resolution inputs.
   - Store high-resolution (HR) and low-resolution (LR) image pairs.

3. **Model Training**

   - Train a super-resolution model using any ML architecture (SRCNN, EDSR, ESRGAN, or custom CNN/Transformer-based models).
   - Use high-resolution images as ground truth and low-resolution images as input.

4. **Evaluation**

   - Use PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) as metrics.
   - Compare restored images with original high-resolution versions.

5. **Testing & Inference**
   - Test the model on unseen images.
   - Deploy the model for real-time super-resolution tasks.

## Installation

### Prerequisites

- Python 3.x
- `gallery-dl` (for downloading images from Unsplash)
- TensorFlow / PyTorch
- OpenCV
- NumPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/super-resolution.git
cd super-resolution

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install gallery-dl for downloading images
pip install gallery-dl
```

## License

MIT License

## Group: 4

### Wiki

[Project Wiki](<[https://github.com/MisbahKhan0009/CSE445-G4](https://github.com/MisbahKhan0009/CSE445-G4/wiki)>)

## Contributors

- **[2132089642] Md. Misbah Khan** ([GitHub](https://github.com/MisbahKhan0009/))
- **[2013823642] Raju Ahamed Rabby** ([GitHub](https://github.com/ahamedrabby123))
- **[2212592042] Md. Abdula Al Shyed** ([GitHub](https://github.com/AbdulaAlShyed-2212592042))
- **[2212058642] Rakibul Islam** ([GitHub](https://github.com/Rakib-28169-islam))

---
