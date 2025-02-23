import os
from PIL import Image, ImageFilter

high_res_dir = "data/highRes"
low_res_dir = "data/lowRes"

os.makedirs(low_res_dir, exist_ok=True)

REDUCTION_FACTOR = 10   

high_res_files = os.listdir(high_res_dir)

for index, file_name in enumerate(high_res_files, start=1):
    high_res_path = os.path.join(high_res_dir, file_name)
    
    
    if file_name.lower().endswith('.jpg'):
        try:
            
            image = Image.open(high_res_path)
            
            low_res_image = image.resize(
                (image.width // REDUCTION_FACTOR, image.height // REDUCTION_FACTOR), 
                Image.Resampling.LANCZOS  
            )
            
            low_res_path = os.path.join(low_res_dir, f"LR{index:03d}.jpg")
            
            low_res_image.save(low_res_path, "JPEG")
            print(f"Saved low-res image: {low_res_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print("Image resolution reduction completed!")
