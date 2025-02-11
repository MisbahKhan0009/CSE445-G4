import requests
import random
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PEXELS_API_KEY")

URL = "https://api.pexels.com/v1/search?query=random&per_page=50"
total_images = 100

headers = {"Authorization": API_KEY}

image_counter = 0
os.makedirs("data/highRes", exist_ok=True)

while image_counter < total_images:
    res = requests.get(URL, headers=headers)
    
    if res.status_code == 200:
        data = res.json()
        for photo in data.get("photos",[]):
            image_url = photo["src"]["original"]
            image_data = requests.get(image_url).content
            filename = f"data/highRes/HR{image_counter + 1:03d}.jpg"
            with open(filename, "wb") as f:
                f.write(image_data)
            image_counter += 1
            if image_counter >= total_images:
                break
    else:
        print("Error fetching images:", response.text)
        break

print(f"Downloaded {image_counter} images successfully!")