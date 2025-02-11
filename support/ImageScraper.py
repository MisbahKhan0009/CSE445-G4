import requests
import os
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("PEXELS_API_KEY")

URL = "https://api.pexels.com/v1/search"
QUERY = "tree"
PER_PAGE = 20  
TOTAL_IMAGES = 100

headers = {"Authorization": API_KEY}


os.makedirs("data/highRes", exist_ok=True)

image_counter = 0
page = 1  
downloaded_urls = set()  

while image_counter < TOTAL_IMAGES:
    params = {"query": QUERY, "per_page": PER_PAGE, "page": page}
    res = requests.get(URL, headers=headers, params=params)

    if res.status_code == 200:
        data = res.json()
        for photo in data.get("photos", []):
            image_url = photo["src"]["original"]

          
            if image_url in downloaded_urls:
                continue
            
            downloaded_urls.add(image_url)

            image_data = requests.get(image_url).content
            filename = f"data/highRes/HR{image_counter + 1:03d}.jpg"

            with open(filename, "wb") as f:
                f.write(image_data)

            image_counter += 1
            if image_counter >= TOTAL_IMAGES:
                break

        page += 1  
    else:
        print("Error fetching images:", res.text)
        break

print(f"Downloaded {image_counter} unique images successfully!")
