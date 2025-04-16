import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import uvicorn
from pyngrok import ngrok
import nest_asyncio

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model (same as previous implementation)
class SRModel(nn.Module):
    def __init__(self, input_shape):
        super(SRModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize FastAPI app
app = FastAPI(title="Super Resolution Image Enhancer")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model
input_shape = (3, 128, 128)  # Expected input shape: (C, H, W)
model = SRModel(input_shape).to(device)
model.load_state_dict(torch.load('super_resolution_model_200epoch.pth'))
model.eval()

# Function to preprocess uploaded image
def preprocess_image(image, target_size=(128, 128)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)
    print(f"Preprocessed image shape: {img_tensor.shape}")
    return img_tensor

# Function to enhance image
def enhance_image(model, lr_tensor):
    lr_tensor = lr_tensor.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        sr_tensor = model(lr_tensor)[0].cpu()
    sr_img = sr_tensor.permute(1, 2, 0).numpy()
    sr_img = np.clip(sr_img * 255.0, 0, 255).astype(np.uint8)
    print(f"Super-resolved image array shape: {sr_img.shape}, min: {sr_img.min()}, max: {sr_img.max()}")
    return Image.fromarray(sr_img)

# FastAPI endpoint to handle image upload and return super-resolved image
@app.post("/enhance")
async def enhance_image_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert('RGB')
    print(f"Received image size: {img.size}")
    
    lr_tensor = preprocess_image(img)
    sr_img = enhance_image(model, lr_tensor)
    print(f"Generated super-resolved image size: {sr_img.size}")
    
    output_buffer = BytesIO()
    sr_img.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    
    return StreamingResponse(output_buffer, media_type="image/png")

# Set up ngrok and run the server
def run_server():
    nest_asyncio.apply()
    
    # Set ngrok auth token
    ngrok.set_auth_token("2owD7wYZFV3m4i6Lfav6hYkKGFz_59T3nd2QpgfW5nMXTkXQA")  # Replace with your ngrok token
    
    # Ensure no existing tunnels are open
    try:
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            ngrok.disconnect(tunnel.public_url)
    except:
        pass
    
    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print(f"FastAPI server running at: {public_url}")
    
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()