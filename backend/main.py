from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

from utils.sampler import load_model, generate_digit
from utils.image_tools import numpy_to_base64
from config import T


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

model_path = "model/model.pth"
model = load_model(model_path, device)
print("Model loaded.")

betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_one_minus_ac = torch.sqrt(1 - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

class GenerateRequest(BaseModel):
    digit: int

@app.post("/generate")
def generate(req: GenerateRequest):
    digit = req.digit


    if digit < 0 or digit > 9:
        return {"error": "digit must be between 0 and 9"}
    
    img_np = generate_digit(
        model,
        digit,
        betas,
        sqrt_one_minus_ac,
        sqrt_recip_alphas
    )

    img_base64 = numpy_to_base64(img_np)

    return {
        "digit": digit,
        "image": img_base64
    }

@app.get("/")
def root():
    return {"message": "MNIST Diffusion API is running"}