# Text-Conditioned Diffusion on MNIST  

This project demonstrates a minimal working example of a **text-conditioned diffusion model**: given a text description of a digit (zero-nine), the model generates a handwritten style digit image (from the MNIST domain).

It includes both:  
- A training notebook (in Jupyter)  
- A backend API with FastAPI for inference  
- A React + Tailwind frontend for user interaction 

## How to run  

### 1. Training (optional)

If you want to retrain the model from scratch or fine-tune:

- In the notebook:

    - Define beta schedule, forward process & UNet model

    - Train for ~20-50 epochs (or more for better results)

    - Save model.pth into backend/model/

### 2. Backend server

Install dependencies and run FastAPI:

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Frontend

Assuming you have React + Tailwind setup:
```
cd frontend
npm install
npm start
```
1. Then open ://localhost:3000

2. Choose digit description (“zero”, “one”, …)

3. Click Generate → Trigger backend → Display image

## Results
The results of the generation: 
    - Good quality
    ![](images/good.jpg)
    - Medium quality
    ![](images/medium.jpg)
    - Bad quality (more generated)
    ![](images/bad.jpg)

## Model architecture

- **Conditional UNet**: input image + time embedding + label embedding

- Down-sampling blocks → bottleneck → up-sampling blocks + skip-connections

- **Trained as a DDPM**: minimise MSE between model’s predicted noise and actual noise

- Sampling uses classifier-free guidance (optional) for stronger conditioning

## Known issues & tips

- Training only ~20 epochs may yield blurry or ambiguous digits — increasing epochs (50-100) improves clarity.

- Ensure backend architecture matches exactly the model you trained (channel sizes, skip connections) — mismatches cause loading errors.

- For inference on weak GPU/CPU, reduce batch size to 1 and disable gradients.

- Save training metrics (loss, PSNR, cosine similarity) and sample grids every few epochs to monitor progress.