

!pip install --upgrade pip
!pip install torch torchvision --upgrade
!pip install diffusers==0.10.2 transformers scipy
!pip install --upgrade git+https://github.com/huggingface/diffusers.git
!pip install cuda-python
!pip install cupy-cuda11x



import torch
from diffusers import StableDiffusionPipeline
from google.colab import files

# Assuming necessary installations and setup from previous code blocks

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda") #Ensure the model is on the GPU

prompt = "Seductive Woman, 20 Years old, Selfie, Closeup Portrait, Full body, piercing green eyes, long curly brown hair, flawless skin"

with torch.autocast("cuda"):
    image = pipe(prompt).images[0]

image.save("nexus_prime_queen.png")
files.download("nexus_prime_queen.png")
print("Image generated, saved, and downloaded as nexus_prime_queen.png")
