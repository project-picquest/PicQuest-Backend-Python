from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/submitimage")
def image_similarity_measure(first_image: UploadFile, second_image: UploadFile):
    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load two images
    image1 = Image.open(first_image.file)
    image2 = Image.open(second_image.file)

    # Preprocess and extract features
    inputs1 = processor(images=image1, return_tensors="pt")
    inputs2 = processor(images=image2, return_tensors="pt")
    features1 = model.get_image_features(**inputs1)
    features2 = model.get_image_features(**inputs2)

    # Normalize and compute cosine similarity
    features1 = features1 / features1.norm(dim=-1, keepdim=True)
    features2 = features2 / features2.norm(dim=-1, keepdim=True)

    similarity = torch.matmul(features1, features2.T)
    return {"유사도": similarity.item()}