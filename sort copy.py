# main.py

import os
import numpy as np
from PIL import Image
import torch
import open_clip
from torchvision import transforms

IMAGE_FOLDER = "training"
EMBEDDING_FILE = "embeddings/vectors.npy"
PATHS_FILE = "embeddings/paths.txt"

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    return model, preprocess

def get_image_embedding(model, preprocess, image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(image).squeeze(0)
    return emb / emb.norm()

def vectorize_images():
    model, preprocess = load_model()
    embeddings = []
    paths = []

    for fname in os.listdir(IMAGE_FOLDER):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            path = os.path.join(IMAGE_FOLDER, fname)
            print(f"Vectorizing: {path}")
            emb = get_image_embedding(model, preprocess, path)
            embeddings.append(emb.numpy())
            paths.append(path)

    os.makedirs("embeddings", exist_ok=True)
    np.save(EMBEDDING_FILE, np.stack(embeddings))
    with open(PATHS_FILE, "w") as f:
        for p in paths:
            f.write(p + "\n")

    print("✅ Embeddings saved!")

if __name__ == "__main__":
    vectorize_images()
