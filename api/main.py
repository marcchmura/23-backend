from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

import os
from fastapi import Query
from starlette.middleware.base import BaseHTTPMiddleware


app = FastAPI()



class CORSMiddlewareForStaticFiles(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/images"):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET"
            response.headers["Access-Control-Allow-Headers"] = "*"
        return response

app.add_middleware(CORSMiddlewareForStaticFiles)


# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

import open_clip
import faiss

# Load model, index, and paths
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
vectors = np.load("embeddings/vectors.npy")
with open("embeddings/paths.txt", "r") as f:
    paths = [line.strip() for line in f.readlines()]
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

# Pydantic models
class TextQuery(BaseModel):
    query: str
    
S3_BASE_URL = "https://pub-1572fd5f6cef49e9bd8dcad74763c24b.r2.dev/"

@app.get("/")
def read_root():
    return {"message": "Good"}

import torch

@app.post("/search_text")
def search_text(body: TextQuery, offset: int = Query(0), limit: int = Query(100)):
    tokenized = tokenizer([body.query])
    with torch.no_grad():
        text_feat = model.encode_text(tokenized).squeeze(0)
    text_feat = text_feat / text_feat.norm()
    scores, indices = index.search(text_feat.unsqueeze(0).numpy(), offset + limit)
    results = [
        {"url": f"{os.path.basename(paths[i])}", "score": float(scores[0][j])}
        for j, i in enumerate(indices[0][offset:offset + limit])
    ]
    return results

from fastapi import HTTPException

@app.get("/search_image")
def search_image_by_name(filename: str, offset: int = Query(0), limit: int = Query(100)):
    try:
        idx = paths.index(filename)
    except ValueError:
        raise HTTPException(status_code=404, detail="Image not found in database.")

    query_vector = vectors[idx]
    query_vector = query_vector / np.linalg.norm(query_vector)

    scores, indices = index.search(np.expand_dims(query_vector, axis=0), offset + limit)
    results = [
        {"url": f"{os.path.basename(paths[i])}", "score": float(scores[0][j])}
        for j, i in enumerate(indices[0][offset:offset + limit])
    ]
    return results

