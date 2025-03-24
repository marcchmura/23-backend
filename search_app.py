# search_app.py

import streamlit as st
import numpy as np
import faiss
from PIL import Image
import torch
import open_clip
import os

EMBEDDING_FILE = "embeddings/vectors.npy"
PATHS_FILE = "embeddings/paths.txt"

@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

@st.cache_resource
def load_index():
    vectors = np.load(EMBEDDING_FILE)
    with open(PATHS_FILE, "r") as f:
        paths = [line.strip() for line in f.readlines()]
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return vectors, paths, index

def get_text_embedding(model, tokenizer, text):
    tokenized = tokenizer([text])
    with torch.no_grad():
        text_feat = model.encode_text(tokenized).squeeze(0)
    return text_feat / text_feat.norm()

def get_image_embedding(model, preprocess, image):
    image = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(image).squeeze(0)
    return emb / emb.norm()

def main():
    st.title("🔍 Same Energy - Text & Image Similarity Search")

    model, preprocess, tokenizer = load_model()
    vectors, paths, index = load_index()

    st.subheader("Text Search")
    query_text = st.text_input("Search for something visual (e.g. 'red car', 'mountain sunset')")

    if "refined_vector" not in st.session_state:
        st.session_state.refined_vector = None

    if query_text:
        st.write(f"Searching for: **{query_text}**")
        query_vector = get_text_embedding(model, tokenizer, query_text).numpy().reshape(1, -1)
        scores, indices = index.search(query_vector, 15)

    st.subheader("🔎 Results")

    query_vector = None

    # 1. If user entered text, use CLIP to embed text
    if query_text and st.session_state.refined_vector is None:
        query_vector = get_text_embedding(model, tokenizer, query_text).numpy().reshape(1, -1)

    # 2. If user clicked "More like this", use the image vector
    elif st.session_state.refined_vector is not None:
        query_vector = st.session_state.refined_vector

    # 3. Run similarity search
    if query_vector is not None:
        scores, indices = index.search(query_vector, 10)
        selected_vector = None
        result_indices = indices[0]

        # Make the first result the refined query image (if refining)
        if st.session_state.refined_vector is not None:
            result_indices = [st.session_state.refined_index] + [i for i in indices[0] if i != st.session_state.refined_index]

        cols = st.columns(5)
        for i, idx in enumerate(result_indices):
            with cols[i % 5]:
                st.image(paths[idx], use_container_width=True, caption=f"Result {i+1}")
                if st.button("More like this", key=f"btn_{i}"):
                    st.session_state.refined_vector = vectors[idx].reshape(1, -1)
                    st.session_state.refined_index = idx
                    st.rerun()

                if st.button("🔁 Reset Search"):
                    st.session_state.refined_vector = None
                    st.session_state.refined_index = None
                    st.rerun()

if __name__ == "__main__":
    main()
