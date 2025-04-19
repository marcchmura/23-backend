import os
import numpy as np
import torch
import cv2
from PIL import Image
import open_clip

IMAGE_FOLDER = "training"
EMBEDDING_FILE = "embeddings/vectors.npy"
PATHS_FILE = "embeddings/paths.txt"  # plain text paths only

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    return model, preprocess

def get_image_embedding(model, preprocess, image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(image).squeeze(0)
    return emb / emb.norm()

def extract_middle_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_path, frame)
    cap.release()
    return success

def vectorize_all_media():
    model, preprocess = load_model()
    embeddings = []
    paths = []

    # Step 1: Vectorize existing image files in folder
    for fname in os.listdir(IMAGE_FOLDER):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            path = os.path.join(IMAGE_FOLDER, fname)
            print(f"🖼️ Vectorizing image: {path}")
            emb = get_image_embedding(model, preprocess, path)
            embeddings.append(emb.numpy())
            paths.append(path)

    # Step 2: Extract middle frame from each video and treat as image
    for fname in os.listdir(IMAGE_FOLDER):
        if fname.lower().endswith(('.mp4', '.mov', '.webm')):
            video_path = os.path.join(IMAGE_FOLDER, fname)
            frame_filename = os.path.splitext(fname)[0] + ".jpg"
            frame_path = os.path.join(IMAGE_FOLDER, frame_filename)

            if os.path.exists(frame_path):  # skip if already exists
                continue

            print(f"🎥 Extracting frame from: {video_path}")
            if extract_middle_frame(video_path, frame_path):
                emb = get_image_embedding(model, preprocess, frame_path)
                embeddings.append(emb.numpy())
                paths.append(frame_path)

    # Save all to disk
    os.makedirs("embeddings", exist_ok=True)
    np.save(EMBEDDING_FILE, np.stack(embeddings))

    with open(PATHS_FILE, "w") as f:
        for p in paths:
            f.write(p + "\n")

    print(f"✅ Vectorized {len(embeddings)} files (images + video frames)")

if __name__ == "__main__":
    vectorize_all_media()
