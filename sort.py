import os
import json
import numpy as np
import torch
from PIL import Image
import open_clip
import cv2

IMAGE_FOLDER = "training"
VIDEO_FOLDER = "training"
FRAME_FOLDER = "video_frames"
EMBEDDING_FILE = "embeddings/vectors.npy"
PATHS_FILE = "embeddings/paths.jsonl"  # Changed to JSON lines for richer metadata

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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
    cap.release()
    return success

def vectorize_all_media():
    model, preprocess = load_model()
    embeddings = []
    metadata = []

    # Process images
    for fname in os.listdir(IMAGE_FOLDER):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            path = os.path.join(IMAGE_FOLDER, fname)
            print(f"🖼️ Vectorizing image: {path}")
            emb = get_image_embedding(model, preprocess, path)
            embeddings.append(emb.numpy())
            metadata.append({
                "type": "image",
                "url": f"/images/{fname}"
            })

    # Process videos
    for fname in os.listdir(VIDEO_FOLDER):
        if fname.lower().endswith(('.mp4', '.mov', '.webm')):
            video_path = os.path.join(VIDEO_FOLDER, fname)
            frame_path = os.path.join(FRAME_FOLDER, f"{os.path.splitext(fname)[0]}.jpg")

            print(f"🎥 Extracting middle frame from video: {video_path}")
            if extract_middle_frame(video_path, frame_path):
                emb = get_image_embedding(model, preprocess, frame_path)
                embeddings.append(emb.numpy())
                metadata.append({
                    "type": "video",
                    "url": f"/frames/{os.path.basename(frame_path)}",
                    "video": f"/videos/{fname}"
                })

    # Save outputs
    os.makedirs("embeddings", exist_ok=True)
    np.save(EMBEDDING_FILE, np.stack(embeddings))
    with open(PATHS_FILE, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Saved {len(embeddings)} embeddings (images + videos)!")

if __name__ == "__main__":
    vectorize_all_media()
