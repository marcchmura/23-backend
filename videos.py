import os
import cv2

VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "training"

def extract_middle_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_path, frame)
        print(f"✅ Saved: {output_path}")
    cap.release()

for fname in os.listdir(VIDEO_FOLDER):
    if fname.lower().endswith((".mp4", ".mov", ".webm")):
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(OUTPUT_FOLDER, f"video_{base}.jpg")
        video_path = os.path.join(VIDEO_FOLDER, fname)

        if not os.path.exists(out_path):
            extract_middle_frame(video_path, out_path)
