import numpy as np
import os

# Paths
existing_vectors_path = "embeddings_save/vectors.npy"
new_vectors_path = "embeddings/new_vectors.npy"

existing_paths_path = "embeddings_save/paths.txt"
new_paths_path = "embeddings/new_paths.txt"

output_vectors_path = "embeddings/vectors_new.npy"
output_paths_path = "embeddings/paths_new.txt"

# Load vectors
print("📦 Loading vector files...")
existing_vectors = np.load(existing_vectors_path)
new_vectors = np.load(new_vectors_path)

# Combine vectors
print(f"🔗 Combining vectors: {existing_vectors.shape} + {new_vectors.shape}")
combined_vectors = np.vstack([existing_vectors, new_vectors])
np.save(output_vectors_path, combined_vectors)
print(f"✅ Saved combined vectors to {output_vectors_path} (shape: {combined_vectors.shape})")

# Merge paths
print("📂 Merging paths.txt files...")
with open(existing_paths_path, "r") as f1, open(new_paths_path, "r") as f2:
    all_lines = f1.readlines() + f2.readlines()

with open(output_paths_path, "w") as f:
    f.writelines(all_lines)

print(f"✅ Saved combined paths to {output_paths_path} (total: {len(all_lines)} lines)")
