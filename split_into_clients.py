import os
import shutil
import random
from pathlib import Path

# Source directory
source_dir = r'D:\Brain_Tumor\Brain-Tumor-Classification-using-Federated-Learning-Diffusion-Models\data\final_dataset'

# Target base directory
target_base = r'D:\Brain_Tumor\Brain-Tumor-Classification-using-Federated-Learning-Diffusion-Models\data\final_clients'
os.makedirs(target_base, exist_ok=True)

# Clients
client_names = [f'client_{i+1}' for i in range(6)]

# Create folder structure for each client
classes = os.listdir(source_dir)
for client in client_names:
    for cls in classes:
        os.makedirs(os.path.join(target_base, client, cls), exist_ok=True)

# Split logic
for cls in classes:
    class_path = os.path.join(source_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    # Divide images evenly among clients
    split_images = [images[i::6] for i in range(6)]

    for i, client in enumerate(client_names):
        for img in split_images[i]:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_base, client, cls, img)
            shutil.copy2(src, dst)

print("✅ Dataset successfully split into 6 clients!")
