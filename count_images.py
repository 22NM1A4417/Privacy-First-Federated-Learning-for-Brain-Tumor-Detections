import os

# List of dataset directories
dataset_paths = {
    'Final_dataset': r'D:\Brain_Tumor\Brain-Tumor-Classification-using-Federated-Learning-Diffusion-Models\data\final_dataset',
    'Final_Train_data': r'D:\Brain_Tumor\Brain-Tumor-Classification-using-Federated-Learning-Diffusion-Models\data\final_train_data',
    'Final_test_data': r'D:\Brain_Tumor\Brain-Tumor-Classification-using-Federated-Learning-Diffusion-Models\data\final_test_data'
}

# Count images
for dataset_name, path in dataset_paths.items():
    print(f"\n📁 {dataset_name}")
    total = 0
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {class_name}: {count} images")
            total += count
    print(f"  ➤ Total: {total} images")

# Count images in each client
clients_root = r'D:\Brain_Tumor\Brain-Tumor-Classification-using-Federated-Learning-Diffusion-Models\data\final_clients'  
# Go through each client folder
for client_name in os.listdir(clients_root):
    client_path = os.path.join(clients_root, client_name)
    if os.path.isdir(client_path):
        print(f"\n📁 {client_name}")
        total = 0
        for class_name in os.listdir(client_path):
            class_path = os.path.join(client_path, class_name)
            if os.path.isdir(class_path):
                count = len([
                    f for f in os.listdir(class_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                print(f"  {class_name}: {count} images")
                total += count
        print(f"  ➤ Total: {total} images")
