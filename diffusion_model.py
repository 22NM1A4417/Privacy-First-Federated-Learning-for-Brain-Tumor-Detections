import os
from PIL import Image, ImageEnhance
import random

# Define directories for resized and augmented images
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
final_train_dir = 'data/final_train_data'
final_test_dir = 'data/final_test_data'
augmented_train_dir = 'data/augmented_train_data'
final_dataset_dir = 'data/final_dataset'  # New final dataset directory

# Create directories for resized and augmented images
os.makedirs(final_train_dir, exist_ok=True)
os.makedirs(final_test_dir, exist_ok=True)
os.makedirs(augmented_train_dir, exist_ok=True)
os.makedirs(final_dataset_dir, exist_ok=True)  # Create final dataset directory

# Ensure source directories exist
training_source_dir = 'data/brain_tumor_dataset/Training'
testing_source_dir = 'data/brain_tumor_dataset/Testing'

if not os.path.exists(training_source_dir):
    os.makedirs(training_source_dir)
    print(f"Training dataset directory created at {training_source_dir}. Please add data.")

if not os.path.exists(testing_source_dir):
    os.makedirs(testing_source_dir)
    print(f"Testing dataset directory created at {testing_source_dir}. Please add data.")

def resize_images(source_dir, target_dir):
    total_resized = 0
    total_errors = 0

    for category in categories:
        category_path = os.path.join(source_dir, category)
        save_path = os.path.join(target_dir, category)
        os.makedirs(save_path, exist_ok=True)

        if not os.path.exists(category_path):
            print(f"Warning: Source category directory {category_path} not found.")
            continue

        files = os.listdir(category_path)
        for img_name in files:
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path)
                # Fix: convert mode P as well before saving as JPEG
                if img.mode in ['L', 'RGBA', 'P']:
                    img = img.convert('RGB')
                img = img.resize((224, 224))
                img.save(os.path.join(save_path, img_name))
                total_resized += 1
            except Exception as e:
                print(f"Error resizing image {img_path}: {e}")
                total_errors += 1

    print(f"Resizing completed: {total_resized} images resized successfully, {total_errors} errors encountered.")

def augment_image(img):
    try:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        img = img.rotate(random.randint(-10, 10))
        return img
    except Exception as e:
        print(f"Error during image augmentation: {e}")
        return None

def augment_images(source_dir, target_dir, augmentation_ratio=0.5):  # increased from 0.25 to 0.5
    total_augmented = 0
    total_errors = 0

    for category in categories:
        category_path = os.path.join(source_dir, category)
        save_path = os.path.join(target_dir, category)
        os.makedirs(save_path, exist_ok=True)

        if not os.path.exists(category_path):
            print(f"Warning: Source category directory {category_path} not found.")
            continue

        files = os.listdir(category_path)
        num_augments = int(augmentation_ratio * len(files))

        for img_name in files:
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    total_errors += 1
                    continue

                for i in range(num_augments // len(files) + 1):
                    augmented = augment_image(img)
                    if augmented is not None:
                        augmented.save(os.path.join(save_path, f"aug_{i}_{img_name}"))
                        total_augmented += 1
                    else:
                        total_errors += 1
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                total_errors += 1

    print(f"Augmentation completed: {total_augmented} images augmented successfully, {total_errors} errors encountered.")

# New function to create final dataset
def create_final_dataset():
    resize_images(training_source_dir, final_train_dir)
    resize_images(testing_source_dir, final_test_dir)

    augment_images(final_train_dir, augmented_train_dir, augmentation_ratio=0.5)  # increased augmentation

    for category in categories:
        augmented_category_path = os.path.join(augmented_train_dir, category)
        final_category_path = os.path.join(final_dataset_dir, category)
        os.makedirs(final_category_path, exist_ok=True)

        for img_name in os.listdir(augmented_category_path):
            img_path = os.path.join(augmented_category_path, img_name)
            try:
                img = Image.open(img_path)
                img.save(os.path.join(final_category_path, img_name))
            except Exception as e:
                print(f"Error saving augmented image {img_name}: {e}")

    total_images = sum(len(os.listdir(os.path.join(final_dataset_dir, category))) for category in categories)
    if total_images < 7616:
        raise ValueError("Final dataset size is insufficient. Consider increasing augmentation.")
    print(f"Final dataset created with a total of {total_images} images.")

if __name__ == "__main__":
    create_final_dataset()
