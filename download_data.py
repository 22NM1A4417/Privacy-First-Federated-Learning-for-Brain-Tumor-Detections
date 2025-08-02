import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """
    Download dataset from Kaggle and unzip it into the data directory.
    """
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Define the Kaggle dataset path and destination file
    dataset_path = 'masoudnickparvar/brain-tumor-mri-dataset'  
    download_path = 'data/brain-tumor-mri-dataset.zip'  

    # Create a 'data' directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Download the dataset
    print("Downloading dataset...")
    api.dataset_download_files(dataset_path, path='data', unzip=False)

    # Check if the download was successful
    if not os.path.exists(download_path):
        print(f"Error: The file {download_path} does not exist. Download may have failed.")
        return

    # Unzip the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall('data/brain_tumor_dataset')  # Define extraction path
    print("Download and extraction complete.")

    # Clean up the downloaded zip file
    os.remove(download_path)

if __name__ == "__main__":
    download_dataset()
