# Update `load_images_from_folder` function in `Data-Dimensionality-Reduction/utils/load_datasets.py`
import pandas as pd
import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm  # Import tqdm for progress bar


def load_images_from_folder(folder: str, imagesize: tuple = (256, 256)):
    images = []
    print("Loading images")
    for filename in tqdm(os.listdir(folder), desc="Loading images"):  # Wrap loop with tqdm
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure image is in RGB format
            img = img.resize((imagesize[0], imagesize[1]))
            img_array = np.array(img)
            images.append(img_array)
    print("Images loaded successfully")
    return np.array(images)


def load_csv_data(file_path: str):
    data = pd.read_csv(file_path)
    return data
