import pandas as pd
import os
import numpy as np
from PIL import Image


def load_images_from_folder(folder: str, imagesize: tuple = (256,256)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            img = img.convert('RGB')  # 确保图像是RGB格式
            img = img.resize((imagesize[0], imagesize[1]))
            img_array = np.array(img)
            images.append(img_array)
    return np.array(images)


def load_csv_data(file_path: str):
    data = pd.read_csv(file_path)
    return data