from sklearn.decomposition import FastICA
from utils.load_datasets import load_images_from_folder, load_csv_data
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体（SimHei），用于Windows
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class Ica:
    def __init__(self, n_components: int):
        self.ica = FastICA(n_components=n_components)
        self.data = None
        self.label = None
        self.data_mean = None
        self.n_components = n_components
        self.ica = None
        self.S = None
        self.W = None
