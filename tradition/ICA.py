import numpy as np
import os
from utils.load_datasets import load_images_from_folder, load_csv_data
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import minmax_scale
import datetime
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体（SimHei），用于Windows
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class Ica:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.data = None
        self.transformed_data = None
        self.label = None
        self.ica = None
        self.original_shape = None

    def set_params(self, config):
        root = config['Datasets']['datasets_path']
        self.data = load_images_from_folder(root + config['Datasets']['test_data'])
        self.label = load_csv_data(root + config['Datasets']['test_label'])
        self.ica = FastICA(n_components=self.n_components)
        self.original_shape = self.data.shape

    def frontend(self, config):
        n_samples, h, w, c = self.data.shape
        self.data = self.data.reshape(n_samples, h * w * c)
        self.data = minmax_scale(self.data)
        self.transformed_data = self.ica.fit_transform(self.data)

    def reconstructed(self, config, num: int):
        reconstructed_data = self.ica.inverse_transform(self.transformed_data)
        n_samples, h, w, c = self.original_shape  # 使用保存的形状

        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = config['save_path'] + "/reconstructed/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, cur_time + '.png')

        plt.figure(figsize=(10, 7))
        for i in tqdm(range(num)):
            img_ica = reconstructed_data[i].reshape(h, w, c)
            img_ica = np.clip(img_ica, 0, 1)  # Ensure values are between 0 and 1

            plt.subplot(2, 5, i + 1)
            plt.imshow(img_ica)
            plt.axis('off')
            plt.title(f'Image {i + 1}')
        plt.suptitle('ICA降维后的图像重建')
        plt.savefig(save_file)

    def visualize(self, config):
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = config['save_path'] + "/visualize/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, cur_time + '.png')

        if self.n_components == 2 or self.n_components == 3:
            plt.figure(figsize=(10, 8))
            ax = plt.subplot(projection='3d') if self.n_components == 3 else plt.subplot()

            color_map = {
                0: 'green',
                1: 'yellow',
                2: 'orange',
                3: 'red',
                4: 'black',
            }

            unique_labels = self.label.iloc[:, 1].unique()

            for label in unique_labels:
                mask = (self.label.iloc[:, 1] == label).to_numpy()
                if self.n_components == 2:
                    ax.scatter(self.transformed_data[mask, 0], self.transformed_data[mask, 1],
                               c=color_map[label], label=f'Class {label}')
                elif self.n_components == 3:
                    ax.scatter(self.transformed_data[mask, 0], self.transformed_data[mask, 1],
                               self.transformed_data[mask, 2], c=color_map[label], label=f'Class {label}')

            plt.legend()
            plt.title('ICA降维后的数据分布')
            plt.savefig(save_file)



