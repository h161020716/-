import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch
import os
import pandas as pd
from utils.load_datasets import load_images_from_folder, load_csv_data

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体（SimHei），用于Windows
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class Pca:
    def __init__(self, n: int):
        self.data = None
        self.label = None
        self.data_mean = None
        self.n_components = n
        self.pca = None
        self.U = None
        self.S = None
        self.Vt = None
        self.h = None
        self.w = None
        self.c = None

    def set_params(self, config: dict):
        root = config['Datasets']['datasets_path']
        self.data = load_images_from_folder(root + config['Datasets']['test_data'])
        self.label = load_csv_data(root + config['Datasets']['test_label'])

    def frontend(self, config: dict):
        n_samples, self.h, self.w, self.c = self.data.shape

        self.data = torch.tensor(
            self.data.reshape(
                n_samples, self.h * self.w * self.c),
            dtype=torch.float32).to(config['Device'])

        self.data_mean = torch.mean(self.data, dim=0)

        data_centered = self.data - self.data_mean

        self.U, self.S, self.Vt = torch.svd(data_centered)

        self.pca = torch.matmul(data_centered, self.Vt[:, :self.n_components])

    def reconstructed(self, config: dict, num: int):
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = config['save_path'] + "/reconstructed/"

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file = os.path.join(save_path, cur_time + '.png')

        plt.figure(figsize=(10, 7))
        for i in range(num):  # 显示前10个降维后的图像
            # 还原降维后的图像
            img_pca = torch.matmul(self.pca[i], self.Vt[:, :self.n_components].T) + self.data_mean  # 重建图像
            img_pca = img_pca.reshape(self.h, self.w, self.c)  # 将图像恢复原始形状
            img_pca = img_pca.cpu().numpy()  # 将图像数据移动回CPU

            # 可视化
            plt.subplot(2, 5, i + 1)
            plt.imshow(np.clip(img_pca, 0, 255).astype(np.uint8))  # 将值截断到0-255并转换为uint8类型
            plt.axis('off')
            plt.title(f'Image {i + 1}')
        plt.suptitle('PCA降维后的图像重建')
        plt.savefig(save_file)

    def visualize(self, config: dict):
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = config['save_path'] + "/visualize/"

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file = os.path.join(save_path, cur_time + '.png')

        # 2D或3D可视化降维后的数据
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

            unique_labels = self.label.iloc[:, 1].unique()  # 获取唯一的标签

            for label in unique_labels:
                # 为每个标签创建布尔索引
                mask = (self.label.iloc[:, 1] == label).to_numpy()  # 转换为 NumPy 数组
                if self.n_components == 2:
                    ax.scatter(self.pca[mask, 0].cpu().numpy(), self.pca[mask, 1].cpu().numpy(),
                               c=color_map[label], label=f'Class {label}')
                elif self.n_components == 3:
                    ax.scatter(self.pca[mask, 0].cpu().numpy(), self.pca[mask, 1].cpu().numpy(),
                               self.pca[mask, 2].cpu().numpy(), c=color_map[label], label=f'Class {label}')

            plt.legend()
            plt.title('PCA降维后的数据分布')
            plt.savefig(save_file)
            # plt.show()
