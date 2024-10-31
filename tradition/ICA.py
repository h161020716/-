import torch
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from utils.load_datasets import load_images_from_folder, load_csv_data


class Ica:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.data = None
        self.label = None
        self.data_mean = None
        self.W = None
        self.S = None  # 独立成分

    def set_params(self, config: dict):
        root = config['Datasets']['datasets_path']
        self.data = load_images_from_folder(root + config['Datasets']['test_data'])
        self.label = load_csv_data(root + config['Datasets']['test_label'])

    def orthogonalize(self, W):
        # 正交化步骤
        for i in range(W.shape[0]):
            for j in range(i):
                W[i] -= torch.dot(W[i], W[j]) * W[j]
        return W / torch.norm(W, dim=1, keepdim=True)

    def frontend(self, config: dict):
        n_samples, h, w, c = self.data.shape
        self.data = torch.tensor(self.data.reshape(n_samples, -1), dtype=torch.float32).to(config['Device'])

        # 数据中心化
        self.data_mean = torch.mean(self.data, dim=0)
        data_centered = self.data - self.data_mean

        # 白化
        cov = torch.cov(data_centered.T)  # 协方差矩阵
        eigenvalues, eigenvectors = torch.eig(cov, eigenvectors=True)  # 特征值和特征向量
        D = torch.diag(1.0 / torch.sqrt(eigenvalues[:, 0]))  # 白化矩阵
        self.W = torch.matmul(eigenvectors, D)  # 白化后的数据

        # ICA 算法
        self.W = torch.randn(self.n_components, self.n_components).to(data_centered.device)

        # 迭代计算
        for _ in range(1000):
            # 计算信号
            Y = torch.matmul(self.W, data_centered.T)
            Y = Y.T  # 转置为样本为行的形式

            # 通过非线性函数求导计算
            g = torch.tanh(Y)  # 使用双曲正切函数
            g_prime = 1 - g ** 2  # 导数

            # 更新权重
            self.W = self.W + (1 / data_centered.shape[0]) * torch.matmul(g.T, data_centered) - (
                    torch.mean(g_prime, dim=0)[:, None] * self.W)

            # 正交化
            self.W = self.orthogonalize(self.W)

        # 计算独立成分
        self.S = torch.matmul(self.W, data_centered.T).T

    def reconstructed(self, config: dict, num: int):
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = config['save_path'] + "/reconstructed/"

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file = os.path.join(save_path, cur_time + '.png')

        plt.figure(figsize=(10, 7))
        for i in range(num):  # 显示前10个独立成分
            img_ica = self.S[i].reshape(*self.data.shape[1:]) + self.data_mean.cpu().numpy()  # 重建图像
            img_ica = np.clip(img_ica, 0, 255).astype(np.uint8)  # 限制范围并转换类型

            # 可视化
            plt.subplot(2, 5, i + 1)
            plt.imshow(img_ica)
            plt.axis('off')
            plt.title(f'Independent Component {i + 1}')
        plt.suptitle('ICA 降维后的独立成分')
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
                mask = (self.label.iloc[:, 1] == label).to_numpy()
                if self.n_components == 2:
                    ax.scatter(self.ica[mask, 0].cpu().numpy(), self.ica[mask, 1].cpu().numpy(),
                               c=color_map[label], label=f'Class {label}')
                elif self.n_components == 3:
                    ax.scatter(self.ica[mask, 0].cpu().numpy(), self.ica[mask, 1].cpu().numpy(),
                               self.ica[mask, 2].cpu().numpy(), c=color_map[label], label=f'Class {label}')

            plt.legend()
            plt.title('ICA降维后的数据分布')
            plt.savefig(save_file)
            # plt.show()
