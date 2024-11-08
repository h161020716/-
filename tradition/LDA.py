import os

import numpy as np
import tqdm
import datetime
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils.load_datasets import load_images_from_folder, load_csv_data

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体（SimHei），用于Windows
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class Lda:
    def __init__(self, n_components):
        self.n_components = n_components
        self.data = None
        self.label = None
        self.lda = None
        self.transformed_data = None
        self.original_shape = None  # 保存原始数据的形状

    def set_params(self, config):
        root = config['Datasets']['datasets_path']
        self.data = load_images_from_folder(root + config['Datasets']['test_data'])
        self.label = load_csv_data(root + config['Datasets']['test_label'])
        self.lda = LDA(n_components=self.n_components)
        self.original_shape = self.data.shape  # 保存原始形状

    def frontend(self, config):
        # 获取样本数和图像的形状信息
        n_samples, h, w, c = self.data.shape
        # 将数据展平为二维数组
        self.data = self.data.reshape(n_samples, h * w * c)

        # 使用标签进行LDA降维
        self.transformed_data = self.lda.fit_transform(self.data, self.label.iloc[:, 1].values)

    def reconstructed(self, config, save_num):
        # 近似重建数据：LDA没有直接的逆变换，因此使用线性组合的方式将降维数据映射回原空间
        if self.transformed_data is None:
            raise ValueError("请先调用 `frontend` 方法进行降维。")

        # 近似重建数据
        approx_reconstructed_data = np.dot(self.transformed_data, self.lda.scalings_.T)
        # 使用原始形状将数据恢复
        n_samples, h, w, c = self.original_shape
        approx_reconstructed_data = approx_reconstructed_data.reshape((n_samples, h, w, c))

        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = os.path.join(config['save_path'], "lda_reconstructed")
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, cur_time + '.png')

        # 绘制和保存重建图像
        plt.figure(figsize=(10, 7))
        for i in tqdm(range(min(save_num, n_samples))):  # 限制图像数量
            img = approx_reconstructed_data[i].reshape(h, w, c)
            img = np.clip(img, 0, 1)  # 将像素值限制在0-1范围内
            plt.subplot(2, 5, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Image {i + 1}')
        plt.suptitle('LDA降维后的图像重建')
        plt.savefig(save_file)
        plt.close()

    def visualize(self, config):
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = os.path.join(config['save_path'], "lda_visualize")
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, cur_time + '.png')

        plt.figure(figsize=(10, 8))

        if self.n_components == 2:
            # 绘制二维图
            ax = plt.subplot()
            for label in set(self.label.iloc[:, 1]):
                mask = (self.label.iloc[:, 1] == label).to_numpy()
                ax.scatter(self.transformed_data[mask, 0], self.transformed_data[mask, 1], label=f'Class {label}')
                for i in range(mask.sum()):
                    plt.text(self.transformed_data[mask][i, 0], self.transformed_data[mask][i, 1], str(label),
                             fontsize=8, color='black', ha='center', va='center')

            plt.legend()
            plt.title('LDA降维后的二维数据分布')

        elif self.n_components == 3:
            # 绘制三维图
            ax = plt.subplot(projection='3d')
            for label in set(self.label.iloc[:, 1]):
                mask = (self.label.iloc[:, 1] == label).to_numpy()
                ax.scatter(self.transformed_data[mask, 0], self.transformed_data[mask, 1],
                           self.transformed_data[mask, 2], label=f'Class {label}')
                for i in range(mask.sum()):
                    ax.text(self.transformed_data[mask][i, 0], self.transformed_data[mask][i, 1],
                            self.transformed_data[mask][i, 2],
                            str(label), fontsize=8, color='black', ha='center')

            ax.set_title('LDA降维后的三维数据分布')
            ax.set_xlabel('X轴')
            ax.set_ylabel('Y轴')
            ax.set_zlabel('Z轴')
            plt.legend()

        plt.savefig(save_file)
        plt.close()
