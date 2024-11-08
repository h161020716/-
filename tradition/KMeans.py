import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from utils.load_datasets import load_images_from_folder, load_csv_data

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体（SimHei），用于Windows
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class Kmeans:
    def __init__(self, n_clusters: int, n_components: int):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.data = None
        self.labels = None
        self.centroids = None
        self.original_shape = None
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)

    def set_params(self, config):
        root = config['Datasets']['datasets_path']
        self.data = load_images_from_folder(root + config['Datasets']['test_data'])
        self.label = load_csv_data(root + config['Datasets']['test_label'])
        self.original_shape = self.data.shape

    def frontend(self, config):
        n_samples, h, w, c = self.data.shape
        self.data = self.data.reshape(n_samples, h * w * c)
        self.data = minmax_scale(self.data)
        self.kmeans.fit(self.data)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_

    def reconstructed(self, config, num: int):
        n_samples, h, w, c = self.original_shape
        # 使用聚类中心作为“重建”后的数据
        reconstructed_data = self.centroids[self.labels]

        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = os.path.join(config['save_path'], "kmeans_reconstructed")
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, cur_time + '.png')

        plt.figure(figsize=(10, 7))
        for i in tqdm(range(num)):
            img_kmeans = reconstructed_data[i].reshape(h, w, c)
            img_kmeans = np.clip(img_kmeans, 0, 1)  # 确保像素值在0到1之间

            plt.subplot(2, 5, i + 1)
            plt.imshow(img_kmeans)
            plt.axis('off')
            plt.title(f'Image {i + 1}')
        plt.suptitle('K-Means聚类后的图像重建')
        plt.savefig(save_file)

    def visualize(self, config):
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = os.path.join(config['save_path'], "kmeans_visualize")
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, cur_time + '.png')

        plt.figure(figsize=(10, 8))

        if self.n_components == 2:
            # 绘制二维图
            scatter = plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis')
            for i, label in enumerate(self.labels):
                plt.text(self.data[i, 0], self.data[i, 1], str(label), fontsize=8, color='black', ha='center',
                         va='center')
            plt.colorbar(scatter)
            plt.title('K-Means 2D聚类结果')

        elif self.n_components == 3:
            # 绘制三维图
            ax = plt.subplot(projection='3d')
            scatter = ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=self.labels, cmap='viridis')
            for i, label in enumerate(self.labels):
                ax.text(self.data[i, 0], self.data[i, 1], self.data[i, 2], str(label), fontsize=8, color='black',
                        ha='center')
            plt.colorbar(scatter)
            ax.set_title('K-Means 3D聚类结果')
            ax.set_xlabel('X轴')
            ax.set_ylabel('Y轴')
            ax.set_zlabel('Z轴')

        plt.savefig(save_file)
        plt.close()


