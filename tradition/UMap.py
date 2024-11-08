import matplotlib.pyplot as plt
import datetime
import os
import umap
from utils.load_datasets import load_images_from_folder, load_csv_data
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体（SimHei），用于Windows
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class UMap:
    def __init__(self, n: int):
        self.data = None
        self.label = None
        self.n_components = n
        self.umap_model = None
        self.h = None
        self.w = None
        self.c = None
        self.embedding = None

    def set_params(self, config: dict):
        root = config['Datasets']['datasets_path']
        self.data = load_images_from_folder(root + config['Datasets']['test_data'])
        self.label = load_csv_data(root + config['Datasets']['test_label'])

    def frontend(self, config: dict):
        n_samples, self.h, self.w, self.c = self.data.shape

        # Reshape the data for UMAP
        self.data = self.data.reshape(n_samples, self.h * self.w * self.c)

        # Apply UMAP transformation
        self.umap_model = umap.UMAP(n_components=self.n_components)
        self.embedding = self.umap_model.fit_transform(self.data)

    def reconstructed(self, config, num=10):
        save_path = config['save_path'] + "/umap_reconstructed/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 尝试将嵌入数据重新映射回图像格式（非真实重建）
        transformed_data = self.embedding[:num]
        approx_reconstructed_data = transformed_data @ np.random.normal(0, 1,
                                                                        (self.n_components, self.h * self.w * self.c))

        # 归一化数据以便于展示
        approx_reconstructed_data = (approx_reconstructed_data - approx_reconstructed_data.min()) / (
                    approx_reconstructed_data.max() - approx_reconstructed_data.min())

        plt.figure(figsize=(10, 7))
        for i in range(num):
            img_reconstructed = approx_reconstructed_data[i].reshape(self.h, self.w, self.c)
            plt.subplot(2, 5, i + 1)
            plt.imshow(np.clip(img_reconstructed, 0, 1))
            plt.axis('off')
            plt.title(f'Approx Image {i + 1}')

        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        plt.suptitle('UMAP降维后近似重建图像')
        plt.savefig(os.path.join(save_path, cur_time + '_reconstructed.png'))

    def visualize(self, config: dict):
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = config['save_path'] + "/umap_visualize/"

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file = os.path.join(save_path, cur_time + '.png')

        # 2D or 3D visualization of the UMAP-reduced data
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
                # Boolean indexing for each label
                mask = (self.label.iloc[:, 1] == label).to_numpy()
                if self.n_components == 2:
                    ax.scatter(self.embedding[mask, 0], self.embedding[mask, 1],
                               c=color_map[label], label=f'Class {label}')
                elif self.n_components == 3:
                    ax.scatter(self.embedding[mask, 0], self.embedding[mask, 1], self.embedding[mask, 2],
                               c=color_map[label], label=f'Class {label}')

            plt.legend()
            plt.title('UMAP降维后的数据分布')
            plt.savefig(save_file)
            # plt.show()


