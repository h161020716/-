import matplotlib.pyplot as plt
import datetime
import os
from sklearn.manifold import TSNE
from utils.load_datasets import load_images_from_folder, load_csv_data

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体字体（SimHei），用于Windows
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class Tsne:
    def __init__(self,n_components: int):
        self.data = None
        self.label = None
        self.tsne_results = None
        self.n_components = n_components

    def set_params(self, config: dict):
        root = config['Datasets']['datasets_path']
        self.data = load_images_from_folder(root + config['Datasets']['test_data'])
        self.label = load_csv_data(root + config['Datasets']['test_label'])

    def frontend(self, config: dict):
        n_samples, h, w, c = self.data.shape
        self.data = self.data.reshape(n_samples, h * w * c)

        # 使用sklearn的t-SNE实现进行降维
        tsne = TSNE(n_components=config['n_components'], random_state=0)
        self.tsne_results = tsne.fit_transform(self.data)

    def visualize(self, config: dict):
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        save_path = config['save_path'] + "/tsne_visualize/"

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, cur_time + '.png')

        plt.figure(figsize=(10, 8))
        color_map = {
            0: 'green',
            1: 'yellow',
            2: 'orange',
            3: 'red',
            4: 'black',
        }

        unique_labels = self.label.iloc[:, 1].unique()  # 假设标签在第二列

        if self.n_components == 2:
            for label in unique_labels:
                mask = (self.label.iloc[:, 1] == label).to_numpy()
                plt.scatter(self.tsne_results[mask, 0], self.tsne_results[mask, 1],
                            c=color_map[label], label=f'Class {label}')
        elif self.n_components == 3:
            ax = plt.figure().add_subplot(111, projection='3d')
            for label in unique_labels:
                mask = (self.label.iloc[:, 1] == label).to_numpy()
                ax.scatter(self.tsne_results[mask, 0], self.tsne_results[mask, 1], self.tsne_results[mask, 2],
                           c=color_map[label], label=f'Class {label}')
        else:
            raise ValueError("n_components must be 2 or 3 for visualization")

        plt.legend()
        plt.title('t-SNE降维后的数据分布')
        plt.savefig(save_file)
        # plt.show()
