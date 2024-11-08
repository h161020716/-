import yaml
import datetime
from tradition.PCA import Pca
from tradition.ICA import Ica
from tradition.UMap import UMap
from tradition.KMeans import Kmeans
from tradition.LDA import Lda
import argparse
import os


def load_config(path: str):
    with open(path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="图像数据降维工具")
    parser.add_argument("--PCA", action="store_true", help="选择使用PCA方法进行降维")
    parser.add_argument("--ICA", action="store_true", help="选择使用ICA方法进行降维")
    parser.add_argument("--UMAP", action="store_true", help="选择使用UMAP方法进行降维")
    parser.add_argument("--KMEANS", action="store_true", help="选择使用KMEANS方法进行降维")
    parser.add_argument("--LDA", action="store_true", help="选择使用LDA方法进行降维")
    parser.add_argument("--n_components", type=int, default=2, help="指定PCA的降维维数，默认为2")
    parser.add_argument("--n_clusters", type=int, default=5, help="指定KMeans的聚类簇数，默认为2")
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda"], help="选择运行设备：cpu或cuda")
    parser.add_argument("--config", type=str, default="configs/aptosconfig.yaml", help="指定配置文件路径")
    parser.add_argument("--save_num", type=int, default=10, help="保存前n个降维后的图像")
    parser.add_argument("--visualize", action="store_true", help="是否可视化降维后的数据")
    parser.add_argument("--reconstructed", action="store_true", help="是否保存重建后的图像")

    args = parser.parse_args()

    config = load_config(args.config)
    config['Device'] = args.device
    config['n_components'] = args.n_components
    config['PCA']['n_components'] = args.n_components

    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    if not args.visualize:
        config['PCA']['visualize'] = False
        config['ICA']['visualize'] = False
        config['UMAP']['visualize'] = False
        config['KMEANS']['visualize'] = False
        config['LDA']['visualize'] = False
    if not args.reconstructed:
        config['PCA']['reconstructed'] = False
        config['ICA']['reconstructed'] = False
        config['UMAP']['reconstructed'] = False
        config['KMEANS']['reconstructed'] = False
        config['LDA']['reconstructed'] = False

    if args.PCA:
        config['save_path'] = "results/PCA/" + cur_time
        if not os.path.exists("results/PCA/"):
            os.mkdir("results/PCA/")
        if not os.path.exists(config['save_path']):
            os.mkdir(config['save_path'])
        p = Pca(args.n_components)
        p.set_params(config)
        p.frontend(config)
        if config['PCA']['visualize']:
            p.visualize(config)
        if config['PCA']['reconstructed']:
            p.reconstructed(config, args.save_num)
    elif args.ICA:
        config['save_path'] = "results/ICA/" + cur_time
        if not os.path.exists("results/ICA/"):
            os.mkdir("results/ICA/")
        if not os.path.exists(config['save_path']):
            os.mkdir(config['save_path'])
        i = Ica(args.n_components)
        i.set_params(config)
        i.frontend(config)
        if config['ICA']['visualize']:
            i.visualize(config)
        if config['ICA']['reconstructed']:
            i.reconstructed(config, args.save_num)
    elif args.UMAP:
        config['save_path'] = "results/UMAP/" + cur_time
        if not os.path.exists("results/UMAP/"):
            os.mkdir("results/UMAP/")
        if not os.path.exists(config['save_path']):
            os.mkdir(config['save_path'])
        u = UMap(args.n_components)
        u.set_params(config)
        u.frontend(config)
        if config['UMAP']['visualize']:
            u.visualize(config)
        if config['UMAP']['reconstructed']:
            u.reconstructed(config, args.save_num)
    elif args.KMEANS:
        config['save_path'] = "results/KMEANS/" + cur_time
        if not os.path.exists("results/KMEANS/"):
            os.mkdir("results/KMEANS/")
        if not os.path.exists(config['save_path']):
            os.mkdir(config['save_path'])
        k = Kmeans(int(config['KMEANS']['n_clusters']), args.n_components)
        k.set_params(config)
        k.frontend(config)
        if config['KMEANS']['visualize']:
            k.visualize(config)
        if config['KMEANS']['reconstructed']:
            k.reconstructed(config, args.save_num)
    elif args.LDA:
        config['save_path'] = "results/LDA/" + cur_time
        if not os.path.exists("results/LDA/"):
            os.mkdir("results/LDA/")
        if not os.path.exists(config['save_path']):
            os.mkdir(config['save_path'])
        l = Lda(args.n_components)
        l.set_params(config)
        l.frontend(config)
        if config['LDA']['visualize']:
            l.visualize(config)
        # if config['LDA']['reconstructed']:
        #     l.reconstructed(config, args.save_num)

