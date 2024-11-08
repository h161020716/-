# 数据挖掘实验一
​	基于本学期课程数据挖掘的实验需求，所以创建了仓库用作随笔，欢迎大家批评指正

​	**建议** ：在开始实验前请查看***实验指导书***了解本实验的主要内容

## 实验内容
​       参考实验指导书
## 实现方案

​	总共实现了六种数据降维的方式，分别是：pca，lca，umap，lda，k-means，t-sne。从结果来看，效果最好的是：LDA ，这么好😥是因为引入了标签数据。其他的都分不出来.......（还得是深度学习🐶）

### 1.PCA

​	pca没有用sklearn里面的，太慢了，自己写了个使用GPU加速的，凑合用

### 2.LCA

​	与PCA不同，PCA侧重于方差最大化，而ICA侧重于统计独立性。

### 3.UMap

​	UMAP的核心思想是通过构造高维数据的邻接图，然后映射到低维空间，保持数据的局部结构。

### 4.LDA

​	LDA是一种监督学习的降维方法，其主要目标是找到一个能最小化类内方差、最大化类间方差的投影空间。

### 5.K-Means

​	K-Means是一种常见的无监督学习聚类算法，通过将数据划分为K个簇，使得簇内的数据点尽可能相似，簇间的数据点尽可能不同。

### 6.T-Sne

​	T-SNE在高维数据可视化中效果显著，但计算量大，适合小规模数据集。

## 快速开始

### 数据集下载

​	运行shell脚本获取数据集 

```shell
scripts/apots2019.sh 
```

### 环境配置

​	提供的默认安装方法基于 Conda 包和环境管理：

```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

### 命令解析

​	运行以下命令快速开始

```python
python main.py --PCA --n_components 2 --visualize
```

通过更换参数来选择不同方法

- --PCA
- --ICA
- --UMAP
- --LDA
- --KMEANS
- --T_SNE

其余参数
* --n_components	指定降低到几维（2/3）
* --n_clusters                k-means专用，用来指定有几个聚类
* --device                       pca专用，用来指定GPU设备
* --config                       加载配置文件，默认为configs下的atosconfig.yaml
* --save_num               与 `--reconstructed` 参数共同使用，用于指明从降维后的数据中重建回原始图片
* --visualize                  对于数据执行可视化操作
* --reconstructed        对于降维后的数据进行重建