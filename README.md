# AINR-Clustering

这是一个基于隐式神经表示（INR）和注意力机制的空间转录组联合聚类模型。

## 1. 项目结构
* **Data/** : 存放数据文件（.h5ad 和真值标签）。
* **model.py** : 核心模型与训练逻辑。
* **network.py** : 深度学习网络架构。
* **run.py** : 项目启动脚本，包含训练与绘图。

## 2. 安装依赖
请确保安装了以下 Python 库：
`pip install torch scanpy anndata pandas numpy matplotlib scikit-learn`

## 3. 运行方法
将数据放入 Data 文件夹后，在终端执行：
`python run.py --training_steps 10000 --n_clusters 7`


---
*注：本项目仅供学术研究使用。*
