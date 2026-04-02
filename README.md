AINR: 基于注意力机制的隐式神经表示的空间域识别工具

AINR 是一种专为空间转录组学设计的深度学习模型。该模型结合了隐式神经表示和注意力机制，实现高精度的聚类。



📁 目录结构

.
├── Data/                   # 数据存放目录 (需手动创建)
│   ├── adata_st_DLPFC.h5ad # 主数据文件
│   ├── adata_st_list_raw0.h5ad ... # 各切片原始数据
│   └── DLPFC_annotations/  # 真值标注文件夹
│       └── 151673_truth.txt ...
├── network.py              # 网络架构定义 (Encoder/Decoder/Transformer)
├── model.py                # 核心训练逻辑与数据加载
├── run.py                  # 命令行启动脚本与可视化
├── README.md               # 项目说明
└── .gitignore              # 忽略非代码文件




🛠️ 环境要求

建议使用 Python 3.8+ 以及 CUDA 环境。

pip install torch scanpy anndata pandas numpy matplotlib scikit-learn



🚀 快速开始

1. 数据准备

请确保你的数据存放在同级目录下的 Data 文件夹中。文件命名应符合代码中的默认规范，或通过修改 model.py 中的路径来适配。

2. 运行模型

使用默认参数启动训练：

python run.py --training_steps 10000 --seed 112 --n_clusters 7



