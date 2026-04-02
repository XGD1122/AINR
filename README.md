AINR是一个结合隐式神经表示和注意力机制的空间转录组聚类模型。

📁 目录结构说明.
├── core/                   # 核心模型代码
│   ├── model.py            # 训练与评估逻辑
│   └── network.py          # 网络架构定义
│   └── run.py              # 脚本
├── Data/                   # 数据存放目录 (请手动创建)
│   ├── adata_st_DLPFC.h5ad 
│   └── DLPFC_annotations/  # 真值标注
└── README.md              # 项目文档
🛠️ 环境要求项目基于 Python 3.8+ 开发，核心依赖如下：pip install torch scanpy anndata pandas numpy matplotlib scikit-learn
🚀 快速开始1. 数据准备将数据文件放置在 Data/ 目录下。若目录结构不同，请修改 core/model.py 中的路径配置。
2. 运行模型在根目录下直接执行：python run.py --training_steps 10000 --seed 112 --n_clusters 7
