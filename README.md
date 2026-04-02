AINR-Clustering: 基于隐式神经表示的空谱反卷积聚类模型AINR-Clustering 是一种专为空间转录组学（Spatial Transcriptomics）设计的深度学习模型。该模型结合了隐式神经表示（Implicit Neural Representation, INR）和特征重建机制，能够有效地融合空间坐标信息与基因表达谱，实现高精度的多样本联合聚类。🌟 主要特性坐标感知：通过 INR 编码精确的空间位置信息。自适应去噪：通过重构机制学习基因表达的潜在分布。高效稳定：采用早停（Early Stopping）机制，仅在训练结束时计算最终 ARI，提升训练效率。多切片支持：支持多个 2D 切片的联合分析。📁 目录结构.
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
🛠️ 环境要求建议使用 Python 3.8+ 以及 CUDA 环境。pip install torch scanpy anndata pandas numpy matplotlib scikit-learn
🚀 快速开始1. 数据准备请确保你的数据存放在同级目录下的 Data 文件夹中。文件命名应符合代码中的默认规范，或通过修改 model.py 中的路径来适配。2. 运行模型使用默认参数启动训练：python run.py --training_steps 10000 --seed 112 --n_clusters 7
3. 可用参数说明参数默认值说明--lr0.001学习率--batch_size512批处理大小--hidden_dim64潜在空间维度--training_steps10000最大训练迭代次数--patience1500早停忍受步数 (基于 Loss)--n_clusters7聚类数目📊 结果输出训练完成后，程序会自动在 results_seed[SEED] 目录下生成：训练日志：实时打印每 100 步的 Loss。最终指标：训练结束时输出所有切片的平均 ARI。可视化图表：每个切片的 Spatial 空间聚类图 和 UMAP 降维图。📝 引用如果你在研究中使用了本项目，请引用相应的论文。本项目仅供学术交流使用。
