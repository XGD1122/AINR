# AINR: Attention-Guided Implicit Neural Representations for Spatial Domain Identification in Spatial Transcriptomics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![Scanpy](https://img.shields.io/badge/scanpy-1.7+-green.svg)](https://scanpy.readthedocs.io/)

AINR 基于 SIREN 隐式神经表征和空间感知注意力机制，用于空间转录组学中的**空间域聚类**、**基因表达去噪**和**组织边界检测**。

---

## 目录

- [数据流程](#数据流程)
- [硬件要求](#硬件要求)
- [Demo（两个独立脚本）](#demo两个独立脚本)
  - [Demo ①：3D 数据对齐与坐标构造](#demo-3d-数据对齐与坐标构造)
  - [Demo ②：AINR 模型训练与可视化](#demo-ainr-模型训练与可视化)
  - [拷贝到 GPU 电脑的文件清单](#拷贝到-gpu-电脑的文件清单)
- [安装](#安装)
- [完整实验流程](#完整实验流程)
- [项目文件说明](#项目文件说明)
- [引用](#引用)

---

## 数据流程

```
原始 Visium 数据                     STitch3D                    AINR
Data/{sid}/filtered_feature_    →   ICP 对齐      →    SIREN INR
     bc_matrix.h5                   3D 坐标构造         空间感知 Attention
tissue_positions_list.csv           filter/norm/        Decoder 去噪
                                    log1p/HVG/scale     GMM 聚类
                                         ↓                 ↓
                              adata_st_list_raw{i}.h5ad  空间域图
                              adata_st_DLPFC.h5ad        UMAP / 梯度边界
                              (Demo ① 产出)              ARI 评估
                                                         (Demo ② 产出)
```

---

## 硬件要求

| 组件 | Demo ①（数据对齐） | Demo ②（模型训练） |
|------|-------------------|-------------------|
| CPU | 4 核 | 8+ 核 |
| RAM | 16 GB | 32+ GB |
| GPU | 不需要 | NVIDIA GPU 8GB+ VRAM 推荐 |
| 磁盘 | 5 GB | 10+ GB |
| 耗时 | ~3-5 min | GPU ~30 min / CPU ~4-6 hrs |

---

## Demo（两个独立脚本）

两个 Demo 各自独立，可分别上传和运行。

### Demo ①：3D 数据对齐与坐标构造

```bash
conda activate stitch3d
jupyter notebook demo/demo_3d_alignment.ipynb
```

**内容**：`sc.read_visium()` → `STitch3D.utils.align_spots()` → 3D 坐标构造 → 预处理 → `.h5ad`

**输出**：

| 文件 | 说明 |
|------|------|
| `demo/demo_output/3d_coordinate_visualization.pdf` | 3D 坐标可视化 |
| `demo/demo_output/alignment_before_after.pdf` | ICP 对齐前后对比 |
| `adata_st_list_raw{0,1,2,3}.h5ad` | 各切片 log-normalized 数据 |
| `adata_st_DLPFC.h5ad` | 合并预处理数据（含 `obsm["3D_coor"]`） |

---

### Demo ②：AINR 模型训练与可视化

```bash
conda activate stinr
jupyter notebook demo/AINR_demo.ipynb
# Cell → Run All
```

**内容**：加载 `.h5ad` → `Model.__init__()` → `Model.train()` → GMM 聚类 → 全部可视化 + ARI

**输出**：

```
demo_output/
├── best_model.pth / loss_curves.pdf / loss_history.csv
├── 151673/
│   ├── pred_spatial.pdf / truth_spatial.pdf
│   ├── pred_umap.pdf / truth_umap.pdf
│   ├── spatial_gradient_boundaries.pdf
│   ├── raw_spatial_<gene>.pdf / denoised_spatial_<gene>.pdf  (×6 genes)
│   └── stacked_violin_raw_*.pdf / stacked_violin_AINR_*.pdf
├── 151674/ ...
├── 151675/ ...
└── 151676/ ...
```

> 等价命令行：`python run.py --slice_idx 151673 151674 151675 151676 --slice_count 4 --n_clusters 7`

---

### 拷贝到 GPU 电脑的文件清单

在 GPU 电脑上创建一个项目目录，把以下文件拷过去：

```
DLPFC/                          ← 项目根目录
├── model.py                    ← AINR 模型类
├── network.py                  ← 网络架构（SIREN + SAA + Decoder）
├── run.py                      ← 命令行入口
├── requirements.txt            ← pip 依赖列表
├── environment.yml             ← conda 环境配置
├── README.md                   ← 项目说明
├── adata_st_list_raw0.h5ad     ← Demo ① 产出：切片 0 数据 (~64 MB)
├── adata_st_list_raw1.h5ad     ← Demo ① 产出：切片 1 数据 (~78 MB)
├── adata_st_list_raw2.h5ad     ← Demo ① 产出：切片 2 数据 (~52 MB)
├── adata_st_list_raw3.h5ad     ← Demo ① 产出：切片 3 数据 (~54 MB)
├── adata_st_DLPFC.h5ad         ← Demo ① 产出：合并预处理数据 (~167 MB)
├── DLPFC_annotations/          ← Ground Truth 标注（用于 ARI 评估）
│   ├── 151673_truth.txt
│   ├── 151674_truth.txt
│   ├── 151675_truth.txt
│   └── 151676_truth.txt
└── demo/
    ├── AINR_demo.ipynb         ← Demo ② Notebook
    └── demo_output/            ← 输出目录（运行后生成）
```

> `.h5ad` 文件共约 415 MB。`.gitignore` 中已排除它们不推送到 GitHub。

---

## 安装

### Demo ① 环境（stitch3d，含 STitch3D 包）

```bash
conda activate stitch3d
# STitch3D 包已预装，无需额外安装
```

### Demo ② 环境（stinr，含 PyTorch）

```bash
# 方式一：Conda
conda env create -f environment.yml
conda activate stinr

# 方式二：Pip
pip install -r requirements.txt

# 验证
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 完整实验流程

### 1. 运行主实验

```bash
# 默认参数（4 切片，7 聚类，TV=1e-5）
python run.py

# 自定义参数
python run.py \
  --slice_idx 151673 151674 151675 151676 \
  --slice_count 4 --n_clusters 7 \
  --inr_width 160 --inr_depth 3 \
  --tv_weight 1e-5 --recon_weight 0.5 --omega_0 20.0 \
  --hidden_dim 32 --batch_size 2048 --lr 0.001 \
  --training_steps 10000 --patience 1500
```

输出保存在 `results_sweep/<exp_tag>/`。

### 2. 消融实验：INR+PCA 基线

```bash
cd INR+PCA
python experiment_inr_pca.py \
  --slice_idx 151669 151670 151671 151672 \
  --n_clusters 5 --pca_dims 50 --inr_width 160
```

### 3. 超参数扫描

| 参数 | 扫描范围 |
|------|---------|
| `tv_weight` | 1e-5, 2e-5, 5e-5 |
| `recon_weight` | 0.1, 0.5, 1.0, 2.0 |
| `omega_0` | 10.0, 20.0, 30.0 |
| `inr_width` | 80, 160, 256 |

每组合运行一次 `run.py`，结果自动追加到 `summary_report.csv`。

---

## 项目文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `model.py` | AINR 模型类：数据加载、训练循环、GMM 聚类、早停、空间梯度 |
| `network.py` | 神经网络架构：SIREN INR、空间感知 Transformer (SAA)、Decoder |
| `run.py` | 主实验入口：参数解析、训练、可视化生成、日志 |
| `demo/demo_3d_alignment.ipynb` | **Demo ①**：STitch3D ICP 对齐 + 3D 坐标 + 预处理 |
| `demo/AINR_demo.ipynb` | **Demo ②**：Model 训练 + GMM 聚类 + 全部可视化 |
| `demo/README.md` | Demo 详细使用说明 |
| `INR+PCA/experiment_inr_pca.py` | 消融实验：INR-only + PCA + GMM |
| `requirements.txt` | Pip 依赖 |
| `environment.yml` | Conda 环境 |
| `Data/` | 原始 10X Visium 数据（需自行下载） |
| `DLPFC_annotations/` | Ground Truth 层标注 |
| `results_sweep/` | 实验输出目录 |
| `ari.txt` | 消融实验 ARI 记录 |

---

## 引用

```bibtex
@article{ainr2025,
  title     = {AINR: Attention-Guided Implicit Neural Representations
               for Spatial Domain Identification in Spatial Transcriptomics},
  author    = {Li, Ponian and ...},
  journal   = {Bioinformatics Advances},
  year      = {2025},
  note      = {Under review}
}
```

## License

MIT License.
