# AINR Demo

DLPFC 四切片（151673–151676）联合训练的完整 Demo。

## 数据链路

```
原始 Visium 数据                        STitch3D ICP 对齐               3D 坐标构造
Data/151673/filtered_feature_bc_matrix.h5 ─┐
Data/151674/filtered_feature_bc_matrix.h5 ─┤    align_slices_icp()          aligned_xy + z
Data/151675/filtered_feature_bc_matrix.h5 ─┼──────────────────────────→ ──────────────────→
Data/151676/filtered_feature_bc_matrix.h5 ─┘   (与 对齐脚本.py 一致)       obsm["3D_coor"]

     ↓ preprocess (filter → norm → log1p → HVG → scale)
     ↓
adata_st_list_raw{0,1,2,3}.h5ad  +  adata_st_DLPFC.h5ad
     ↓
  Model.__init__()  →  Model.train()  →  GMM 聚类 + 可视化 + ARI
```

## 快速开始

```bash
# 1. 环境
conda activate ainr

# 2. 预处理（从原始 Visium 数据 → .h5ad）
python demo/prepare_demo_data.py

# 3. 运行 Demo（两种方式等价）
# 方式 A: Jupyter Notebook（逐步展示）
jupyter notebook demo/AINR_demo.ipynb

# 方式 B: 命令行（一键运行）
python run.py --slice_idx 151673 151674 151675 151676 --slice_count 4 --n_clusters 7
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `demo_3d_alignment.ipynb` | **Demo 脚本 ①**：ICP 对齐 + 3D 坐标构造，带对齐前后对比图和 3D 可视化 |
| `AINR_demo.ipynb` | **Demo 脚本 ②**：模型训练 + 聚类 + 可视化，直接调用 `model.py` 的 `Model` 类 |
| `README.md` | 本文件 |

## 3D 坐标构造逻辑

`prepare_demo_data.py` 中的 ICP 对齐和 3D 坐标构造与 STitch3D 的 `对齐脚本.py` 一致：

1. **ICP 对齐**：提取各切片边缘 spots，用迭代最近点（ICP）算法将切片 1/2/3 依次对齐到切片 0
2. **z 轴构造**：对齐后的 2D 坐标 + 切片物理深度（默认相邻 10μm）缩放为 z 坐标
3. **归一化**：最终 3D 坐标归一化到 [-1, 1]

详见 `prepare_demo_data.py` 中 `align_slices_icp()` 和 `prepare_data()` 函数的注释。

## 硬件 & 耗时

| 步骤 | GPU | CPU |
|------|-----|-----|
| 预处理（含 ICP 对齐） | ~3 min | ~5 min |
| 训练 10000 步 | ~20-30 min | ~3-5 hrs |
| 可视化 | ~5 min | ~10 min |
| **总计** | **~35 min** | **~4-6 hrs** |

## 输出

```
demo_output/
├── best_model.pth
├── loss_curves.pdf / loss_history.csv
├── 151673/  (pred_spatial, truth_spatial, UMAP, 6基因×2, 小提琴图×2, 梯度边界)
├── 151674/
├── 151675/
└── 151676/
```

## 调参建议

| 参数 | 作用 | 推荐范围 |
|------|------|---------|
| `tv_weight` | 空间平滑强度 | 1e-6 (异构组织) ~ 1e-4 (均质组织) |
| `recon_weight` | 去噪 vs 表征质量 | 0.1 ~ 2.0 |
| `omega_0` | SIREN 频率 | 10 ~ 30 |
