# AINR: Attention-Guided Implicit Neural Representations for Spatial Domain Identification in Spatial Transcriptomics

AINR 基于 SIREN 隐式神经表征和空间感知注意力机制，用于空间转录组学中的空间域聚类、基因表达去噪和组织边界检测。

---

## Demo

两个独立的 Jupyter Notebook，展示完整工作流：

| Demo | 文件 | 内容 |
|------|------|------|
| ① 数据对齐 | `demo/demo_3d_alignment.ipynb` | ICP 切片对齐 → 3D 坐标构造 → 预处理 |
| ② 模型训练 | `demo/AINR_demo.ipynb` | AINR 训练 → GMM 聚类 → 空间域图 / 梯度边界 / ARI |

> 预处理后的演示数据（.h5ad）可从 Google Drive 下载，跳过 Demo ① 直接运行 Demo ②：
> https://drive.google.com/drive/folders/1EltGoufcGY4n-PCZtwN8SPJ8J_OaUUUj

---

## 安装

```bash
# Demo ① 环境（含 STitch3D）
conda activate stitch3d

# Demo ② 环境（含 PyTorch）
conda env create -f environment.yml
conda activate stinr
# 或: pip install -r requirements.txt
```

---

## 运行

```bash
# Demo ①
jupyter notebook demo/demo_3d_alignment.ipynb

# Demo ②
jupyter notebook demo/AINR_demo.ipynb

# 或命令行
python run.py --slice_idx 151673 151674 151675 151676 --slice_count 4 --n_clusters 7
```
