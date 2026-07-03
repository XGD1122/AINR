# AINR: Attention-Guided Implicit Neural Representations for Spatial Domain Identification in Spatial Transcriptomics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![Scanpy](https://img.shields.io/badge/scanpy-1.7+-green.svg)](https://scanpy.readthedocs.io/)

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

---

## 项目文件

| 文件 | 说明 |
|------|------|
| `model.py` | 模型类：数据加载、训练、GMM 聚类、早停、空间梯度 |
| `network.py` | 网络架构：SIREN INR、空间感知 Transformer (SAA)、Decoder |
| `run.py` | 主入口：参数解析、训练、可视化、日志 |
| `demo/demo_3d_alignment.ipynb` | Demo ①：STitch3D ICP 对齐 + 3D 坐标 + 预处理 |
| `demo/AINR_demo.ipynb` | Demo ②：模型训练 + 聚类 + 可视化 |
| `requirements.txt` / `environment.yml` | 依赖管理 |

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
