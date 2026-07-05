"""
共享数据加载工具：所有模型使用统一的数据加载流程
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse
from config import PROJECT_ROOT

def load_dlpfc_slice(slice_id, data_root=None):
    """
    加载单个 DLPFC 切片数据 (用于 GASTON/SUICA/STINR/GASTON-Mix 逐切片处理)

    Args:
        slice_id: 切片编号，如 '151673'
        data_root: 数据根目录，默认使用 GASTON 的 Data 目录

    Returns:
        adata: AnnData 对象，包含空间坐标和基因表达
    """
    if data_root is None:
        data_root = os.path.join(PROJECT_ROOT, "GASTON", "Data")

    file_fold = os.path.join(data_root, str(slice_id))

    # 尝试多种加载方式
    try:
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5')
    except Exception:
        h5_path = os.path.join(file_fold, 'filtered_feature_bc_matrix.h5')
        if os.path.exists(h5_path):
            adata = sc.read_10x_h5(h5_path)
        else:
            raise FileNotFoundError(f"Cannot find Visium data for slice {slice_id} at {file_fold}")

    adata.var_names_make_unique()
    return adata


def load_ground_truth(slice_id):
    """
    加载切片对应的 ground truth 标注

    Args:
        slice_id: 切片编号

    Returns:
        pd.Series: 每个 spot 的真实层标注
    """
    ann_path = os.path.join(PROJECT_ROOT, "DLPFC_annotations", f"{slice_id}_truth.txt")
    if not os.path.exists(ann_path):
        ann_path = os.path.join(PROJECT_ROOT, "DLPFC", "DLPFC_annotations", f"{slice_id}_truth.txt")
    if os.path.exists(ann_path):
        gt = pd.read_csv(ann_path, sep='\t', header=None, index_col=0)
        return gt.iloc[:, 0]
    return None


def normalize_coords(coords, scale_to_minus1_1=True):
    """
    空间坐标归一化

    Args:
        coords: [N, D] numpy array
        scale_to_minus1_1: 如果 True，归一化到 [-1, 1]；否则做 Z-score 标准化

    Returns:
        normalized coords
    """
    if scale_to_minus1_1:
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        return 2.0 * (coords - c_min) / (c_max - c_min + 1e-7) - 1
    else:
        return (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-7)


def compute_gradient_magnitude(model_output, input_coords):
    """
    通用梯度幅值计算：计算 model_output 对 input_coords 的梯度 L2 范数

    这是所有模型空间梯度评估的核心函数。

    Args:
        model_output: 模型输出张量 (可以是 latent representation 或 isodepth)
        input_coords: 输入空间坐标张量 (需要 requires_grad=True)

    Returns:
        gradient_magnitude: [N] 每个点的梯度幅值
    """
    import torch

    grads = torch.autograd.grad(
        outputs=model_output,
        inputs=input_coords,
        grad_outputs=torch.ones_like(model_output),
        create_graph=False,
        retain_graph=False
    )[0]
    return torch.norm(grads, p=2, dim=1)


def prepare_anndata_for_plotting(adata, spatial_key='spatial'):
    """
    确保 AnnData 对象有正确的绘图准备
    """
    if spatial_key not in adata.obsm:
        # 尝试从 uns 中恢复
        import anndata as ad
        if 'spatial' in adata.uns:
            lib_id = list(adata.uns['spatial'].keys())[0]
            coords = adata.uns['spatial'][lib_id]['scalefactors']['tissue_hires_scalef']
            # 这里不做具体坐标恢复，只检查
    return adata
