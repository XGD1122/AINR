"""
可视化工具：统一的梯度图绘制，生成类似 Fig 11 的空间梯度对比图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from config import VIS_GENES

# 全局绘图风格
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False


def plot_spatial_gradient(adata, gradient_col='Spatial_Gradient',
                          title=None, save_path=None,
                          cmap='turbo', figsize=(6, 6)):
    """
    绘制空间梯度图 (核心对比图 - 对应 Fig 11)

    Args:
        adata: AnnData 对象，obs 中需包含 gradient_col
        gradient_col: 梯度列名
        title: 图标题 (通常为模型名称)
        save_path: 保存路径
        cmap: colormap
        figsize: 图片尺寸
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 获取 library_id
    lib_id = list(adata.uns['spatial'].keys())[0] if 'spatial' in adata.uns else None

    if gradient_col in adata.obs.columns:
        sc.pl.spatial(adata, color=gradient_col, color_map=cmap,
                     ax=ax, show=False, title="",
                     library_id=lib_id)

    if title:
        ax.set_title(title, fontdict={'family': 'Times New Roman', 'size': 14})
    ax.set_axis_off()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        return fig


def plot_gradient_comparison(adata_dict, slice_id, save_path=None,
                             cmap='turbo', figsize=(24, 5)):
    """
    并排对比多个模型的空间梯度 (多模型对比面板)

    Args:
        adata_dict: {model_name: adata} 字典
        slice_id: 切片编号
        save_path: 保存路径
        cmap: colormap
        figsize: 总图尺寸
    """
    n_models = len(adata_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, adata) in zip(axes, adata_dict.items()):
        grad_col = [c for c in adata.obs.columns if 'gradient' in c.lower() or 'isodepth' in c.lower()]
        if grad_col:
            lib_id = list(adata.uns['spatial'].keys())[0] if 'spatial' in adata.uns else None
            sc.pl.spatial(adata, color=grad_col[0], color_map=cmap,
                         ax=ax, show=False, title="", library_id=lib_id)
        ax.set_title(model_name, fontdict={'family': 'Times New Roman', 'size': 14})
        ax.set_axis_off()

    plt.suptitle(f"Spatial Gradient Comparison - Slice {slice_id}",
                 fontsize=16, fontfamily='Times New Roman', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        return fig


def plot_gradient_vs_ground_truth(adata, gradient_col='Spatial_Gradient',
                                   gt_col='Ground Truth', title=None,
                                   save_path=None, figsize=(12, 6)):
    """
    双面板图：左侧梯度，右侧 ground truth (展示梯度与真实层的对应关系)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    lib_id = list(adata.uns['spatial'].keys())[0] if 'spatial' in adata.uns else None

    # 梯度图
    if gradient_col in adata.obs.columns:
        sc.pl.spatial(adata, color=gradient_col, color_map='turbo',
                     ax=axes[0], show=False, title="", library_id=lib_id)
    axes[0].set_title(f"{title} - Spatial Gradient",
                      fontdict={'family': 'Times New Roman', 'size': 13})
    axes[0].set_axis_off()

    # Ground Truth
    if gt_col in adata.obs.columns:
        sc.pl.spatial(adata, color=gt_col, ax=axes[1], show=False,
                     title="", library_id=lib_id,
                     legend_loc='right margin')
    axes[1].set_title("Ground Truth",
                      fontdict={'family': 'Times New Roman', 'size': 13})
    axes[1].set_axis_off()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        return fig


def plot_gene_expression_comparison(adata_dict, genes=None, slice_id=None,
                                     save_dir=None, figsize_per_gene=(18, 6)):
    """
    对比不同模型对特定基因的去噪/重建效果
    """
    if genes is None:
        genes = VIS_GENES

    for gene in genes:
        n_models = len(adata_dict)
        fig, axes = plt.subplots(1, n_models, figsize=figsize_per_gene)
        if n_models == 1:
            axes = [axes]

        for ax, (model_name, adata) in zip(axes, adata_dict.items()):
            if gene in adata.var_names:
                lib_id = list(adata.uns['spatial'].keys())[0] if 'spatial' in adata.uns else None
                sc.pl.spatial(adata, color=gene, color_map='magma',
                             ax=ax, show=False, title="", library_id=lib_id)
            ax.set_title(f"{model_name} - {gene}",
                        fontdict={'family': 'Times New Roman', 'size': 12})
            ax.set_axis_off()

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"gene_{gene}_comparison.pdf"),
                       bbox_inches='tight', dpi=300)
            plt.close(fig)
