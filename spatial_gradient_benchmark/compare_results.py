"""
多模型梯度结果对比脚本
汇总所有模型的梯度统计，生成统一的对比图和表格 (对应 Fig 11 风格)

用法:
    cd spatial_gradient_benchmark
    python compare_results.py --slices 151673 151674 151675 151676
"""
import os, sys, argparse, glob, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr
import scanpy as sc
import anndata as ad

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (RESULTS_DIR, ALL_SLICES, VIS_GENES, get_n_clusters,
                    GASTON_DATA_DIR, PROJECT_ROOT)
from data_utils import load_dlpfc_slice, load_ground_truth
from visualization import plot_gradient_comparison

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description='Compare spatial gradients across models')
parser.add_argument('--slices', type=str, nargs='+',
                    default=['151673', '151674', '151675', '151676'])
parser.add_argument('--models', type=str, nargs='+',
                    default=['AINR', 'GASTON', 'SUICA', 'STINR', 'GASTON-Mix'])
args = parser.parse_args()

MODEL_NAMES = args.models
MODEL_DIRS = {
    'AINR': 'AINR_gradient',
    'GASTON': 'GASTON_gradient',
    'SUICA': 'SUICA_gradient',
    'STINR': 'STINR_gradient',
    'GASTON-Mix': 'GASTONMix_gradient',
}

compare_dir = os.path.join(RESULTS_DIR, "comparison")
os.makedirs(compare_dir, exist_ok=True)

# ============================================================
# 1. 收集所有模型的梯度数据
# ============================================================
print("=== Collecting Gradient Data Across All Models ===\n")

all_data = {}  # {slice_id: {model_name: {'gradient': array, 'summary': dict}}}

for slice_id in args.slices:
    all_data[slice_id] = {}
    for model_name in MODEL_NAMES:
        model_result_dir = os.path.join(RESULTS_DIR, MODEL_DIRS[model_name], slice_id)

        # AINR 特殊处理 (多切片联合)
        if model_name == 'AINR':
            csv_path = os.path.join(RESULTS_DIR, MODEL_DIRS[model_name],
                                    "spatial_gradient_values.csv")
        else:
            csv_path = os.path.join(model_result_dir, "spatial_gradient_values.csv")

        summary_path = os.path.join(model_result_dir, "summary.csv") if model_name != 'AINR' \
            else os.path.join(RESULTS_DIR, MODEL_DIRS[model_name], "summary.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'gradient_magnitude' in df.columns:
                grad_vals = df['gradient_magnitude'].dropna().values
                all_data[slice_id][model_name] = {
                    'gradient': grad_vals,
                    'df': df,
                }
                print(f"  [{slice_id}] {model_name}: {len(grad_vals)} spots loaded")
            else:
                print(f"  [{slice_id}] {model_name}: gradient_magnitude column not found")
        else:
            print(f"  [{slice_id}] {model_name}: No data found at {csv_path}")

# ============================================================
# 2. 汇总统计表格
# ============================================================
print("\n=== Generating Summary Statistics ===")

summary_rows = []
for slice_id in args.slices:
    for model_name in MODEL_NAMES:
        if model_name in all_data.get(slice_id, {}):
            g = all_data[slice_id][model_name]['gradient']
            summary_rows.append({
                'Slice': slice_id,
                'Model': model_name,
                'Mean': round(np.mean(g), 4),
                'Std': round(np.std(g), 4),
                'Min': round(np.min(g), 4),
                'Max': round(np.max(g), 4),
                'Median': round(np.median(g), 4),
                'CV': round(np.std(g) / (np.mean(g) + 1e-8), 4),
            })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(compare_dir, "gradient_summary_stats.csv"), index=False)

# Pivot 表格用于论文
pivot_mean = df_summary.pivot(index='Slice', columns='Model', values='Mean')
pivot_std = df_summary.pivot(index='Slice', columns='Model', values='Std')
pivot_cv = df_summary.pivot(index='Slice', columns='Model', values='CV')

print("\n--- Gradient Mean (Lower = smoother boundaries) ---")
print(pivot_mean.to_string())
print("\n--- Gradient Std ---")
print(pivot_std.to_string())
print("\n--- Coefficient of Variation ---")
print(pivot_cv.to_string())

# ============================================================
# 3. 梯度对比图 (每切片多模型并排 - Fig 11 风格)
# ============================================================
print("\n=== Generating Comparison Figures ===")

for slice_id in args.slices:
    print(f"\n  Generating comparison for Slice {slice_id}...")

    # 加载原始数据用于空间可视化
    try:
        adata_raw = load_dlpfc_slice(slice_id, data_root=GASTON_DATA_DIR)
        adata_raw.var_names_make_unique()
    except Exception:
        adata_raw = None
        print(f"    Warning: Cannot load raw data for slice {slice_id}")

    # 3a. 梯度分布对比 (小提琴图/箱线图)
    fig, ax = plt.subplots(figsize=(12, 6))
    data_for_plot = []
    labels_for_plot = []
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']

    for i, model_name in enumerate(MODEL_NAMES):
        if model_name in all_data.get(slice_id, {}):
            g = all_data[slice_id][model_name]['gradient']
            data_for_plot.append(g)
            labels_for_plot.append(model_name)

    bp = ax.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True,
                    widths=0.6, showfliers=False)
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors[:len(labels_for_plot)])):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    ax.set_ylabel('Spatial Gradient Magnitude', fontfamily='Times New Roman', fontsize=13)
    ax.set_title(f'Gradient Distribution Comparison - Slice {slice_id}',
                 fontfamily='Times New Roman', fontsize=14)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, f"gradient_boxplot_{slice_id}.pdf"),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # 3b. 梯度直方图叠加
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(MODEL_NAMES):
        if model_name in all_data.get(slice_id, {}):
            g = all_data[slice_id][model_name]['gradient']
            ax.hist(g, bins=40, alpha=0.4, label=model_name,
                   color=colors[i % len(colors)], density=True)
    ax.set_xlabel('Spatial Gradient Magnitude', fontfamily='Times New Roman', fontsize=13)
    ax.set_ylabel('Density', fontfamily='Times New Roman', fontsize=13)
    ax.set_title(f'Gradient Density Overlay - Slice {slice_id}',
                 fontfamily='Times New Roman', fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, f"gradient_density_{slice_id}.pdf"),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # 3c. 梯度对比热图 (模型 vs 统计量)
    fig, ax = plt.subplots(figsize=(10, 3))
    stats_matrix = []
    stat_names = ['Mean', 'Std', 'Median', 'CV']
    for model_name in MODEL_NAMES:
        if model_name in all_data.get(slice_id, {}):
            g = all_data[slice_id][model_name]['gradient']
            stats_matrix.append([
                np.mean(g), np.std(g), np.median(g),
                np.std(g) / (np.mean(g) + 1e-8)
            ])

    if stats_matrix:
        im = ax.imshow(np.array(stats_matrix).T, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(MODEL_NAMES)))
        ax.set_xticklabels(MODEL_NAMES, rotation=30, fontsize=11)
        ax.set_yticks(range(len(stat_names)))
        ax.set_yticklabels(stat_names, fontsize=12)
        for i in range(len(stat_names)):
            for j in range(len(MODEL_NAMES)):
                ax.text(j, i, f'{np.array(stats_matrix).T[i, j]:.3f}',
                       ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Gradient Statistics Heatmap - Slice {slice_id}',
                     fontfamily='Times New Roman', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(compare_dir, f"gradient_heatmap_{slice_id}.pdf"),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)

# ============================================================
# 4. 跨模型相关性分析
# ============================================================
print("\n=== Cross-Model Gradient Correlation ===")

for slice_id in args.slices:
    model_names_present = [m for m in MODEL_NAMES if m in all_data.get(slice_id, {})]
    if len(model_names_present) < 2:
        continue

    n = len(model_names_present)
    corr_matrix = np.zeros((n, n))
    for i, m1 in enumerate(model_names_present):
        for j, m2 in enumerate(model_names_present):
            g1 = all_data[slice_id][m1]['gradient']
            g2 = all_data[slice_id][m2]['gradient']
            # 使用等长采样
            min_len = min(len(g1), len(g2))
            corr_matrix[i, j], _ = pearsonr(g1[:min_len], g2[:min_len])

    corr_df = pd.DataFrame(corr_matrix, index=model_names_present, columns=model_names_present)
    corr_df.to_csv(os.path.join(compare_dir, f"gradient_correlation_{slice_id}.csv"))

    # 相关性热图
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_xticklabels(model_names_present, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_names_present, fontsize=11)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                   ha='center', va='center', fontsize=10,
                   color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f'Gradient Correlation Matrix - Slice {slice_id}',
                 fontfamily='Times New Roman', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(compare_dir, f"gradient_correlation_{slice_id}.pdf"),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

# ============================================================
# 5. 终极汇总：所有切片所有模型的整体对比
# ============================================================
print("\n=== Global Summary ===")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 5a. 所有切片梯度均值条图
ax = axes[0]
x = np.arange(len(args.slices))
width = 0.15
for i, model_name in enumerate(MODEL_NAMES):
    means = []
    for slice_id in args.slices:
        if model_name in all_data.get(slice_id, {}):
            means.append(np.mean(all_data[slice_id][model_name]['gradient']))
        else:
            means.append(0)
    ax.bar(x + i * width, means, width, label=model_name,
           color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336'][i],
           alpha=0.8)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(args.slices, fontsize=11)
ax.set_ylabel('Mean Gradient Magnitude', fontfamily='Times New Roman', fontsize=13)
ax.set_title('Spatial Gradient Comparison Across Models and Slices',
             fontfamily='Times New Roman', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 5b. 雷达图 (多模型梯度特征)
ax = axes[1]
# 汇总所有切片
global_stats = {}
for model_name in MODEL_NAMES:
    all_grads = []
    for slice_id in args.slices:
        if model_name in all_data.get(slice_id, {}):
            all_grads.extend(all_data[slice_id][model_name]['gradient'].tolist())
    if all_grads:
        all_grads = np.array(all_grads)
        global_stats[model_name] = {
            'Mean': np.mean(all_grads),
            'Std': np.std(all_grads),
            'CV': np.std(all_grads) / (np.mean(all_grads) + 1e-8),
            'Range': np.max(all_grads) - np.min(all_grads),
        }

df_global = pd.DataFrame(global_stats).T
df_global.to_csv(os.path.join(compare_dir, "global_gradient_summary.csv"))

# 条图展示全局统计
metrics = ['Mean', 'Std', 'CV']
x_glob = np.arange(len(metrics))
width_glob = 0.15
for i, model_name in enumerate(MODEL_NAMES):
    if model_name in df_global.index:
        vals = [df_global.loc[model_name, m] for m in metrics]
        ax.bar(x_glob + i * width_glob, vals, width_glob, label=model_name,
               color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336'][i],
               alpha=0.8)

ax.set_xticks(x_glob + width_glob * 2)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel('Value', fontfamily='Times New Roman', fontsize=13)
ax.set_title('Global Gradient Statistics (All Slices Pooled)',
             fontfamily='Times New Roman', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(compare_dir, "global_comparison.pdf"), bbox_inches='tight', dpi=300)
plt.close(fig)

print(f"\n=== Comparison Complete ===")
print(f"All comparison results saved to: {compare_dir}")
print(f"Files generated:")
for f in sorted(os.listdir(compare_dir)):
    print(f"  - {f}")
