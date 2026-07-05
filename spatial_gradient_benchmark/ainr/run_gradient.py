"""
AINR 空间梯度实验
基于已有的 DLPFC 实验扩展，专门计算和保存空间梯度用于多模型对比

用法:
    cd spatial_gradient_benchmark/ainr
    python run_gradient.py --slice_idx 151673 151674 151675 151676
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'DLPFC'))

import numpy as np, pandas as pd, torch, random, time, argparse, warnings, copy
warnings.filterwarnings("ignore")

from model import Model
import scanpy as sc
import anndata as ad
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, TensorDataset

# 导入共享工具
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (RESULTS_DIR, AINR_TRAIN_SLICES, DEFAULT_SEED,
                    GRADIENT_BATCH_SIZE, get_n_clusters)
from data_utils import normalize_coords, load_ground_truth
from visualization import plot_spatial_gradient, plot_gradient_vs_ground_truth

# ============================================================
# 参数解析
# ============================================================
parser = argparse.ArgumentParser(description='AINR Spatial Gradient Experiment')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
parser.add_argument('--training_steps', type=int, default=10000)
parser.add_argument('--patience', type=int, default=1500)
parser.add_argument('--inr_width', type=int, default=160)
parser.add_argument('--inr_depth', type=int, default=3)
parser.add_argument('--recon_weight', type=float, default=0.5)
parser.add_argument('--tv_weight', type=float, default=1e-05)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--omega_0', type=float, default=20.0)
parser.add_argument('--slice_idx', type=int, nargs='+', default=AINR_TRAIN_SLICES)
parser.add_argument('--slice_count', type=int, default=4)
args = parser.parse_args()

# ============================================================
# 实验设置
# ============================================================
exp_name = f"AINR_gradient"
save_dir = os.path.join(RESULTS_DIR, exp_name)
os.makedirs(save_dir, exist_ok=True)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 初始化并训练模型
# ============================================================
print("\n=== Initializing AINR Model ===")
model_instance = Model(
    slice_count=args.slice_count, slice_idx_gt=args.slice_idx,
    hidden_dims=[None, args.hidden_dim],
    training_steps=args.training_steps, lr=args.lr, seed=args.seed,
    n_clusters=5, batch_size=args.batch_size, weight_decay=args.weight_decay,
    recon_weight=args.recon_weight, tv_weight=args.tv_weight,
    save_path=save_dir, nhead=args.nhead,
    inr_width=args.inr_width, inr_depth=args.inr_depth,
    dec_width=args.inr_width, dec_depth=args.inr_depth,
    omega_0=args.omega_0
)

print("\n=== Training AINR ===")
start_time = time.time()
final_info = model_instance.train(report_loss=True, eval_interval=100,
                                   patience_steps=args.patience)
runtime = time.time() - start_time
print(f"Training completed in {runtime:.2f}s")

# ============================================================
# 提取空间梯度 (核心对比指标)
# ============================================================
print("\n=== Computing Spatial Gradients ===")
adata_st = final_info['adata_st']

# spatial_gradient 已在 model.train() 中计算
spatial_gradient = final_info['spatial_gradient']
adata_st.obs['Spatial_Gradient'] = spatial_gradient

# 保存梯度值
gradient_df = pd.DataFrame({
    'spot': adata_st.obs_names,
    'gradient_magnitude': spatial_gradient,
    'slice': [n.split('-slice')[0] if '-slice' in str(n) else n for n in adata_st.obs_names]
})
gradient_df.to_csv(os.path.join(save_dir, "spatial_gradient_values.csv"), index=False)

# ============================================================
# 逐切片可视化
# ============================================================
print("\n=== Generating Per-Slice Gradient Visualizations ===")

for i, slice_id in enumerate(args.slice_idx):
    sid = str(slice_id)
    slice_save_dir = os.path.join(save_dir, sid)
    os.makedirs(slice_save_dir, exist_ok=True)

    sec_raw = final_info['adata_st_list_raw'][i].copy()
    sec_raw.var_names_make_unique()

    # 将梯度信息映射到单切片
    sec_raw.obs['Spatial_Gradient'] = adata_st.obs.loc[sec_raw.obs_names, 'Spatial_Gradient'].values

    # 加载 ground truth
    gt = load_ground_truth(sid)
    if gt is not None:
        sec_raw.obs['Ground Truth'] = gt.reindex(sec_raw.obs_names).fillna('nan').astype(str).values

    # 图1: 空间梯度图 (Fig 11 风格 - 核心对比图)
    plot_spatial_gradient(
        sec_raw, gradient_col='Spatial_Gradient',
        title=f'AINR Spatial Gradient (Slice {sid})',
        save_path=os.path.join(slice_save_dir, "gradient_boundary.pdf")
    )

    # 图2: 梯度 vs Ground Truth 并列图
    if 'Ground Truth' in sec_raw.obs.columns:
        plot_gradient_vs_ground_truth(
            sec_raw, gradient_col='Spatial_Gradient',
            gt_col='Ground Truth',
            title='AINR',
            save_path=os.path.join(slice_save_dir, "gradient_vs_gt.pdf")
        )

    # 图3: 梯度 histogram (数值分布)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sec_raw.obs['Spatial_Gradient'].values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Spatial Gradient Magnitude', fontfamily='Times New Roman', fontsize=13)
    ax.set_ylabel('Frequency', fontfamily='Times New Roman', fontsize=13)
    ax.set_title(f'AINR Gradient Distribution - Slice {sid}',
                 fontfamily='Times New Roman', fontsize=14)
    fig.savefig(os.path.join(slice_save_dir, "gradient_histogram.pdf"), bbox_inches='tight')
    plt.close(fig)

    print(f"  Slice {sid}: gradient mean={spatial_gradient[adata_st.obs_names.isin(sec_raw.obs_names)].mean():.4f}, "
          f"std={spatial_gradient[adata_st.obs_names.isin(sec_raw.obs_names)].std():.4f}")

# ============================================================
# 汇总统计
# ============================================================
summary = {
    'Model': 'AINR',
    'Runtime(s)': round(runtime, 2),
    'Gradient_Mean': float(np.mean(spatial_gradient)),
    'Gradient_Std': float(np.std(spatial_gradient)),
    'Gradient_Min': float(np.min(spatial_gradient)),
    'Gradient_Max': float(np.max(spatial_gradient)),
}
pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)

print(f"\n=== AINR Gradient Experiment Complete ===")
print(f"Results saved to: {save_dir}")
print(f"Gradient stats: mean={summary['Gradient_Mean']:.4f}, std={summary['Gradient_Std']:.4f}")
