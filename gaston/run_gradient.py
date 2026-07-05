"""
GASTON 空间梯度实验
直接使用原始 GASTON 仓库 (models/gaston/src) 的代码和参数。

官方参数:
  - f_S: R^2 -> R (isodepth), hidden=[20,20], activation=ReLU
  - f_A: R -> R^G (expression), hidden=[20,20], activation=ReLU
  - Positional encoding: embed_size=4, sigma=0.2
  - 优化器: Adam, lr=0.001
  - 训练: 10000 epochs
  - 输入 S: Z-score 标准化

空间梯度 = d(isodepth)/d(spatial_coords)

用法:
    conda run -n AINR python run_gradient.py --slice 151673
"""
import os, sys, time, argparse, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
import scanpy as sc, anndata
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 导入共享工具
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (RESULTS_DIR, DEFAULT_SEED, get_n_clusters,
                    GASTON_DATA_DIR, PROJECT_ROOT)
from data_utils import load_dlpfc_slice, load_ground_truth
from visualization import (plot_spatial_gradient, plot_gradient_vs_ground_truth)

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description='GASTON Spatial Gradient Experiment')
parser.add_argument('--slice', type=str, default='151673')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10000)  # 官方默认: 10000
parser.add_argument('--n_comps', type=int, default=15)
parser.add_argument('--n_top_genes', type=int, default=2000)
parser.add_argument('--s_hidden', type=str, default='20,20')
parser.add_argument('--a_hidden', type=str, default='20,20')
parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
args = parser.parse_args()

exp_name = f"GASTON_gradient"
save_dir = os.path.join(RESULTS_DIR, exp_name, args.slice)
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"GASTON settings: S_hidden={args.s_hidden}, A_hidden={args.a_hidden}, "
      f"epochs={args.epochs}, n_comps={args.n_comps}, HVG={args.n_top_genes}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ============================================================
# 1. 加载 GASTON 原始代码
# ============================================================
gaston_src = os.path.join(os.path.dirname(__file__), '..', 'models', 'gaston', 'src')
if os.path.isdir(gaston_src):
    sys.path.insert(0, gaston_src)
    print(f"Using GASTON from: {gaston_src}")

from gaston import neural_net, dp_related

# ============================================================
# 2. 加载数据
# ============================================================
print(f"\n=== Loading Slice {args.slice} ===")
adata = load_dlpfc_slice(args.slice, data_root=GASTON_DATA_DIR)
adata.var_names_make_unique()

coords = np.array(adata.obsm['spatial'], dtype=np.float32)
# GASTON: Z-score 标准化
S_normalized = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-7)
S_torch = torch.tensor(S_normalized, dtype=torch.float32).to(device)

# Pearson 残差 PCA (与 GASTON 官方一致)
adata_hvg = adata.copy()
sc.experimental.pp.highly_variable_genes(
    adata_hvg, flavor="pearson_residuals", n_top_genes=args.n_top_genes
)
adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable']].copy()
sc.experimental.pp.normalize_pearson_residuals(adata_hvg)
sc.tl.pca(adata_hvg, n_comps=args.n_comps)
A_mat = adata_hvg.obsm['X_pca'].astype(np.float32)
A_torch = torch.tensor(A_mat, dtype=torch.float32).to(device)

print(f"  Spots: {adata.shape[0]}, Genes(top): {args.n_top_genes}, "
      f"PCs: {args.n_comps}")

# ============================================================
# 3. 训练 GASTON
# ============================================================
print(f"\n=== Training GASTON ({args.epochs} epochs) ===")
s_hidden_list = [int(x) for x in args.s_hidden.split(',')]
a_hidden_list = [int(x) for x in args.a_hidden.split(',')]

start_time = time.time()

neural_net.train(
    S_torch, A_torch,
    S_hidden_list=s_hidden_list,
    A_hidden_list=a_hidden_list,
    epochs=args.epochs,
    checkpoint=500,
    device=device,
    save_dir=save_dir,
    lr=args.lr,
    seed=args.seed,
    save_final=True,
    optim='adam',  # 与官方一致
    sigma=0.2  # 官方 CLI 默认值 (neural_net.train 默认 0.1, 但 CLI 入口为 0.2)
)

runtime = time.time() - start_time
print(f"Training completed in {runtime:.2f}s")

# ============================================================
# 4. 加载训练好的模型并提取 isodepth
# ============================================================
print("\n=== Loading Model and Extracting Isodepth ===")

# 直接加载 final_model.pt
model_path = os.path.join(save_dir, 'final_model.pt')
if not os.path.exists(model_path):
    # 找最新的 epoch 模型
    import glob
    model_files = glob.glob(os.path.join(save_dir, 'model_epoch_*.pt'))
    if model_files:
        model_path = max(model_files, key=lambda f: int(f.split('_')[-1][:-3]))
    else:
        raise FileNotFoundError(f"No model found in {save_dir}")

print(f"  Loading model from: {model_path}")
gaston_model = torch.load(model_path, map_location=device)
gaston_model.eval()

# 加载 S 和 A 参考数据
S_ref = torch.load(os.path.join(save_dir, 'Storch.pt'), map_location=device)
A_ref = torch.load(os.path.join(save_dir, 'Atorch.pt'), map_location=device)

n_clusters = get_n_clusters(args.slice)

# 使用 dp_related 计算 isodepth 和 labels
# dp_related 内部创建 CPU tensor，所以需要把 model 临时移到 CPU
gaston_model.cpu()
gaston_isodepth, gaston_labels = dp_related.get_isodepth_labels(
    gaston_model, A_ref.detach().cpu().numpy(), S_ref.detach().cpu().numpy(), n_clusters
)
gaston_model.to(device)

# ============================================================
# 5. 核心：计算 GASTON 空间梯度
# ============================================================
print("Computing spatial gradient from isodepth function (f_S)...")

# GASTON 的属性名是 spatial_embedding (对应 f_S)
f_S = gaston_model.spatial_embedding
f_S.eval()

# 使用全量 S 计算梯度 (而非 S_ref 的子集)
S_grad = S_torch.clone().detach().requires_grad_(True)
isodepth_pred = f_S(S_grad)

if isodepth_pred.dim() > 1 and isodepth_pred.shape[1] > 1:
    isodepth_scalar = isodepth_pred[:, 0]
else:
    isodepth_scalar = isodepth_pred.squeeze()

grads = torch.autograd.grad(
    outputs=isodepth_scalar.sum(),
    inputs=S_grad,
    create_graph=False
)[0]
gaston_gradient = torch.norm(grads, p=2, dim=1).detach().cpu().numpy()

# ============================================================
# 6. 构建输出
# ============================================================
adata_out = anndata.AnnData(A_mat)
adata_out.obs_names = adata.obs_names
adata_out.uns['spatial'] = adata.uns['spatial']
adata_out.obsm['spatial'] = coords
adata_out.obs['GASTON_Isodepth'] = gaston_isodepth
adata_out.obs['GASTON_Domain'] = pd.Categorical(gaston_labels.astype(str))
adata_out.obs['Spatial_Gradient'] = gaston_gradient

gt = load_ground_truth(args.slice)
if gt is not None:
    adata_out.obs['Ground Truth'] = gt.reindex(adata_out.obs_names).astype(str).values

# ============================================================
# 7. 保存
# ============================================================
gradient_df = pd.DataFrame({
    'spot': adata_out.obs_names,
    'isodepth': gaston_isodepth,
    'gradient_magnitude': gaston_gradient,
})
gradient_df.to_csv(os.path.join(save_dir, "spatial_gradient_values.csv"), index=False)

# ============================================================
# 8. 可视化
# ============================================================
print("\n=== Generating Visualizations ===")

plot_spatial_gradient(
    adata_out, gradient_col='Spatial_Gradient',
    title=f'GASTON Spatial Gradient (Slice {args.slice})',
    save_path=os.path.join(save_dir, "gradient_boundary.pdf")
)

# Isodepth 图
fig, ax = plt.subplots(figsize=(6, 6))
lib_id = list(adata_out.uns['spatial'].keys())[0]
sc.pl.spatial(adata_out, color='GASTON_Isodepth', color_map='turbo',
             ax=ax, show=False, title="", library_id=lib_id)
ax.set_title(f'GASTON Isodepth (Slice {args.slice})',
             fontfamily='Times New Roman', fontsize=14)
ax.set_axis_off()
fig.savefig(os.path.join(save_dir, "isodepth.pdf"), bbox_inches='tight', dpi=300)
plt.close(fig)

if 'Ground Truth' in adata_out.obs.columns:
    plot_gradient_vs_ground_truth(
        adata_out, gradient_col='Spatial_Gradient',
        gt_col='Ground Truth', title='GASTON',
        save_path=os.path.join(save_dir, "gradient_vs_gt.pdf")
    )

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(gaston_gradient, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
ax.set_xlabel('Spatial Gradient Magnitude', fontfamily='Times New Roman', fontsize=13)
ax.set_ylabel('Frequency', fontfamily='Times New Roman', fontsize=13)
ax.set_title(f'GASTON Gradient Distribution - Slice {args.slice}',
             fontfamily='Times New Roman', fontsize=14)
fig.savefig(os.path.join(save_dir, "gradient_histogram.pdf"), bbox_inches='tight')
plt.close(fig)

summary = {
    'Model': 'GASTON',
    'Slice': args.slice,
    'S_hidden': args.s_hidden, 'A_hidden': args.a_hidden,
    'Epochs': args.epochs, 'LR': args.lr,
    'Runtime(s)': round(runtime, 2),
    'Gradient_Mean': float(np.mean(gaston_gradient)),
    'Gradient_Std': float(np.std(gaston_gradient)),
}
pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)

print(f"\n=== GASTON Gradient Experiment Complete ===")
print(f"Results: {save_dir}")
print(f"Gradient: mean={summary['Gradient_Mean']:.4f}, std={summary['Gradient_Std']:.4f}")
