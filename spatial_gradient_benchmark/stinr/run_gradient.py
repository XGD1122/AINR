"""
STINR 空间梯度实验 — 完整 DeconvNet (INR + 反卷积模块)

官方 STINR 架构 (DeconvNet from STINR/networks.py):
  - INR Encoder:  SIREN 3→200→200→30→gene_dim   (coord → mid_fea)
  - Latent Proj:  Dense gene_dim→latent_dim       (mid_fea → Z)
  - Decoder:      SIREN latent_dim→200→gene_dim   (Z → recon)
  - Deconv:       Z + slice_emb → beta (cell type proportions) + alpha
  - gamma:        每切片 bias
  - 总损失:       decon_loss + fea_loss
    - decon_loss: -5 * Poisson_NLL(count | beta, alpha, gamma, basis)
    - fea_loss:   1*||X - mid_fea||₂ + 2*||X - recon||₂

数据: D:\STINR-123\STINR-123\73-76h5ad\ (切片 151673-151676)
训练步数: 14001 (hardcoded in DeconvNet)
优化器: Adamax lr=0.001

用法:
    python run_gradient.py --slice 151673
"""
import os, sys, time, argparse, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import scanpy as sc, anndata
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---- paths ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'stinr'))

from config import (RESULTS_DIR, get_n_clusters, PROJECT_ROOT)
from data_utils import load_ground_truth
from visualization import (plot_spatial_gradient, plot_gradient_vs_ground_truth)
from STINR.networks import DeconvNet

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description='STINR Gradient — Full DeconvNet')
parser.add_argument('--slice', type=str, default='151673',
                    choices=['151507','151508','151509','151510',
                             '151669','151670','151671','151672',
                             '151673','151674','151675','151676'])
parser.add_argument('--stinr_base', type=str,
                    default='D:/STINR-123/STINR-123')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--slice_emb_dim', type=int, default=16)
parser.add_argument('--eval_interval', type=int, default=500)
args = parser.parse_args()

# 切片 → 数据目录 + 内部索引映射
SLICE_TO_DATADIR = {
    '151507': '07-10h5ad', '151508': '07-10h5ad',
    '151509': '07-10h5ad', '151510': '07-10h5ad',
    '151669': '69-72h5ad', '151670': '69-72h5ad',
    '151671': '69-72h5ad', '151672': '69-72h5ad',
    '151673': '73-76h5ad', '151674': '73-76h5ad',
    '151675': '73-76h5ad', '151676': '73-76h5ad',
}
SLICE_TO_IDX = {
    '151507': 0, '151508': 1, '151509': 2, '151510': 3,
    '151669': 0, '151670': 1, '151671': 2, '151672': 3,
    '151673': 0, '151674': 1, '151675': 2, '151676': 3,
}

# DeconvNet 内部 hardcode seed=1, training_steps=14001 (无法覆盖)
TRAINING_STEPS = 14001

exp_name = "STINR_gradient"
save_dir = os.path.join(RESULTS_DIR, exp_name, args.slice)
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 1. 加载数据 (73-76h5ad)
# ============================================================
stinr_data_dir = os.path.join(args.stinr_base, SLICE_TO_DATADIR[args.slice])
target_idx = SLICE_TO_IDX[args.slice]

print(f"\n=== STINR: Loading data from {stinr_data_dir} ===")

adata_basis = sc.read_h5ad(os.path.join(stinr_data_dir, 'adata_basis_DLPFC.h5ad'))
adata_st = sc.read_h5ad(os.path.join(stinr_data_dir, 'adata_st_DLPFC.h5ad'))
adata_st.var_names_make_unique()

raw_paths = [os.path.join(stinr_data_dir, f'adata_st_list_raw{i}.h5ad') for i in range(4)]
adata_st_list_raw = [sc.read_h5ad(p) for p in raw_paths]

n_celltypes = adata_basis.shape[0]
n_slices = len(set(adata_st.obs["slice"].values))
gene_dim = adata_st.shape[1]

print(f"  Spots: {adata_st.shape[0]}, Genes: {gene_dim}")
print(f"  Cell types: {n_celltypes}, Slices: {n_slices}")
print(f"  Target: slice {args.slice} -> list_raw[{target_idx}]")

# ============================================================
# 2. 构建完整 DeconvNet
# ============================================================
print(f"\n=== Building Full DeconvNet ===")

# hidden_dims[0]=gene_dim, hidden_dims[2]=latent_dim (hidden_dims[1] unused)
hidden_dims = [gene_dim, 512, args.latent_dim]

# 准备 tensor
X = torch.from_numpy(adata_st.X.toarray() if hasattr(adata_st.X, 'toarray') else np.array(adata_st.X)).float().to(device)
A = torch.from_numpy(np.array(adata_st.obsm["graph"])).float().to(device)
Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(device)
lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(device)
slice_labels = torch.from_numpy(np.array(adata_st.obs["slice"].values)).long().to(device)
basis = torch.from_numpy(np.array(adata_basis.X)).float().to(device)
coord = torch.from_numpy(np.array(adata_st.obsm['3D_coor'])).float().to(device)

adj_dim = A.shape[1]

net = DeconvNet(
    hidden_dims=hidden_dims,
    n_celltypes=n_celltypes,
    n_slices=n_slices,
    n_heads=1,
    slice_emb_dim=args.slice_emb_dim,
    adj_dim=adj_dim,
    coef_fe=0.1,
).to(device)

optimizer = torch.optim.Adamax(net.parameters(), lr=args.lr)

# ============================================================
# 3. 训练
# ============================================================
print(f"\n=== Training Full DeconvNet ({TRAINING_STEPS} steps) ===")
start_time = time.time()
loss_history = []

net.train()
for step in range(TRAINING_STEPS):
    loss, recon, denoise, Z, ind_min, ind_max = net(
        coord=coord,
        adj_matrix=A,
        node_feats=X,
        count_matrix=Y,
        library_size=lY,
        slice_label=slice_labels,
        basis=basis,
        step=step
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (step + 1) % args.eval_interval == 0:
        print(f"  Step {step+1}/{TRAINING_STEPS}, Loss: {loss.item():.4f}")

runtime = time.time() - start_time
print(f"Training completed in {runtime:.2f}s")

pd.DataFrame({'step': range(1, TRAINING_STEPS+1), 'loss': loss_history}).to_csv(
    os.path.join(save_dir, "loss_history.csv"), index=False
)

# ============================================================
# 4. 提取 Latent Z + 计算空间梯度
# ============================================================
print("\n=== Computing Spatial Gradients ===")
net.eval()

# 获取所有 spot 的 Z
with torch.no_grad():
    Z_all, beta_all, alpha_all, gamma_all = net.evaluate(A, coord, X, slice_labels)
    latent_np = Z_all.cpu().numpy()

# 只取目标切片
mask = (slice_labels == target_idx).cpu().numpy()
print(f"  Target slice spots: {mask.sum()} / {len(mask)}")

# 梯度: d(Z)/d(coord), 对目标切片坐标
target_coord = coord[mask].detach().clone().requires_grad_(True)

# DeconvNet 的 encoder 用 self.coord, 这里直接调用 encoder_layer
net.coord = target_coord / 100.0  # 与 forward 一致: coord/100
mid_fea = net.encoder_layer0(net.coord)
Z_target = net.encoder_layer1(mid_fea)

grads = torch.autograd.grad(
    outputs=Z_target, inputs=target_coord,
    grad_outputs=torch.ones_like(Z_target),
    create_graph=False
)[0]
stinr_gradient = torch.norm(grads, p=2, dim=1).detach().cpu().numpy()

# ============================================================
# 5. 构建输出
# ============================================================
adata_raw = adata_st_list_raw[target_idx].copy()
adata_raw.var_names_make_unique()

# 用 raw 的 spatial 坐标来匹配 obs_names
# raw obs_names 格式: "spot_barcode" (无 -slice 后缀)
adata_out = anndata.AnnData(np.array(adata_raw.X.toarray() if hasattr(adata_raw.X, 'toarray') else adata_raw.X))
adata_out.obs_names = adata_raw.obs_names
adata_out.uns['spatial'] = adata_raw.uns['spatial']
adata_out.obsm['spatial'] = np.array(adata_raw.obsm['spatial'])
adata_out.obsm['latent'] = latent_np[mask]
adata_out.obs['Spatial_Gradient'] = stinr_gradient

# GMM
n_clusters = get_n_clusters(args.slice)
gm = GaussianMixture(n_components=n_clusters, covariance_type='tied',
                     reg_covar=1e-4, init_params='kmeans', random_state=42)
stinr_labels = gm.fit_predict(latent_np[mask])
adata_out.obs['STINR_Domain'] = pd.Categorical(stinr_labels.astype(str))

# Ground Truth
gt = load_ground_truth(args.slice)
if gt is not None:
    adata_out.obs['Ground Truth'] = gt.reindex(adata_out.obs_names).astype(str).values

# ============================================================
# 6. 保存
# ============================================================
gradient_df = pd.DataFrame({
    'spot': adata_out.obs_names,
    'gradient_magnitude': stinr_gradient,
})
gradient_df.to_csv(os.path.join(save_dir, "spatial_gradient_values.csv"), index=False)
torch.save(net.state_dict(), os.path.join(save_dir, "stinr_deconvnet_model.pth"))

# ============================================================
# 7. 可视化
# ============================================================
print("\n=== Generating Visualizations ===")
plot_spatial_gradient(
    adata_out, gradient_col='Spatial_Gradient',
    title=f'STINR Spatial Gradient (Slice {args.slice})',
    save_path=os.path.join(save_dir, "gradient_boundary.pdf")
)

if 'Ground Truth' in adata_out.obs.columns:
    plot_gradient_vs_ground_truth(
        adata_out, gradient_col='Spatial_Gradient',
        gt_col='Ground Truth', title='STINR',
        save_path=os.path.join(save_dir, "gradient_vs_gt.pdf")
    )

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(stinr_gradient, bins=50, color='mediumpurple', edgecolor='white', alpha=0.8)
ax.set_xlabel('Spatial Gradient Magnitude', fontfamily='Times New Roman', fontsize=13)
ax.set_ylabel('Frequency', fontfamily='Times New Roman', fontsize=13)
ax.set_title(f'STINR Gradient Distribution - Slice {args.slice}',
             fontfamily='Times New Roman', fontsize=14)
fig.savefig(os.path.join(save_dir, "gradient_histogram.pdf"), bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(loss_history, linewidth=0.5)
ax.set_xlabel('Step', fontfamily='Times New Roman', fontsize=13)
ax.set_ylabel('Loss', fontfamily='Times New Roman', fontsize=13)
ax.set_title(f'STINR Training Loss - Slice {args.slice}',
             fontfamily='Times New Roman', fontsize=14)
fig.savefig(os.path.join(save_dir, "loss_curve.pdf"), bbox_inches='tight')
plt.close(fig)

summary = {
    'Model': 'STINR (Full DeconvNet)',
    'Slice': args.slice,
    'Architecture': f'DeconvNet(INR 3→200→200→30→{gene_dim}, Z={args.latent_dim}, deconv={n_celltypes}CT)',
    'Optimizer': 'Adamax lr=0.001',
    'Training Steps': TRAINING_STEPS,
    'Runtime(s)': round(runtime, 2),
    'Gradient_Mean': float(np.mean(stinr_gradient)),
    'Gradient_Std': float(np.std(stinr_gradient)),
    'Gradient_Min': float(np.min(stinr_gradient)),
    'Gradient_Max': float(np.max(stinr_gradient)),
}
pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)

print(f"\n=== STINR Gradient Experiment Complete ===")
print(f"Results: {save_dir}")
print(f"Gradient: mean={summary['Gradient_Mean']:.4f}, std={summary['Gradient_Std']:.4f}")
