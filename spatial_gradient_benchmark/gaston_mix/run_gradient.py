"""
GASTON-Mix 空间梯度实验
使用原始 GASTON-Mix 仓库 (models/gaston_mix/src) 的 GASTON_MoE 模型。

官方架构:
  - num_experts 个 isodepth 网络 (spatial coords -> 1D isodepth)
  - num_experts 个 expression 网络 (isodepth -> gene expression)
  - Gating network (spatial coords -> expert weights)
  - 激活函数: ReLU
  - 无 Positional encoding (官方默认 OFF)

空间梯度 = d(dominant_isodepth)/d(spatial_coords)

用法:
    conda run -n AINR python run_gradient.py --slice 151673
"""
import os, sys, time, argparse, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import scanpy as sc, anndata
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (RESULTS_DIR, DEFAULT_SEED, get_n_clusters, GASTON_DATA_DIR)
from data_utils import load_dlpfc_slice, load_ground_truth
from visualization import (plot_spatial_gradient, plot_gradient_vs_ground_truth)

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description='GASTON-Mix Spatial Gradient Experiment')
parser.add_argument('--slice', type=str, default='151673')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10000)  # 与 GASTON 一致（官方无默认值）
parser.add_argument('--spatial_arch', nargs='*', type=int, default=[])   # 官方默认: []
parser.add_argument('--expression_arch', nargs='*', type=int, default=[]) # 官方默认: []
parser.add_argument('--gating_arch', nargs='*', type=int, default=[])
parser.add_argument('--n_top_genes', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
args = parser.parse_args()

exp_name = f"GASTONMix_gradient"
save_dir = os.path.join(RESULTS_DIR, exp_name, args.slice)
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_clusters = get_n_clusters(args.slice)
print(f"Device: {device}, K={n_clusters}")
print(f"Spatial arch: {args.spatial_arch}, Expression arch: {args.expression_arch}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ============================================================
# 1. 加载 GASTON-Mix 原始代码
# ============================================================
gastonmix_src = os.path.join(os.path.dirname(__file__), '..', 'models', 'gaston_mix', 'src')
sys.path.insert(0, gastonmix_src)
from gastonmix.run_moe_script import GASTON_MoE

# ============================================================
# 2. 加载数据
# ============================================================
print(f"\n=== Loading Slice {args.slice} ===")
adata = load_dlpfc_slice(args.slice, data_root=GASTON_DATA_DIR)
adata.var_names_make_unique()

# Pearson 残差 + PCA
sc.experimental.pp.highly_variable_genes(
    adata, flavor="pearson_residuals", n_top_genes=args.n_top_genes
)
adata = adata[:, adata.var['highly_variable']].copy()
sc.experimental.pp.normalize_pearson_residuals(adata)

gene_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
coords = np.array(adata.obsm['spatial'], dtype=np.float32)

# Z-score 标准化 (与 GASTON 一致)
S_raw = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-7)
A_raw = gene_expr.astype(np.float32)

S_torch = torch.FloatTensor(S_raw).to(device)
A_torch = torch.FloatTensor(A_raw).to(device)
G = A_raw.shape[1]  # gene dim

print(f"  Spots: {len(S_raw)}, Genes: {G}")

# ============================================================
# 3. 构建 GASTON-Mix MoE 模型
# ============================================================
print(f"\n=== Building GASTON-Mix MoE (K={n_clusters} experts) ===")

moe_model = GASTON_MoE(
    G=G,
    S_hidden_list=args.spatial_arch,
    A_hidden_list=args.expression_arch,
    gating_hidden_list=args.gating_arch,
    num_experts=n_clusters,
    k=1,
    activation_fn=nn.ReLU(),
    noisy_gating=False,
    routing_loss=False,
    pos_encoding_i=False,  # 官方默认: OFF (不传 --pos_encoding_isodepth 时不启用)
    pos_encoding_g=False,  # 官方默认: OFF (不传 --pos_encoding_gating 时不启用)
).to(device)

optimizer = torch.optim.Adam(moe_model.parameters(), lr=args.lr)

# ============================================================
# 4. 训练 (与官方 run_moe_script.py 一致: 每 epoch 随机 shuffle + mini-batch)
# ============================================================
print(f"\n=== Training ({args.epochs} epochs) ===")
start_time = time.time()
loss_history = []
N = len(S_torch)
batch_size = min(args.batch_size, N)

moe_model.train()
for epoch in range(args.epochs):
    # shuffle
    indices = torch.randperm(N, device=device)
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, N, batch_size):
        batch_idx = indices[i:i + batch_size]
        S_batch = S_torch[batch_idx].clone().detach().requires_grad_(True)
        A_batch = A_torch[batch_idx]

        predicted_expression, expert_gates, full_gates, logits, _ = moe_model.forward(S_batch)

        recon_loss = F.mse_loss(predicted_expression, A_batch)
        total_loss = recon_loss  # 与官方一致: 丢弃 forward 返回的 regularization_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)

    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

runtime = time.time() - start_time
print(f"Training completed in {runtime:.2f}s")

# ============================================================
# 5. 计算空间梯度
# ============================================================
print("\n=== Computing GASTON-Mix Spatial Gradients ===")

moe_model.eval()
with torch.no_grad():
    predicted_expression, _, full_gates, logits, _ = moe_model.forward(S_torch)
    recon_np = predicted_expression.cpu().numpy()

# 获取每个点的主导 expert
dominant_expert = logits.argmax(dim=1)  # [N]

# 自定义 gradient-safe positional encoding (原始 PE 有 .detach() 切断梯度)
def pe_grad_safe(coords, enc_dim=8, sigma=0.1, include_orig_coords=False):
    freqs = 2 * np.pi * sigma ** (torch.arange(enc_dim//2, dtype=torch.float,
                                                device=coords.device) / enc_dim)
    freqs = freqs.reshape(1, 1, -1)
    coords_exp = coords.unsqueeze(-1)  # [N, 2, 1] - NO detach!
    freqs = coords_exp * freqs  # [N, 2, enc_dim/2]
    s = torch.sin(freqs)
    c = torch.cos(freqs)
    x = torch.cat((s, c), axis=-1)  # [N, 2, enc_dim]
    x = x.reshape(x.shape[0], -1)   # [N, 2*enc_dim]
    if include_orig_coords:
        x = torch.cat((coords, x), dim=1)
    return x

# 对每个点计算主导 expert 的 isodepth 梯度
S_grad = S_torch.clone().detach().requires_grad_(True)

# 应用 gradient-safe PE
if moe_model.pos_encoding_i:
    S_grad_pe = pe_grad_safe(S_grad, moe_model.enc_dim_i,
                             moe_model.sigma_i, moe_model.include_orig_coords)
else:
    S_grad_pe = S_grad

grad_mags = [(i, 0.0) for i in range(len(S_torch))]
for k in range(n_clusters):
    mask = dominant_expert == k
    if mask.sum() == 0:
        continue
    S_k = S_grad_pe[mask]
    iso_k = moe_model.isodepths_list[k](S_k)
    grad_k = torch.autograd.grad(outputs=iso_k.sum(), inputs=S_grad,
                                  create_graph=False, retain_graph=True)[0]
    grad_mag_k = torch.norm(grad_k[mask], p=2, dim=1)
    for j, idx in enumerate(mask.nonzero(as_tuple=True)[0].cpu().numpy()):
        grad_mags[idx] = (idx, grad_mag_k[j].detach().cpu().item())

grad_mags.sort(key=lambda x: x[0])
gastmix_gradient = np.array([m[1] for m in grad_mags])

# ============================================================
# 6. 获取主导 isodepth
# ============================================================
with torch.no_grad():
    dominant_full = logits.argmax(dim=1).cpu().numpy()
    isodepth_vals = np.zeros(len(S_torch))
    # 使用 gradient-safe PE for isodepth computation
    if moe_model.pos_encoding_i:
        S_pe_no_grad = pe_grad_safe(S_torch, moe_model.enc_dim_i,
                                     moe_model.sigma_i, moe_model.include_orig_coords)
    else:
        S_pe_no_grad = S_torch
    for k in range(n_clusters):
        mask = dominant_full == k
        if mask.sum() > 0:
            S_k = S_pe_no_grad[mask]
            iso_k = moe_model.isodepths_list[k](S_k)
            isodepth_vals[mask] = iso_k.squeeze().cpu().numpy()

# ============================================================
# 7. 构建输出
# ============================================================
adata_out = adata.copy()
adata_out.obs['GMix_DominantDomain'] = pd.Categorical(dominant_full.astype(str))
adata_out.obs['GMix_Isodepth'] = isodepth_vals
adata_out.obs['Spatial_Gradient'] = gastmix_gradient
adata_out.layers['denoised'] = recon_np

gt = load_ground_truth(args.slice)
if gt is not None:
    adata_out.obs['Ground Truth'] = gt.reindex(adata_out.obs_names).astype(str).values

# GMM clustering (on isodepth values)
gm = GaussianMixture(n_components=n_clusters, covariance_type='tied',
                    reg_covar=1e-4, random_state=args.seed)
gastmix_labels = gm.fit_predict(isodepth_vals.reshape(-1, 1))
adata_out.obs['GMix_Domain'] = pd.Categorical(gastmix_labels.astype(str))

# ============================================================
# 8. 保存
# ============================================================
gradient_df = pd.DataFrame({
    'spot': adata_out.obs_names,
    'isodepth': isodepth_vals,
    'gradient_magnitude': gastmix_gradient,
    'dominant_expert': dominant_full,
})
gradient_df.to_csv(os.path.join(save_dir, "spatial_gradient_values.csv"), index=False)
torch.save(moe_model.state_dict(), os.path.join(save_dir, "gaston_mix_model.pth"))

# ============================================================
# 9. 可视化
# ============================================================
print("\n=== Generating Visualizations ===")

plot_spatial_gradient(
    adata_out, gradient_col='Spatial_Gradient',
    title=f'GASTON-Mix Spatial Gradient (Slice {args.slice})',
    save_path=os.path.join(save_dir, "gradient_boundary.pdf")
)

# Isodepth
fig, ax = plt.subplots(figsize=(6, 6))
lib_id = list(adata_out.uns['spatial'].keys())[0]
sc.pl.spatial(adata_out, color='GMix_Isodepth', color_map='turbo',
             ax=ax, show=False, title="", library_id=lib_id)
ax.set_title(f'GASTON-Mix Isodepth (Slice {args.slice})',
             fontfamily='Times New Roman', fontsize=14)
ax.set_axis_off()
fig.savefig(os.path.join(save_dir, "isodepth.pdf"), bbox_inches='tight', dpi=300)
plt.close(fig)

if 'Ground Truth' in adata_out.obs.columns:
    plot_gradient_vs_ground_truth(
        adata_out, gradient_col='Spatial_Gradient',
        gt_col='Ground Truth', title='GASTON-Mix',
        save_path=os.path.join(save_dir, "gradient_vs_gt.pdf")
    )

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(gastmix_gradient, bins=50, color='crimson', edgecolor='white', alpha=0.8)
ax.set_xlabel('Spatial Gradient Magnitude', fontfamily='Times New Roman', fontsize=13)
ax.set_ylabel('Frequency', fontfamily='Times New Roman', fontsize=13)
ax.set_title(f'GASTON-Mix Gradient Distribution - Slice {args.slice}',
             fontfamily='Times New Roman', fontsize=14)
fig.savefig(os.path.join(save_dir, "gradient_histogram.pdf"), bbox_inches='tight')
plt.close(fig)

summary = {
    'Model': 'GASTON-Mix',
    'Slice': args.slice, 'K': n_clusters,
    'Spatial_arch': str(args.spatial_arch),
    'Expression_arch': str(args.expression_arch),
    'Epochs': args.epochs, 'LR': args.lr,
    'Runtime(s)': round(runtime, 2),
    'Gradient_Mean': float(np.mean(gastmix_gradient)),
    'Gradient_Std': float(np.std(gastmix_gradient)),
}
pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)

print(f"\n=== GASTON-Mix Gradient Experiment Complete ===")
print(f"Results: {save_dir}")
print(f"Gradient: mean={summary['Gradient_Mean']:.4f}, std={summary['Gradient_Std']:.4f}")
