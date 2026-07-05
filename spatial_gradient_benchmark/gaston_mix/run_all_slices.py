"""
GASTON-Mix 批量梯度实验 — 逐切片 MoE 训练，自动循环全部 12 个 DLPFC 切片

官方架构:
  - num_experts 个 isodepth 网络 (spatial coords -> 1D isodepth)
  - num_experts 个 expression 网络 (isodepth -> gene expression)
  - Gating network (spatial coords -> expert weights)
  - 激活函数: ReLU, 无 Positional encoding (官方默认)
  - K = n_clusters (层数)

空间梯度 = d(dominant_isodepth)/d(spatial_coords)

用法:
    conda run -n STINR python gaston_mix/run_all_slices.py
"""
import os, sys, time, argparse, warnings, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import scanpy as sc, anndata
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (RESULTS_DIR, DEFAULT_SEED, get_n_clusters, GASTON_DATA_DIR, ALL_SLICES)
from data_utils import load_dlpfc_slice, load_ground_truth
from visualization import (plot_spatial_gradient, plot_gradient_vs_ground_truth)

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description='GASTON-Mix Batch Gradient Experiment')
parser.add_argument('--slices', type=str, nargs='+', default=None,
                    help='Slice IDs (default: all 12)')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--spatial_arch', nargs='*', type=int, default=[])
parser.add_argument('--expression_arch', nargs='*', type=int, default=[])
parser.add_argument('--gating_arch', nargs='*', type=int, default=[])
parser.add_argument('--n_top_genes', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
args = parser.parse_args()

slices_to_run = args.slices if args.slices else ALL_SLICES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GASTON-Mix: spatial_arch={args.spatial_arch}, expr_arch={args.expression_arch}, "
      f"gating_arch={args.gating_arch}, epochs={args.epochs}, HVG={args.n_top_genes}")
print(f"Slices to run: {slices_to_run}")

# ============================================================
# 加载 GASTON-Mix 原始代码
# ============================================================
gmix_src = os.path.join(os.path.dirname(__file__), '..', 'models', 'gaston_mix', 'src')
sys.path.insert(0, gmix_src)
from gastonmix.run_moe_script import GASTON_MoE

# gradient-safe positional encoding (原始 PE 有 .detach() 切断梯度)
def pe_grad_safe(coords, enc_dim=8, sigma=0.1, include_orig_coords=False):
    freqs = 2 * np.pi * sigma ** (torch.arange(enc_dim//2, dtype=torch.float,
                                                device=coords.device) / enc_dim)
    freqs = freqs.reshape(1, 1, -1)
    coords_exp = coords.unsqueeze(-1)
    freqs = coords_exp * freqs
    s = torch.sin(freqs)
    c = torch.cos(freqs)
    x = torch.cat((s, c), axis=-1)
    x = x.reshape(x.shape[0], -1)
    if include_orig_coords:
        x = torch.cat((coords, x), dim=1)
    return x

# ============================================================
# 全局记录
# ============================================================
all_summaries = []
total_start = time.time()

for slice_id in slices_to_run:
    print(f"\n{'='*70}")
    print(f"GASTON-Mix: Slice {slice_id}")
    print(f"{'='*70}")

    n_clusters = get_n_clusters(slice_id)
    print(f"  K (layers/experts): {n_clusters}")

    exp_name = "GASTONMix_gradient"
    save_dir = os.path.join(RESULTS_DIR, exp_name, slice_id)
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 加载数据 ----
    print(f"Loading Slice {slice_id}...")
    adata = load_dlpfc_slice(slice_id, data_root=GASTON_DATA_DIR)
    adata.var_names_make_unique()

    # Pearson 残差 + PCA (GASTON-Mix 直接使用基因表达, 不做 PCA)
    sc.experimental.pp.highly_variable_genes(
        adata, flavor="pearson_residuals", n_top_genes=args.n_top_genes
    )
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.experimental.pp.normalize_pearson_residuals(adata)

    gene_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    coords = np.array(adata.obsm['spatial'], dtype=np.float32)

    # Z-score 标准化
    S_raw = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-7)
    A_raw = gene_expr.astype(np.float32)

    S_torch = torch.FloatTensor(S_raw).to(device)
    A_torch = torch.FloatTensor(A_raw).to(device)
    G = A_raw.shape[1]
    N = len(S_raw)

    print(f"  Spots: {N}, Genes: {G}")

    # ---- 构建模型 ----
    print(f"Building GASTON-Mix MoE (K={n_clusters} experts)...")
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
        pos_encoding_i=False,
        pos_encoding_g=False,
    ).to(device)

    optimizer = torch.optim.Adam(moe_model.parameters(), lr=args.lr)

    # ---- 训练 ----
    print(f"Training ({args.epochs} epochs, batch_size={args.batch_size})...")
    slice_start = time.time()
    loss_history = []
    batch_size = min(args.batch_size, N)

    moe_model.train()
    for epoch in range(args.epochs):
        indices = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            batch_idx = indices[i:i + batch_size]
            S_batch = S_torch[batch_idx].clone().detach().requires_grad_(True)
            A_batch = A_torch[batch_idx]

            predicted_expression, expert_gates, full_gates, logits, _ = moe_model.forward(S_batch)
            recon_loss = F.mse_loss(predicted_expression, A_batch)

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            epoch_loss += recon_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    runtime = time.time() - slice_start
    print(f"  Training completed in {runtime:.2f}s")

    # ---- 计算空间梯度 ----
    print("Computing spatial gradients...")
    moe_model.eval()

    with torch.no_grad():
        predicted_expression, _, full_gates, logits, _ = moe_model.forward(S_torch)
        recon_np = predicted_expression.cpu().numpy()

    # 每个点的 dominant expert
    dominant_expert = logits.argmax(dim=1)

    # 对每个点计算主导 expert 的 isodepth 梯度
    S_grad = S_torch.clone().detach().requires_grad_(True)

    if moe_model.pos_encoding_i:
        S_grad_pe = pe_grad_safe(S_grad, moe_model.enc_dim_i,
                                 moe_model.sigma_i, moe_model.include_orig_coords)
    else:
        S_grad_pe = S_grad

    grad_mags = np.zeros(N, dtype=np.float32)
    for k in range(n_clusters):
        mask = dominant_expert == k
        if mask.sum() == 0:
            continue
        S_k = S_grad_pe[mask]
        iso_k = moe_model.isodepths_list[k](S_k)
        grad_k = torch.autograd.grad(outputs=iso_k.sum(), inputs=S_grad,
                                      create_graph=False, retain_graph=True)[0]
        grad_mag_k = torch.norm(grad_k[mask], p=2, dim=1)
        mask_np = mask.cpu().numpy()
        grad_mags[mask_np] = grad_mag_k.detach().cpu().numpy()

    gastmix_gradient = grad_mags

    # ---- 获取主导 isodepth ----
    with torch.no_grad():
        dominant_np = dominant_expert.cpu().numpy()
        isodepth_vals = np.zeros(N)
        if moe_model.pos_encoding_i:
            S_pe = pe_grad_safe(S_torch, moe_model.enc_dim_i,
                               moe_model.sigma_i, moe_model.include_orig_coords)
        else:
            S_pe = S_torch
        for k in range(n_clusters):
            mask = dominant_np == k
            if mask.sum() > 0:
                S_k = S_pe[mask]
                isodepth_vals[mask] = moe_model.isodepths_list[k](S_k).squeeze().cpu().numpy()

    # ---- 构建输出 ----
    adata_out = adata.copy()
    adata_out.obs['GMix_DominantDomain'] = pd.Categorical(dominant_np.astype(str))
    adata_out.obs['GMix_Isodepth'] = isodepth_vals
    adata_out.obs['Spatial_Gradient'] = gastmix_gradient
    adata_out.layers['denoised'] = recon_np

    gt = load_ground_truth(slice_id)
    if gt is not None:
        adata_out.obs['Ground Truth'] = gt.reindex(adata_out.obs_names).astype(str).values

    # GMM clustering on isodepth
    gm = GaussianMixture(n_components=n_clusters, covariance_type='tied',
                        reg_covar=1e-4, random_state=args.seed)
    gastmix_labels = gm.fit_predict(isodepth_vals.reshape(-1, 1))
    adata_out.obs['GMix_Domain'] = pd.Categorical(gastmix_labels.astype(str))

    # ---- 保存 ----
    gradient_df = pd.DataFrame({
        'spot': adata_out.obs_names,
        'isodepth': isodepth_vals,
        'gradient_magnitude': gastmix_gradient,
        'dominant_expert': dominant_np,
    })
    gradient_df.to_csv(os.path.join(save_dir, "spatial_gradient_values.csv"), index=False)
    torch.save(moe_model.state_dict(), os.path.join(save_dir, "gaston_mix_model.pth"))

    # ---- 可视化 ----
    plot_spatial_gradient(
        adata_out, gradient_col='Spatial_Gradient',
        title=f'GASTON-Mix Spatial Gradient (Slice {slice_id})',
        save_path=os.path.join(save_dir, "gradient_boundary.pdf")
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    lib_id = list(adata_out.uns['spatial'].keys())[0]
    sc.pl.spatial(adata_out, color='GMix_Isodepth', color_map='turbo',
                 ax=ax, show=False, title="", library_id=lib_id)
    ax.set_title(f'GASTON-Mix Isodepth (Slice {slice_id})',
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
    ax.set_title(f'GASTON-Mix Gradient Distribution - Slice {slice_id}',
                 fontfamily='Times New Roman', fontsize=14)
    fig.savefig(os.path.join(save_dir, "gradient_histogram.pdf"), bbox_inches='tight')
    plt.close(fig)

    summary = {
        'Model': 'GASTON-Mix',
        'Slice': slice_id,
        'Spots': N,
        'K (Experts)': n_clusters,
        'Spatial_arch': str(args.spatial_arch),
        'Expression_arch': str(args.expression_arch),
        'Epochs': args.epochs, 'LR': args.lr,
        'Runtime(s)': round(runtime, 2),
        'Gradient_Mean': float(np.mean(gastmix_gradient)),
        'Gradient_Std': float(np.std(gastmix_gradient)),
        'Gradient_Min': float(np.min(gastmix_gradient)),
        'Gradient_Max': float(np.max(gastmix_gradient)),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)
    all_summaries.append(summary)

    print(f"  Gradient: mean={summary['Gradient_Mean']:.4f}, std={summary['Gradient_Std']:.4f}")

    # ---- 清理 ----
    del moe_model, optimizer, adata, S_torch, A_torch
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 总汇总
# ============================================================
total_runtime = time.time() - total_start
print(f"\n{'='*70}")
print(f"GASTON-Mix All Slices Complete!")
print(f"Total runtime: {total_runtime:.2f}s ({total_runtime/60:.1f} min)")
print(f"Slices processed: {len(slices_to_run)}")
print(f"{'='*70}")

df_summary = pd.DataFrame(all_summaries)
summary_path = os.path.join(RESULTS_DIR, "GASTONMix_gradient", "all_summaries.csv")
df_summary.to_csv(summary_path, index=False)
print(f"Summary: {summary_path}")
print(df_summary[['Slice', 'Spots', 'Gradient_Mean', 'Gradient_Std', 'Runtime(s)']].to_string(index=False))
