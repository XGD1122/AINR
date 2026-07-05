"""
SUICA 批量梯度实验 — 原始两阶段训练 (GAE + INR)，逐切片循环全部 12 个 DLPFC 切片

原始设置 (完全保留):
  Stage 1 — GAE (Graph Autoencoder):
    - GraphST2D, KNN k=5, include_self=True
    - dim_hidden=[2048, 512, 64], dim_latent=32
    - Adam lr=1e-5, 200 epochs, batch_size=512
    - MSE reconstruction loss

  Stage 2 — INR (SIREN):
    - SirenNet: dim_in=2, dim_hidden=2048, dim_out=32, num_layers=3
    - 拟合 GAE embedding + Phase 2 decoder 重建
    - AdamW lr=1e-4, 2000 epochs (phase=1000), batch_size=8192

空间梯度 = || d(SIREN(coords)) / d(coords) ||_F  (Frobenius norm of 32×2 Jacobian)

用法:
    D:\Anaconda\envs\STINR\python.exe suica/run_all_slices.py
    D:\Anaconda\envs\STINR\python.exe suica/run_all_slices.py --slices 151673 151674
"""
import os, sys, time, argparse, warnings, shutil, yaml, subprocess, glob, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
import scanpy as sc, anndata
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (RESULTS_DIR, DEFAULT_SEED, get_n_clusters, GASTON_DATA_DIR, ALL_SLICES)
from data_utils import load_dlpfc_slice, load_ground_truth
from visualization import (plot_spatial_gradient, plot_gradient_vs_ground_truth)

# ============================================================
# 参数 — 全部使用 SUICA 原始设置
# ============================================================
parser = argparse.ArgumentParser(description='SUICA Batch Gradient Experiment (Original Settings)')
parser.add_argument('--slices', type=str, nargs='+', default=None,
                    help='Slice IDs (default: all 12)')
parser.add_argument('--suica_dir', type=str,
                    default=os.path.join(os.path.dirname(__file__), '..', '..', 'SUICA'))
parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
args = parser.parse_args()

slices_to_run = args.slices if args.slices else ALL_SLICES
suica_dir = os.path.abspath(args.suica_dir)
assert os.path.isdir(suica_dir), f"SUICA not found: {suica_dir}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"SUICA dir: {suica_dir}")
print(f"Slices: {slices_to_run}")
print("Settings: GAE(lr=1e-5, epochs=200, GCN k=5) + INR(SIREN lr=1e-4, epochs=2000, phase=1000)")

# ============================================================
# 导入 SUICA 的 SirenNet
# ============================================================
sys.path.insert(0, suica_dir)
from networks.siren import SirenNet

# ============================================================
# SUICA 坐标归一化 (与 datasets.py _normalize_coordinates 完全一致)
# ============================================================
def suica_normalize_coords(coords, keep_ratio=True):
    """
    与 SUICA datasets.py _normalize_coordinates(keep_ratio=True) 完全一致:
      1. min-max normalize to [0, 1]
      2. center by subtracting 0.5 → [-0.5, 0.5]
      3. scale by 2.0 → [-1, 1]
      4. keep_ratio: scale each dim by (range / max_range)
    """
    coords = coords.astype(np.float64).copy()
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_range, y_range = x_max - x_min, y_max - y_min

    coords[:, 0] = (coords[:, 0] - x_min) / x_range
    coords[:, 1] = (coords[:, 1] - y_min) / y_range
    coords -= 0.5
    coords *= 2.0

    if keep_ratio:
        max_range = max(x_range, y_range)
        scale_x, scale_y = x_range / max_range, y_range / max_range
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y

    return coords


# ============================================================
# SIREN 加载：从 Lightning checkpoint 提取 SirenNet 权重
# ============================================================
def load_siren_from_checkpoint(ckpt_path, device='cuda'):
    """从 INR checkpoint 直接加载 SirenNet 权重 (绕过 Lightning wrapper)"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']

    # 提取 fitting_model.* 权重 (SirenNet)
    siren_state = {}
    for k, v in state_dict.items():
        if k.startswith('fitting_model.'):
            siren_state[k.replace('fitting_model.', '')] = v

    # SirenNet: dim_in=2, dim_hidden=2048, dim_out=32, num_layers=3
    siren = SirenNet(
        dim_in=2, dim_hidden=2048, dim_out=32, num_layers=3,
        w0=1., w0_initial=30., final_activation="Identity"
    ).to(device)
    siren.load_state_dict(siren_state, strict=True)
    siren.eval()
    return siren


# ============================================================
# 梯度计算：逐维 Jacobian → Frobenius norm (修复 .sum() 维度抵消)
# ============================================================
def compute_siren_gradient(siren, coords_norm, device='cuda'):
    """
    计算 SIREN 空间梯度的 Frobenius norm:
      J_i ∈ R^{32×2}: d(emb_i) / d(xy_i)
      gradient_i = ||J_i||_F = sqrt(sum_{d=1}^{32} sum_{c∈{x,y}} (d(emb_d)/d(c))^2)

    使用逐维 autograd 避免 emb.sum() 的维度抵消问题。
    """
    N = coords_norm.shape[0]
    coord_t = torch.FloatTensor(coords_norm).to(device).requires_grad_(True)

    with torch.enable_grad():
        emb = siren(coord_t)  # (N, 32)

    D = emb.shape[1]
    grad_sq = torch.zeros(N, device=device)

    for d in range(D):
        if d > 0:
            # retain_graph needed for all but last dim
            g = torch.autograd.grad(
                outputs=emb[:, d].sum(), inputs=coord_t,
                retain_graph=(d < D - 1), create_graph=False
            )[0]  # (N, 2)
        else:
            g = torch.autograd.grad(
                outputs=emb[:, d].sum(), inputs=coord_t,
                retain_graph=True, create_graph=False
            )[0]
        grad_sq += (g[:, 0] ** 2 + g[:, 1] ** 2)

    gradient = torch.sqrt(grad_sq)
    return gradient.detach().cpu().numpy()


# ============================================================
# 全局记录
# ============================================================
exp_name = "SUICA_gradient"
all_summaries = []
total_start = time.time()

for slice_id in slices_to_run:
    print(f"\n{'='*70}")
    print(f"SUICA: Slice {slice_id}")
    print(f"{'='*70}")

    save_dir = os.path.join(RESULTS_DIR, exp_name, slice_id)
    os.makedirs(save_dir, exist_ok=True)
    n_clusters = get_n_clusters(slice_id)

    # ---- 准备 SUICA 数据 ----
    print(f"Loading Slice {slice_id}...")
    adata = load_dlpfc_slice(slice_id, data_root=GASTON_DATA_DIR)
    adata.var_names_make_unique()

    # 保存原始坐标 (用于后续可视化)
    coords_raw = np.array(adata.obsm['spatial'], dtype=np.float32)

    # 写入 SUICA 期望的 preprocessed_data 目录
    preproc_dir = os.path.join(suica_dir, "data", "preprocessed_data")
    os.makedirs(preproc_dir, exist_ok=True)
    h5ad_path = os.path.join(preproc_dir, f"{slice_id}.h5ad")
    adata.write_h5ad(h5ad_path)

    # 清理旧 log
    for log_sub in [f"logs/GAE/{slice_id}", f"logs/GAE+FFN/{slice_id}"]:
        d = os.path.join(suica_dir, log_sub)
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  Cleaned old logs: {d}")

    # ---- 生成 YAML 配置 (原始设置) ----
    print("Generating YAML configs...")
    with open(os.path.join(suica_dir, "configs", "ST", "embedder_gae.yaml"), 'r') as f:
        gae_cfg = yaml.safe_load(f)
    with open(os.path.join(suica_dir, "configs", "ST", "inr_embd.yaml"), 'r') as f:
        inr_cfg = yaml.safe_load(f)

    gae_cfg['case'] = slice_id
    gae_cfg['pipeline']['optimization']['seed'] = args.seed
    inr_cfg['case'] = slice_id
    inr_cfg['pipeline']['optimization']['seed'] = args.seed

    # 强制 num_workers=0 (Windows 兼容)
    def set_num_workers_zero(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'num_workers':
                    d[k] = 0
                elif isinstance(v, dict):
                    set_num_workers_zero(v)
    set_num_workers_zero(gae_cfg)
    set_num_workers_zero(inr_cfg)

    gae_yaml = os.path.join(suica_dir, "configs", "ST", f"embedder_gae_{slice_id}.yaml")
    inr_yaml = os.path.join(suica_dir, "configs", "ST", f"inr_embd_{slice_id}.yaml")
    with open(gae_yaml, 'w') as f: yaml.safe_dump(gae_cfg, f)
    with open(inr_yaml, 'w') as f: yaml.safe_dump(inr_cfg, f)

    # ---- Stage 1: GAE 训练 ----
    print(f"\n=== Stage 1: GAE (200 epochs, lr=1e-5) ===")
    slice_start = time.time()

    r1 = subprocess.run(
        [sys.executable, "train.py", "--mode", "embedder", "--conf", gae_yaml],
        cwd=suica_dir,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        capture_output=True, encoding='utf-8', errors='replace', timeout=3600
    )
    print(f"  GAE exit code: {r1.returncode}")
    if r1.returncode != 0:
        lines = r1.stderr.split('\n')[-15:] if r1.stderr else []
        for l in lines:
            print(f"  [STDERR] {l}")

    # ---- Stage 2: INR 训练 ----
    print(f"\n=== Stage 2: INR (2000 epochs, lr=1e-4, SIREN) ===")

    # 检查 GAE checkpoint 是否存在 (INR 需要加载 GAE decoder)
    gae_ckpt = os.path.join(suica_dir,
        f"logs/GAE/{slice_id}/lightning_logs/version_0/checkpoints/last.ckpt")
    if not os.path.exists(gae_ckpt):
        print(f"  WARNING: GAE checkpoint not found at {gae_ckpt}")
        print(f"  Disabling decoder in INR config")
        if 'decoder' in inr_cfg.get('pipeline', {}).get('inr', {}):
            inr_cfg['pipeline']['inr']['decoder'] = None
            inr_cfg['pipeline']['inr']['phase'] = None
            with open(inr_yaml, 'w') as f: yaml.safe_dump(inr_cfg, f)

    r2 = subprocess.run(
        [sys.executable, "train.py", "--mode", "inr", "--conf", inr_yaml],
        cwd=suica_dir,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        capture_output=True, encoding='utf-8', errors='replace', timeout=7200
    )
    print(f"  INR exit code: {r2.returncode}")
    if r2.returncode != 0:
        lines = r2.stderr.split('\n')[-10:] if r2.stderr else []
        for l in lines:
            print(f"  [STDERR] {l}")

    runtime = time.time() - slice_start
    print(f"  Training completed in {runtime:.2f}s ({runtime/60:.1f} min)")

    # ---- 计算空间梯度 ----
    print("Computing spatial gradients (per-dim Jacobian → Frobenius norm)...")

    # 查找 INR checkpoint
    inr_ckpt_dir = os.path.join(suica_dir, f"logs/GAE+FFN/{slice_id}",
                                 "lightning_logs", "version_0", "checkpoints")
    ckpt_files = glob.glob(os.path.join(inr_ckpt_dir, "*.ckpt"))
    ckpt_path = sorted(ckpt_files)[-1] if ckpt_files else None

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            siren = load_siren_from_checkpoint(ckpt_path, device=device)
            coords_norm = suica_normalize_coords(coords_raw, keep_ratio=True)
            suica_gradient = compute_siren_gradient(siren, coords_norm, device=device)
            grad_method = "siren_analytical"
            del siren
            torch.cuda.empty_cache()
            print(f"  SIREN analytical gradient OK: mean={suica_gradient.mean():.4f}")
        except Exception as e:
            print(f"  SIREN gradient failed: {e}")
            import traceback; traceback.print_exc()
            ckpt_path = None

    if ckpt_path is None or 'suica_gradient' not in dir():
        print("  Falling back to KNN numerical gradient on GAE embeddings...")
        # 加载 GAE embedding
        gae_embd_path = os.path.join(suica_dir, f"logs/GAE/{slice_id}",
                                      "lightning_logs", "version_0", "embedded-all.h5ad")
        if os.path.exists(gae_embd_path):
            adata_gae = sc.read_h5ad(gae_embd_path)
            if 'embeddings' in adata_gae.obsm:
                gae_emb = adata_gae.obsm['embeddings']
            else:
                sc.pp.pca(adata_gae, n_comps=32)
                gae_emb = adata_gae.obsm['X_pca']
        else:
            sc.pp.pca(adata, n_comps=32)
            gae_emb = adata.obsm['X_pca']

        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=10).fit(coords_raw)
        distances, indices = nbrs.kneighbors(coords_raw)
        suica_gradient = np.zeros(len(coords_raw))
        for i in range(len(coords_raw)):
            nei = indices[i, 1:]
            d_emb = gae_emb[nei] - gae_emb[i]
            d_dist = distances[i, 1:][:, np.newaxis] + 1e-8
            suica_gradient[i] = np.linalg.norm((d_emb / d_dist).mean(axis=0))
        grad_method = "knn_fallback"
        print(f"  KNN gradient OK: mean={suica_gradient.mean():.4f}")

    # ---- 加载 embedding 用于聚类可视化 ----
    gae_embd_path = os.path.join(suica_dir, f"logs/GAE/{slice_id}",
                                  "lightning_logs", "version_0", "embedded-all.h5ad")
    if os.path.exists(gae_embd_path):
        adata_gae = sc.read_h5ad(gae_embd_path)
        if 'embeddings' in adata_gae.obsm:
            suica_embed = adata_gae.obsm['embeddings']
        else:
            sc.pp.pca(adata, n_comps=32)
            suica_embed = adata.obsm['X_pca']
    else:
        sc.pp.pca(adata, n_comps=32)
        suica_embed = adata.obsm['X_pca']

    # ---- 构建输出 AnnData ----
    adata_out = adata.copy()
    adata_out.obsm['suica_embed'] = suica_embed
    adata_out.obs['Spatial_Gradient'] = suica_gradient

    # GMM 聚类
    gm = GaussianMixture(n_components=n_clusters, covariance_type='tied',
                         reg_covar=1e-4, random_state=args.seed)
    suica_labels = gm.fit_predict(suica_embed)
    adata_out.obs['SUICA_Domain'] = pd.Categorical(suica_labels.astype(str))

    gt = load_ground_truth(slice_id)
    if gt is not None:
        adata_out.obs['Ground Truth'] = gt.reindex(adata_out.obs_names).astype(str).values

    # ---- 保存 ----
    pd.DataFrame({
        'spot': adata_out.obs_names,
        'gradient_magnitude': suica_gradient,
    }).to_csv(os.path.join(save_dir, "spatial_gradient_values.csv"), index=False)

    # ---- 可视化 ----
    plot_spatial_gradient(
        adata_out, gradient_col='Spatial_Gradient',
        title=f'SUICA Spatial Gradient (Slice {slice_id})',
        save_path=os.path.join(save_dir, "gradient_boundary.pdf")
    )

    if 'Ground Truth' in adata_out.obs.columns:
        plot_gradient_vs_ground_truth(
            adata_out, gradient_col='Spatial_Gradient',
            gt_col='Ground Truth', title='SUICA',
            save_path=os.path.join(save_dir, "gradient_vs_gt.pdf")
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(suica_gradient, bins=50, color='seagreen', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Spatial Gradient Magnitude', fontfamily='Times New Roman', fontsize=13)
    ax.set_ylabel('Frequency', fontfamily='Times New Roman', fontsize=13)
    ax.set_title(f'SUICA Gradient Distribution - Slice {slice_id}',
                 fontfamily='Times New Roman', fontsize=14)
    fig.savefig(os.path.join(save_dir, "gradient_histogram.pdf"), bbox_inches='tight')
    plt.close(fig)

    # ---- Summary ----
    summary = {
        'Model': 'SUICA',
        'Slice': slice_id,
        'Spots': adata.shape[0],
        'Genes': adata.shape[1],
        'Gradient_Method': grad_method,
        'Architecture': 'GAE([2048,512,64],32) + SirenNet(2,2048,32,3)',
        'GAE_epochs': 200, 'GAE_lr': 1e-5,
        'INR_epochs': 2000, 'INR_lr': 1e-4,
        'Runtime(s)': round(runtime, 2),
        'Gradient_Mean': float(np.mean(suica_gradient)),
        'Gradient_Std': float(np.std(suica_gradient)),
        'Gradient_Min': float(np.min(suica_gradient)),
        'Gradient_Max': float(np.max(suica_gradient)),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)
    all_summaries.append(summary)

    print(f"  Gradient: mean={summary['Gradient_Mean']:.4f}, std={summary['Gradient_Std']:.4f}")

    # ---- 清理 ----
    for f in [gae_yaml, inr_yaml, h5ad_path]:
        try: os.remove(f)
        except: pass
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 总汇总
# ============================================================
total_runtime = time.time() - total_start
print(f"\n{'='*70}")
print(f"SUICA All Slices Complete!")
print(f"Total runtime: {total_runtime:.2f}s ({total_runtime/60:.1f} min)")
print(f"Slices processed: {len(slices_to_run)}")
print(f"{'='*70}")

df_summary = pd.DataFrame(all_summaries)
summary_path = os.path.join(RESULTS_DIR, exp_name, "all_summaries.csv")
df_summary.to_csv(summary_path, index=False)
print(f"Summary: {summary_path}")
print(df_summary[['Slice', 'Spots', 'Gradient_Mean', 'Gradient_Std', 'Runtime(s)']].to_string(index=False))
