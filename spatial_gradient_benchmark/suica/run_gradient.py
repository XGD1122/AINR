"""
SUICA 空间梯度实验

两阶段训练 (D:\SEDR\SUICA 已打补丁版本):
  Stage 1: GAE — Graph Autoencoder (200 epochs, 139M params)
  Stage 2: INR — SIREN 拟合 GAE 嵌入 (2000 epochs, 78M params)

梯度 = d(GAE_embedding)/d(spatial_coords)

用法:
    python run_gradient.py --slice 151673
"""
import os, sys, time, argparse, warnings, shutil, yaml, subprocess, glob
import numpy as np, pandas as pd
import torch, scanpy as sc, anndata
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (RESULTS_DIR, get_n_clusters, GASTON_DATA_DIR, PROJECT_ROOT)
from data_utils import load_dlpfc_slice, load_ground_truth
from visualization import (plot_spatial_gradient, plot_gradient_vs_ground_truth)

# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--slice', type=str, default='151673')
parser.add_argument('--suica_dir', type=str, default=None)
parser.add_argument('--test', action='store_true')
parser.add_argument('--seed', type=int, default=8848)
args = parser.parse_args()

# SUICA 目录
if args.suica_dir:
    suica_dir = os.path.abspath(args.suica_dir)
else:
    suica_dir = os.path.join(PROJECT_ROOT, "SUICA")  # 已打补丁的版本
assert os.path.isdir(suica_dir), f"SUICA not found: {suica_dir}"

exp_name = "SUICA_gradient"
save_dir = os.path.join(RESULTS_DIR, exp_name, args.slice)
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, SUICA: {suica_dir}")

gae_epochs  = 5  if args.test else 200
inr_epochs  = 10 if args.test else 2000
inr_phase   = 5  if args.test else 1000

# ============================================================
# 1. 准备数据
# ============================================================
print(f"\n=== Loading Slice {args.slice} ===")
adata = load_dlpfc_slice(args.slice, data_root=GASTON_DATA_DIR)
adata.var_names_make_unique()

preproc_dir = os.path.join(suica_dir, "data", "preprocessed_data")
os.makedirs(preproc_dir, exist_ok=True)
h5ad_path = os.path.join(preproc_dir, f"{args.slice}.h5ad")
adata.write_h5ad(h5ad_path)

# 清理旧 log (GAE 和 INR 两个目录)
for log_dir in [f"logs/GAE/{args.slice}", f"logs/GAE+FFN/{args.slice}"]:
    d = os.path.join(suica_dir, log_dir)
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  Cleaned: {d}")

# ============================================================
# 2. YAML 配置
# ============================================================
print("Preparing YAML configs...")
with open(os.path.join(suica_dir, "configs", "ST", "embedder_gae.yaml"), 'r') as f:
    gae_cfg = yaml.safe_load(f)
with open(os.path.join(suica_dir, "configs", "ST", "inr_embd.yaml"), 'r') as f:
    inr_cfg = yaml.safe_load(f)

gae_cfg['case'] = args.slice
inr_cfg['case'] = args.slice
gae_cfg['pipeline']['optimization']['epochs'] = gae_epochs
inr_cfg['pipeline']['inr']['phase'] = inr_phase
inr_cfg['pipeline']['optimization']['epochs'] = inr_epochs
# val_freq 保持 YAML 默认值 200，不覆盖

# 强制 num_workers=0 (Windows)
def _set_num_workers(cfg):
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == 'num_workers':
                cfg[k] = 0
            elif isinstance(v, dict):
                _set_num_workers(v)

gae_yaml = os.path.join(suica_dir, "configs", "ST", f"embedder_gae_{args.slice}.yaml")
inr_yaml = os.path.join(suica_dir, "configs", "ST", f"inr_embd_{args.slice}.yaml")
with open(gae_yaml, 'w') as f: yaml.safe_dump(gae_cfg, f)
with open(inr_yaml, 'w') as f: yaml.safe_dump(inr_cfg, f)

# ============================================================
# 3. 两阶段训练
# ============================================================
start_time = time.time()
original_cwd = os.getcwd()

try:
    os.chdir(suica_dir)

    print(f"\n=== Stage 1: GAE ({gae_epochs} epochs) ===")
    r1 = subprocess.run(
        [sys.executable, "train.py", "--mode", "embedder", "--conf", gae_yaml],
        cwd=suica_dir, env={**os.environ, "PYTHONUTF8": "1"}
    )
    print(f"  GAE exit code: {r1.returncode}")

    print(f"\n=== Stage 2: INR ({inr_epochs} epochs) ===")
    gae_ckpt = os.path.join(suica_dir,
        f"logs/GAE/{args.slice}/lightning_logs/version_0/checkpoints/last.ckpt")
    if 'decoder' in inr_cfg.get('pipeline', {}).get('inr', {}) and not os.path.exists(gae_ckpt):
        print("  GAE checkpoint not found, disabling decoder+phase")
        del inr_cfg['pipeline']['inr']['decoder']
        inr_cfg['pipeline']['inr']['phase'] = None
        with open(inr_yaml, 'w') as f: yaml.safe_dump(inr_cfg, f)

    r2 = subprocess.run(
        [sys.executable, "train.py", "--mode", "inr", "--conf", inr_yaml],
        cwd=suica_dir, env={**os.environ, "PYTHONUTF8": "1"}
    )
    print(f"  INR exit code: {r2.returncode}")

except Exception as e:
    print(f"  ERROR: {e}")
finally:
    os.chdir(original_cwd)

runtime = time.time() - start_time

# ============================================================
# 4. 加载 GAE embedding
# ============================================================
print("\n=== Loading SUICA Results ===")

gae_embd = os.path.join(suica_dir, "logs", "GAE", args.slice,
                        "lightning_logs", "version_0", "embedded-all.h5ad")

suica_embed = None
embed_source = "unknown"

if os.path.exists(gae_embd):
    adata_gae = sc.read_h5ad(gae_embd)
    if 'embeddings' in adata_gae.obsm:
        suica_embed = adata_gae.obsm['embeddings'].copy()
        embed_source = "gae_embedding"
        print(f"  Loaded GAE embedding: {suica_embed.shape}")
else:
    print(f"  GAE embedding not found at {gae_embd}")

if suica_embed is None:
    print("  Falling back to PCA on raw data")
    sc.tl.pca(adata, n_comps=32)
    suica_embed = adata.obsm['X_pca'].copy()
    embed_source = "pca_fallback"

print(f"  Embedding: {embed_source}, shape={suica_embed.shape}")

# ============================================================
# 5. 计算空间梯度
# ============================================================
print("Computing spatial gradient...")

coords_raw = np.array(adata.obsm['spatial'], dtype=np.float32)
analytical = False
suica_gradient = None

# 尝试从 INR checkpoint 解析梯度
inr_log_dir = inr_cfg['pipeline']['optimization']['logs']
inr_ver_dirs = glob.glob(os.path.join(suica_dir, inr_log_dir, "lightning_logs", "version_*"))
if inr_ver_dirs:
    latest_ver = sorted(inr_ver_dirs, key=lambda d: int(d.rsplit('_',1)[-1]))[-1]
    ckpt_files = glob.glob(os.path.join(latest_ver, "checkpoints", "*.ckpt"))
    if ckpt_files:
        ckpt_path = sorted(ckpt_files)[-1]
        print(f"  Trying analytical gradient from INR checkpoint...")
        try:
            sys.path.insert(0, suica_dir)
            from systems.inr_fitting_system import INRFittingSystem
            pl_model = INRFittingSystem.load_from_checkpoint(ckpt_path, map_location=device)
            pl_model.eval(); pl_model.to(device)

            # SUICA 坐标归一化 (datasets.py _normalize_coordinates)
            c_min = coords_raw.min(axis=0); c_max = coords_raw.max(axis=0)
            c_range = c_max - c_min
            coords_norm = (coords_raw - c_min) / c_range
            coords_norm -= 0.5; coords_norm *= 2.0
            # keep_ratio
            max_range = max(c_range)
            scale = c_range / max_range
            coords_norm[:,0] *= scale[0]; coords_norm[:,1] *= scale[1]

            coord_t = torch.FloatTensor(coords_norm).to(device).requires_grad_(True)
            with torch.enable_grad():
                emb = pl_model.inr(coord_t)
                grads = torch.autograd.grad(outputs=emb.sum(), inputs=coord_t,
                                           create_graph=False)[0]
                suica_gradient = torch.norm(grads, p=2, dim=1).detach().cpu().numpy()
            analytical = True
            print(f"  Analytical gradient OK")
        except Exception as e:
            print(f"  Analytical failed ({e}), using KNN")

# KNN fallback
if suica_gradient is None:
    print("  Using KNN numerical gradient...")
    emb = suica_embed if isinstance(suica_embed, np.ndarray) else np.array(suica_embed)
    nbrs = NearestNeighbors(n_neighbors=10).fit(coords_raw)
    distances, indices = nbrs.kneighbors(coords_raw)
    suica_gradient = np.zeros(len(coords_raw))
    for i in range(len(coords_raw)):
        nei = indices[i, 1:]
        d_emb = emb[nei] - emb[i]
        d_dist = distances[i, 1:][:, np.newaxis] + 1e-8
        suica_gradient[i] = np.linalg.norm((d_emb / d_dist).mean(axis=0))

# ============================================================
# 6. 输出
# ============================================================
adata_out = adata.copy()
adata_out.obsm['suica_embed'] = suica_embed
adata_out.obs['Spatial_Gradient'] = suica_gradient

n_clusters = get_n_clusters(args.slice)
gm = GaussianMixture(n_components=n_clusters, covariance_type='tied',
                     reg_covar=1e-4, random_state=args.seed)
suica_labels = gm.fit_predict(suica_embed)
adata_out.obs['SUICA_Domain'] = pd.Categorical(suica_labels.astype(str))

gt = load_ground_truth(args.slice)
if gt is not None:
    adata_out.obs['Ground Truth'] = gt.reindex(adata_out.obs_names).astype(str).values

# ============================================================
# 7. 保存
# ============================================================
pd.DataFrame({'spot': adata_out.obs_names, 'gradient_magnitude': suica_gradient}
    ).to_csv(os.path.join(save_dir, "spatial_gradient_values.csv"), index=False)

# ============================================================
# 8. 可视化
# ============================================================
print("\n=== Visualizations ===")
plot_spatial_gradient(adata_out, gradient_col='Spatial_Gradient',
    title=f'SUICA Spatial Gradient (Slice {args.slice})',
    save_path=os.path.join(save_dir, "gradient_boundary.pdf"))

if 'Ground Truth' in adata_out.obs.columns:
    plot_gradient_vs_ground_truth(adata_out, gradient_col='Spatial_Gradient',
        gt_col='Ground Truth', title='SUICA',
        save_path=os.path.join(save_dir, "gradient_vs_gt.pdf"))

fig, ax = plt.subplots(figsize=(8,4))
ax.hist(suica_gradient, bins=50, color='seagreen', edgecolor='white', alpha=0.8)
ax.set_xlabel('Spatial Gradient Magnitude'); ax.set_ylabel('Frequency')
fig.savefig(os.path.join(save_dir, "gradient_histogram.pdf"), bbox_inches='tight')
plt.close(fig)

pd.DataFrame([{
    'Model': 'SUICA', 'Slice': args.slice,
    'Embed_source': embed_source,
    'Gradient_method': 'analytical' if analytical else 'KNN',
    'Runtime(s)': round(runtime, 2),
    'Gradient_Mean': float(np.mean(suica_gradient)),
    'Gradient_Std': float(np.std(suica_gradient)),
}]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)

# 清理临时文件
for f in [gae_yaml, inr_yaml, h5ad_path]:
    try: os.remove(f)
    except: pass

print(f"\n=== SUICA Complete ===")
print(f"Results: {save_dir}")
print(f"Gradient: mean={np.mean(suica_gradient):.4f}, std={np.std(suica_gradient):.4f}")
