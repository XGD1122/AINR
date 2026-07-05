"""
STINR 分组批量梯度实验 — 每组 4 切片联合训练一次，分别输出各切片梯度

数据组织:
  07-10h5ad/  → slices 151507-151510 (7 layers)
  69-72h5ad/  → slices 151669-151672 (5 layers)
  73-76h5ad/  → slices 151673-151676 (7 layers)

用法:
  python stinr/run_group.py --group 73-76      # 跑一组 (4 slices)
  python stinr/run_group.py --group all         # 跑全部 (12 slices, 3 组)
"""
import os, sys, time, argparse, warnings
import numpy as np, pandas as pd
import torch
import scanpy as sc, anndata
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'stinr'))

from config import (RESULTS_DIR, get_n_clusters, PROJECT_ROOT)
from data_utils import load_ground_truth
from visualization import (plot_spatial_gradient, plot_gradient_vs_ground_truth)
from STINR.networks import DeconvNet

# ============================================================
# 分组配置
# ============================================================
GROUP_CONFIG = {
    '07-10': {
        'data_dir': '07-10h5ad',
        'slices': ['151507', '151508', '151509', '151510'],
    },
    '69-72': {
        'data_dir': '69-72h5ad',
        'slices': ['151669', '151670', '151671', '151672'],
    },
    '73-76': {
        'data_dir': '73-76h5ad',
        'slices': ['151673', '151674', '151675', '151676'],
    },
}

ALL_GROUPS = ['07-10', '69-72', '73-76']

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description='STINR Group Gradient Experiment')
parser.add_argument('--group', type=str, default='73-76',
                    choices=['07-10', '69-72', '73-76', 'all'],
                    help='Slice group to run, or "all" for all 12 slices')
parser.add_argument('--stinr_base', type=str,
                    default='D:/STINR-123/STINR-123')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--slice_emb_dim', type=int, default=16)
parser.add_argument('--eval_interval', type=int, default=500)
args = parser.parse_args()

TRAINING_STEPS = 14001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

groups_to_run = ALL_GROUPS if args.group == 'all' else [args.group]

# ============================================================
# 对每组分别训练 + 评估
# ============================================================
for group_name in groups_to_run:
    cfg = GROUP_CONFIG[group_name]
    data_dir = os.path.join(args.stinr_base, cfg['data_dir'])
    slices = cfg['slices']

    print(f"\n{'='*70}")
    print(f"STINR Group: {group_name} ({cfg['data_dir']})")
    print(f"Slices: {slices}")
    print(f"{'='*70}")

    # ---- 加载数据 ----
    print(f"\n--- Loading data from {data_dir} ---")
    adata_basis = sc.read_h5ad(os.path.join(data_dir, 'adata_basis_DLPFC.h5ad'))
    adata_st = sc.read_h5ad(os.path.join(data_dir, 'adata_st_DLPFC.h5ad'))
    adata_st.var_names_make_unique()

    raw_paths = [os.path.join(data_dir, f'adata_st_list_raw{i}.h5ad') for i in range(4)]
    adata_st_list_raw = [sc.read_h5ad(p) for p in raw_paths]

    n_celltypes = adata_basis.shape[0]
    n_slices = len(set(adata_st.obs["slice"].values))
    gene_dim = adata_st.shape[1]

    print(f"  Spots: {adata_st.shape[0]}, Genes: {gene_dim}")
    print(f"  Cell types: {n_celltypes}, Slices: {n_slices}")

    # ---- 构建模型 ----
    print(f"\n--- Building Full DeconvNet ---")
    hidden_dims = [gene_dim, 512, args.latent_dim]

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

    # ---- 训练 ----
    print(f"\n--- Training ({TRAINING_STEPS} steps) ---")
    start_time = time.time()
    loss_history = []

    net.train()
    for step in range(TRAINING_STEPS):
        loss, recon, denoise, Z, ind_min, ind_max = net(
            coord=coord, adj_matrix=A, node_feats=X,
            count_matrix=Y, library_size=lY, slice_label=slice_labels,
            basis=basis, step=step
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (step + 1) % args.eval_interval == 0:
            print(f"  Step {step+1}/{TRAINING_STEPS}, Loss: {loss.item():.4f}")

    runtime = time.time() - start_time
    print(f"  Training completed in {runtime:.2f}s")

    # ---- 提取 latent Z ----
    print(f"\n--- Extracting latent representations ---")
    net.eval()
    with torch.no_grad():
        Z_all, beta_all, alpha_all, gamma_all = net.evaluate(A, coord, X, slice_labels)
        latent_np = Z_all.cpu().numpy()

    # ---- 逐切片计算梯度 + 保存 ----
    for slice_id in slices:
        idx = slices.index(slice_id)
        save_dir = os.path.join(RESULTS_DIR, "STINR_gradient", slice_id)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n  --- Slice {slice_id} (idx={idx}) ---")

        # 提取该切片的 latent
        mask = (slice_labels == idx).cpu().numpy()
        print(f"    Spots: {mask.sum()}")

        # 梯度计算
        target_coord = coord[mask].detach().clone().requires_grad_(True)
        net.coord = target_coord / 100.0
        mid_fea = net.encoder_layer0(net.coord)
        Z_target = net.encoder_layer1(mid_fea)

        grads = torch.autograd.grad(
            outputs=Z_target, inputs=target_coord,
            grad_outputs=torch.ones_like(Z_target),
            create_graph=False
        )[0]
        stinr_gradient = torch.norm(grads, p=2, dim=1).detach().cpu().numpy()

        # 构建 AnnData
        adata_raw = adata_st_list_raw[idx].copy()
        adata_raw.var_names_make_unique()

        adata_out = anndata.AnnData(
            np.array(adata_raw.X.toarray() if hasattr(adata_raw.X, 'toarray') else adata_raw.X)
        )
        adata_out.obs_names = adata_raw.obs_names
        adata_out.uns['spatial'] = adata_raw.uns['spatial']
        adata_out.obsm['spatial'] = np.array(adata_raw.obsm['spatial'])
        adata_out.obsm['latent'] = latent_np[mask]
        adata_out.obs['Spatial_Gradient'] = stinr_gradient

        # GMM clustering
        n_clusters = get_n_clusters(slice_id)
        gm = GaussianMixture(n_components=n_clusters, covariance_type='tied',
                             reg_covar=1e-4, init_params='kmeans', random_state=42)
        stinr_labels = gm.fit_predict(latent_np[mask])
        adata_out.obs['STINR_Domain'] = pd.Categorical(stinr_labels.astype(str))

        # Ground Truth
        gt = load_ground_truth(slice_id)
        if gt is not None:
            adata_out.obs['Ground Truth'] = gt.reindex(adata_out.obs_names).astype(str).values

        # 保存梯度值
        gradient_df = pd.DataFrame({
            'spot': adata_out.obs_names,
            'gradient_magnitude': stinr_gradient,
        })
        gradient_df.to_csv(os.path.join(save_dir, "spatial_gradient_values.csv"), index=False)

        # 可视化
        plot_spatial_gradient(
            adata_out, gradient_col='Spatial_Gradient',
            title=f'STINR Spatial Gradient (Slice {slice_id})',
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
        ax.set_title(f'STINR Gradient Distribution - Slice {slice_id}',
                     fontfamily='Times New Roman', fontsize=14)
        fig.savefig(os.path.join(save_dir, "gradient_histogram.pdf"), bbox_inches='tight')
        plt.close(fig)

        # 保存 summary
        summary = {
            'Model': 'STINR (Full DeconvNet)',
            'Slice': slice_id,
            'Group': group_name,
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

        print(f"    Gradient: mean={summary['Gradient_Mean']:.4f}, std={summary['Gradient_Std']:.4f}")

    # 保存该组的 loss history
    pd.DataFrame({'step': range(1, TRAINING_STEPS+1), 'loss': loss_history}).to_csv(
        os.path.join(RESULTS_DIR, "STINR_gradient", f"loss_history_{group_name}.csv"), index=False
    )

    # 保存该组的模型
    torch.save(net.state_dict(),
               os.path.join(RESULTS_DIR, "STINR_gradient", f"stinr_deconvnet_{group_name}.pth"))

    # 清理显存
    del net, optimizer, X, A, Y, lY, slice_labels, basis, coord
    torch.cuda.empty_cache()

print(f"\n{'='*70}")
print("STINR Group Experiment Complete!")
print(f"Groups processed: {groups_to_run}")
print(f"Results saved to: {os.path.join(RESULTS_DIR, 'STINR_gradient')}")
print(f"{'='*70}")
