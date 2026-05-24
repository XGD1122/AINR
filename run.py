import pandas as pd, numpy as np, scanpy as sc, anndata as ad, os, torch, random, matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from model import Model
import argparse
import time
import tracemalloc  
import warnings

warnings.filterwarnings("ignore")


def align_labels_by_max_overlap(y_pred, y_true):
    y_pred, y_true = y_pred.astype(str), y_true.astype(str)
    y_pred_unique = pd.Series(y_pred).unique()
    y_true_unique = pd.Series(y_true).unique()
    y_pred_unique = y_pred_unique[[str(l).lower() != 'nan' for l in y_pred_unique]]
    y_true_unique = y_true_unique[[str(l).lower() != 'nan' for l in y_true_unique]]
    if len(y_pred_unique) == 0 or len(y_true_unique) == 0: return {}
    cm = confusion_matrix(y_true, y_pred, labels=y_true_unique, sample_weight=None)
    mapping, used_pred_indices = {}, set()
    for i in range(len(y_true_unique)):
        best_match_idx, max_overlap = -1, -1
        for j in range(len(y_pred_unique)):
            if j not in used_pred_indices and cm[i, j] > max_overlap:
                max_overlap, best_match_idx = cm[i, j], j
        if best_match_idx != -1:
            mapping[y_pred_unique[best_match_idx]] = y_true_unique[i]
            used_pred_indices.add(best_match_idx)
    for label in y_pred_unique:
        if label not in mapping: mapping[label] = label
    return mapping


parser = argparse.ArgumentParser(description='INR-Recon Clustering with Early Stopping')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--seed', type=int, default=112)
parser.add_argument('--training_steps', type=int, default=10000)
parser.add_argument('--patience', type=int, default=1500)
parser.add_argument('--inr_width', type=int, default=160)
parser.add_argument('--inr_depth', type=int, default=3)
parser.add_argument('--recon_weight', type=float, default=0.5)
parser.add_argument('--tv_weight', type=float, default=1e-05)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--omega_0', type=float, default=20.0)
parser.add_argument('--n_clusters', type=int, default=5)
parser.add_argument('--slice_idx', type=int, nargs='+', default=[151673, 151674, 151675, 151676])
parser.add_argument('--slice_count', type=int, default=4)
args = parser.parse_args()

base_save_dir = "./results_sweep"
exp_tag = f"LR{args.lr}_BS{args.batch_size}_W{args.inr_width}_D{args.inr_depth}_TV{args.tv_weight}_HD{args.hidden_dim}_RW{args.recon_weight}_W0{args.omega_0}"
save_path = os.path.join(base_save_dir, exp_tag)
os.makedirs(save_path, exist_ok=True)
log_file = os.path.join(base_save_dir, "summary_report.csv")

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

model_instance = Model(
    slice_count=args.slice_count, slice_idx_gt=args.slice_idx, hidden_dims=[None, args.hidden_dim],
    training_steps=args.training_steps, lr=args.lr, seed=args.seed, n_clusters=args.n_clusters,
    batch_size=args.batch_size, weight_decay=args.weight_decay, recon_weight=args.recon_weight,
    tv_weight=args.tv_weight, save_path=save_path, nhead=args.nhead,
    inr_width=args.inr_width, inr_depth=args.inr_depth, dec_width=args.inr_width, dec_depth=args.inr_depth,omega_0=args.omega_0
)


print("\n--- Starting Resource Tracking ---")
tracemalloc.start()  
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()  
start_time = time.time()

# 执行训练
final_info = model_instance.train(report_loss=True, eval_interval=100, patience_steps=args.patience)


end_time = time.time()
runtime_seconds = end_time - start_time
current_cpu_mem, peak_cpu_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_cpu_mb = peak_cpu_mem / (1024 ** 2)
peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

print(f"\n[Computational Cost]")
print(f"-> Runtime: {runtime_seconds:.2f} seconds")
print(f"-> Peak CPU Memory: {peak_cpu_mb:.2f} MB")
print(f"-> Peak GPU VRAM: {peak_gpu_mb:.2f} MB")
# ==========================================

best_avg_ari = final_info['avg_ari']


log_data = {
    "Time": [time.strftime("%Y-%m-%d %H:%M:%S")],
    "Tag": [exp_tag],
    "LR": [args.lr],
    "BS": [args.batch_size],
    "HD": [args.hidden_dim],
    "Width": [args.inr_width],
    "Depth": [args.inr_depth],
    "TV": [args.tv_weight],
    "RW": [args.recon_weight],   
    "W0": [args.omega_0],       
    "Best_ARI": [best_avg_ari],
    "Runtime(s)": [round(runtime_seconds, 2)],
    "Peak_CPU(MB)": [round(peak_cpu_mem, 2)],
    "Peak_GPU(MB)": [round(peak_gpu_mb, 2)]
}
df_log = pd.DataFrame(log_data)
df_log.to_csv(log_file, mode='a' if os.path.exists(log_file) else 'w', header=not os.path.exists(log_file), index=False)

print(f"\n=== Visualizing Best Result (Global ARI: {best_avg_ari:.4f}) ===")
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False


def finalize_plot(ax, title=""):
    ax.set_title(title, fontdict={'family': 'Times New Roman', 'size': 16})
    ax.set_axis_off()


adata_st = final_info['adata_st']
adata_st.var_names_make_unique()
adata_st.obs["GM_Best"] = pd.Series([str(z) for z in final_info['gmm_labels']], index=adata_st.obs_names)
adata_st.obs['Spatial_Gradient'] = final_info['spatial_gradient']
adata_st.layers['denoised'] = final_info['recon']

all_gt = pd.Series(index=adata_st.obs_names, dtype='object')
for i, raw_ad in enumerate(final_info['adata_st_list_raw']):
    sid = str(args.slice_idx[i])
    try:
        gt_path = f'DLPFC_annotations/{sid}_truth.txt'
        if os.path.exists(gt_path):
            gt = pd.read_csv(gt_path, sep='\t', header=None, index_col=0)
            idx = raw_ad.obs_names.tolist()
            clean = [n.rsplit('-', 1)[0] if '-slice' in n else n for n in idx]
            mapped = pd.Series(clean).map(gt.iloc[:, 0])
            mapped.index = idx
            all_gt.update(mapped.dropna())
    except:
        pass
adata_st.obs['Ground Truth'] = all_gt.fillna('nan').astype(str).values

for i, sec_raw in enumerate(final_info['adata_st_list_raw']):
    sid = str(args.slice_idx[i])
    slice_save_path = os.path.join(save_path, sid)
    os.makedirs(slice_save_path, exist_ok=True)

    sec = sec_raw.copy()
    sec.var_names_make_unique()
    lib_id = list(sec.uns['spatial'].keys())[0]

    sec.obs["GM"] = adata_st.obs.loc[sec.obs_names, "GM_Best"].values
    sec.obs['Ground Truth'] = adata_st.obs.loc[sec.obs_names, 'Ground Truth'].values
    sec.obsm['latent'] = final_info['latent'][adata_st.obs_names.isin(sec.obs_names)]
    sec.obs['Spatial_Gradient'] = adata_st.obs.loc[sec.obs_names, 'Spatial_Gradient'].values

    df_eval = sec.obs[sec.obs['Ground Truth'] != 'nan']
    ari_val = adjusted_rand_score(df_eval['GM'], df_eval['Ground Truth']) if len(df_eval) > 0 else 0.0
    mapping = align_labels_by_max_overlap(df_eval['GM'], df_eval['Ground Truth']) if len(df_eval) > 0 else {}
    sec.obs["Cluster"] = sec.obs['GM'].map(mapping).fillna(sec.obs['GM']).astype(str)

    print(f"Slice {sid} - Local ARI: {ari_val:.4f}")


    print(f"Slice {sid}: Generating Spatial Gradient Boundary plot...")
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.spatial(sec, color='Spatial_Gradient', color_map='turbo', ax=ax, show=False,
                  title="INR Spatial Gradient", library_id=lib_id)
    finalize_plot(ax, title=f"Boundaries (Slice {sid})")
    fig.savefig(os.path.join(slice_save_path, "spatial_gradient_boundaries.pdf"), bbox_inches='tight')
    plt.close(fig)


    sec_vis = sec[(sec.obs["Cluster"] != 'nan') & (sec.obs["Ground Truth"] != 'nan')].copy()
    if len(sec_vis) > 15:
        sc.pp.neighbors(sec_vis, use_rep='latent', n_neighbors=15)
        sc.tl.umap(sec_vis)

        print(f"Slice {sid}: Generating 5 clustering plots...")
        for plot_type in ['pred_spatial', 'truth_spatial', 'pred_umap', 'pred_umap_on_data', 'truth_umap']:
            fig, ax = plt.subplots(figsize=(6, 6))
            if 'spatial' in plot_type:
                color = 'Cluster' if 'pred' in plot_type else 'Ground Truth'
                sc.pl.spatial(sec_vis, color=color, ax=ax, show=False, title="",
                              legend_loc=None if 'pred' in plot_type else 'right margin', library_id=lib_id)
                finalize_plot(ax, title=f"ARI: {ari_val:.4f}" if 'pred' in plot_type else "")
            else:
                color = 'Cluster' if 'pred' in plot_type else 'Ground Truth'
                leg = 'on data' if 'on_data' in plot_type else (None if 'pred' in plot_type else 'right margin')
                sc.pl.umap(sec_vis, color=color, ax=ax, show=False, title="", legend_loc=leg)
                finalize_plot(ax, title="")
            fig.savefig(os.path.join(slice_save_path, f"{plot_type}.pdf"), bbox_inches='tight')
            plt.close(fig)

    target_genes = ['PCP4', 'CXCL14', 'ENC1', 'CCK', 'KRT17', 'MOBP']
    existing_genes = [name for name in adata_st.var_names if any(name == tg or name.startswith(f"{tg}-") for tg in target_genes)]

    print(f"Slice {sid}: Generating gene expression plots...")
    sec_denoised_ad = ad.AnnData(X=adata_st[sec.obs_names, :].layers['denoised'].copy(), obs=sec.obs, var=adata_st.var)
    sec_denoised_ad.var_names_make_unique()
    sec_denoised_ad.uns, sec_denoised_ad.obsm['spatial'] = sec.uns, sec.obsm['spatial']

    for g in existing_genes:
        if g not in sec.var_names: continue
        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.spatial(sec, color=g, ax=ax, show=False, title=g, legend_loc=None, library_id=lib_id, color_map='magma')
        finalize_plot(ax, title=g)
        fig.savefig(os.path.join(slice_save_path, f"raw_spatial_{g}.pdf"), bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.spatial(sec_denoised_ad, color=g, ax=ax, show=False, title="", legend_loc=None, library_id=lib_id, color_map='magma')
        finalize_plot(ax, title="")
        fig.savefig(os.path.join(slice_save_path, f"denoised_spatial_{g}.pdf"), bbox_inches='tight')
        plt.close(fig)

    print(f"Slice {sid}: Generating slice-specific stacked violin plots...")
    sorted_clusters = sorted(sec.obs['Cluster'].unique())
    sec.obs['Cluster'] = pd.Categorical(sec.obs['Cluster'], categories=sorted_clusters)
    sec_denoised_ad.obs['Cluster'] = pd.Categorical(sec_denoised_ad.obs['Cluster'], categories=sorted_clusters)

    final_genes_to_plot = [g for g in existing_genes if g in sec.var_names]

    if final_genes_to_plot:
        with plt.rc_context({'font.family': 'Times New Roman'}):
            sc.pl.stacked_violin(sec, var_names=final_genes_to_plot, groupby='Cluster', swap_axes=True, cmap='Blues', title=f"Raw - Slice {sid}", show=False)
            plt.savefig(os.path.join(slice_save_path, f"stacked_violin_raw_{sid}.pdf"), bbox_inches='tight')
            plt.close()

            sc.pl.stacked_violin(sec_denoised_ad, var_names=final_genes_to_plot, groupby='Cluster', swap_axes=True, cmap='Blues', title=f"AINR - Slice {sid}", show=False)
            plt.savefig(os.path.join(slice_save_path, f"stacked_violin_AINR_{sid}.pdf"), bbox_inches='tight')
            plt.close()

print(f"Success: ARI calculated and Spatial Gradient Boundaries saved in {save_path}")
