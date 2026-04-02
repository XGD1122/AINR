import pandas as pd, numpy as np, scanpy as sc, anndata as ad, os, torch, random, matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from model import Model
import argparse
import time
import warnings

warnings.filterwarnings("ignore")


# 对齐函数
def align_labels_by_max_overlap(y_pred, y_true):
    y_pred = y_pred.astype(str)
    y_true = y_true.astype(str)
    y_pred_unique = pd.Series(y_pred).unique()
    y_true_unique = pd.Series(y_true).unique()
    y_pred_unique = y_pred_unique[[str(l).lower() != 'nan' for l in y_pred_unique]]
    y_true_unique = y_true_unique[[str(l).lower() != 'nan' for l in y_true_unique]]
    if len(y_pred_unique) == 0 or len(y_true_unique) == 0: return {}
    cm = confusion_matrix(y_true, y_pred, labels=y_true_unique, sample_weight=None)
    mapping = {}
    used_pred_indices = set()
    for i in range(len(y_true_unique)):
        best_match_idx = -1
        max_overlap = -1
        for j in range(len(y_pred_unique)):
            if j not in used_pred_indices and cm[i, j] > max_overlap:
                max_overlap = cm[i, j]
                best_match_idx = j
        if best_match_idx != -1:
            mapping[y_pred_unique[best_match_idx]] = y_true_unique[i]
            used_pred_indices.add(best_match_idx)
    for label in y_pred_unique:
        if label not in mapping: mapping[label] = label
    return mapping


# === 参数解析 ===
parser = argparse.ArgumentParser(description='AINR Clustering Training')

# 基础参数
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--seed', type=int, default=112)
parser.add_argument('--training_steps', type=int, default=10000)
parser.add_argument('--patience', type=int, default=1500, help='Early stopping steps')

# 网络结构参数 (由脚本控制输入)
parser.add_argument('--inr_width', type=int, default=128)
parser.add_argument('--inr_depth', type=int, default=3)
parser.add_argument('--recon_weight', type=float, default=1)
parser.add_argument('--nhead', type=int, default=8)

# 业务固定参数
parser.add_argument('--n_clusters', type=int, default=7)
parser.add_argument('--slice_idx', type=int, nargs='+', default=[151673, 151674, 151675, 151676])
parser.add_argument('--slice_count', type=int, default=4)

args = parser.parse_args()


# === 动态保存路径 ===
base_save_dir = "./results_"
# 文件夹名包含 W(宽度) 和 D(深度)
exp_tag = f"LR{args.lr}_WD{args.weight_decay}_BS{args.batch_size}_HD{args.hidden_dim}_W{args.inr_width}_D{args.inr_depth}_RW{args.recon_weight}"
save_path = os.path.join(base_save_dir, exp_tag)
os.makedirs(save_path, exist_ok=True)

log_file = os.path.join(base_save_dir, "summary_report.csv")

# 设置随机种子
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# === 训练 ===
model_instance = Model(
    slice_count=args.slice_count,
    slice_idx_gt=args.slice_idx,
    hidden_dims=[None, args.hidden_dim],
    training_steps=args.training_steps,
    lr=args.lr,
    seed=args.seed,
    n_clusters=args.n_clusters,
    batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    recon_weight=args.recon_weight,
    save_path=save_path,
    nhead=args.nhead,
    # 实现对称结构
    inr_width=args.inr_width,
    inr_depth=args.inr_depth,
    dec_width=args.inr_width,
    dec_depth=args.inr_depth
)

# === 启动训练 ===
final_info = model_instance.train(report_loss=True, eval_interval=100, patience_steps=args.patience)
best_avg_ari = final_info['avg_ari']

# === 记录结果到表格 ===
log_data = {
    "Time": [time.strftime("%Y-%m-%d %H:%M:%S")],
    "Tag": [exp_tag],
    "LR": [args.lr],
    "WD": [args.weight_decay],
    "BS": [args.batch_size],
    "HD": [args.hidden_dim],
    "Width": [args.inr_width],
    "Depth": [args.inr_depth],
    "RW": [args.recon_weight],
    "Best_ARI": [best_avg_ari]
}
df_log = pd.DataFrame(log_data)
if not os.path.exists(log_file):
    df_log.to_csv(log_file, index=False)
else:
    df_log.to_csv(log_file, mode='a', header=False, index=False)

# === 视觉风格锁定 ===
print(f"\n=== Visualizing Best Result (ARI: {best_avg_ari:.4f}) ===")
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False


def finalize_plot(ax, title=""):
    ax.set_title(title, fontdict={'family': 'Times New Roman', 'size': 16})
    ax.set_axis_off()


# 数据准备
adata_st = final_info['adata_st']
# 【关键修复 1】确保主数据的基因名唯一
adata_st.var_names_make_unique()

adata_st_list_raw = final_info['adata_st_list_raw']
adata_st.obs["GM_Best"] = pd.Series([str(z) for z in final_info['gmm_labels']], index=adata_st.obs_names)
adata_st.layers['denoised'] = final_info['recon']

# === 逐切片生成核心结果图 ===
for i, sec_raw in enumerate(adata_st_list_raw):
    sid = str(args.slice_idx[i])
    slice_save_path = os.path.join(save_path, sid)
    os.makedirs(slice_save_path, exist_ok=True)

    sec = sec_raw.copy()
    lib_id = list(sec.uns['spatial'].keys())[0]

    # 提取当前切片的预测结果
    sec.obs["GM"] = adata_st.obs.loc[sec.obs_names, "GM_Best"].values

    # 只有在存在 Ground Truth 时才进行标签对齐和 ARI 计算
    try:
        gt_path = os.path.join("Data", "DLPFC_annotations", f'{sid}_truth.txt')
        if os.path.exists(gt_path):
            gt = pd.read_csv(gt_path, sep='\t', header=None, index_col=0)
            idx = sec.obs_names.tolist()
            clean = [n.rsplit('-', 1)[0] if '-slice' in n else n for n in idx]
            gt_mapped = pd.Series(clean).map(gt.iloc[:, 0])
            gt_mapped.index = idx
            sec.obs['Ground Truth'] = gt_mapped.fillna('nan').astype(str).values

            df_eval = sec.obs[sec.obs['Ground Truth'] != 'nan']
            ari_val = adjusted_rand_score(df_eval['GM'], df_eval['Ground Truth'])
            mapping = align_labels_by_max_overlap(df_eval['GM'], df_eval['Ground Truth'])
            sec.obs["Cluster"] = sec.obs['GM'].map(mapping).fillna(sec.obs['GM']).astype(str)
            title_str = f"ARI: {ari_val:.4f}"
        else:
            sec.obs["Cluster"] = sec.obs["GM"]
            title_str = "Prediction"
    except:
        sec.obs["Cluster"] = sec.obs["GM"]
        title_str = "Prediction"

    # 生成唯一的预测空间图
    print(f"Slice {sid}: Saving spatial clustering plot...")
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.spatial(sec, color='Cluster', ax=ax, show=False, title="",
                  legend_loc=None, library_id=lib_id)
    finalize_plot(ax, title=title_str)

    fig.savefig(os.path.join(slice_save_path, "pred_spatial.pdf"), bbox_inches='tight')
    plt.close(fig)

print(f"Success:  results saved in {save_path}")