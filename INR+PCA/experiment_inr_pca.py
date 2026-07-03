"""
INR+PCA 基线实验 — 响应审稿人3 第3条意见
===========================================
架构变化：
  AINR-FULL:  coord → INR(SIREN) → LatentProj → SAA(Transformer) → Decoder → GMM
  INR+PCA:    coord → INR(SIREN) → PCA(50) → GMM    ← 砍掉 SAA/Decoder

  即：只保留 INR 模块独立训练，用 PCA 替代后续全部组件。
"""

import os, torch, numpy as np, pandas as pd, scipy.sparse
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
from torch.utils.data import Dataset, DataLoader
import argparse, time

# ==========================================
# 1. 精简 INR-only 网络（只保留 SIREN 编码器）
# ==========================================
class DenseLayer(torch.nn.Module):
    def __init__(self, c_in, c_out, zero_init=False):
        super().__init__()
        self.linear = torch.nn.Linear(c_in, c_out)
        if zero_init:
            torch.nn.init.zeros_(self.linear.weight.data)
        else:
            torch.nn.init.uniform_(self.linear.weight.data,
                                   -np.sqrt(6 / (c_in + c_out)),
                                   np.sqrt(6 / (c_in + c_out)))
        torch.nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        return self.linear(x)


class SineLayer(torch.nn.Module):
    def __init__(self, c_in, c_out, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = c_in
        self.linear = torch.nn.Linear(c_in, c_out, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)
        torch.nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class INR_Only(torch.nn.Module):
    """
    精简 INR 网络：coord → SIREN → gene_expression
    与 AINR 的 coord_encoder_layer0 完全相同的结构，
    但不包含 LatentProj / SAA / Decoder。
    """
    def __init__(self, gene_dim, inr_width=160, inr_depth=3, omega_0=20.0, coord_dim=3):
        super().__init__()
        inr_layers = [SineLayer(coord_dim, inr_width, is_first=True, omega_0=omega_0)]
        for _ in range(inr_depth - 1):
            inr_layers.append(SineLayer(inr_width, inr_width, is_first=False, omega_0=1.0))
        inr_layers.append(DenseLayer(inr_width, gene_dim))
        self.encoder = torch.nn.Sequential(*inr_layers)

    def forward(self, coord):
        """coord: [N, coord_dim] → mid_fea: [N, gene_dim]"""
        return self.encoder(coord)


# ==========================================
# 2. 数据加载（复用原 Model 的逻辑）
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, coord, X):
        self.coord = coord
        self.X = X
    def __len__(self):
        return len(self.coord)
    def __getitem__(self, idx):
        return self.coord[idx], self.X[idx]


def load_data(slice_count, slice_idx):
    """加载并预处理数据，返回 coord, X, adata_st, adata_st_list_raw"""
    adata_st_list_raw = [ad.read_h5ad(f'adata_st_list_raw{i}.h5ad')
                         for i in range(slice_count)]
    adata_st = ad.read_h5ad('adata_st_DLPFC.h5ad')

    for i, a in enumerate(adata_st_list_raw):
        if not any('-slice' in str(name) for name in a.obs.index[:5]):
            clean = [n.split('-slice')[0].split('_')[0] for n in a.obs_names]
            a.obs_names = [f"{c}-slice{i}" for c in clean]

    all_names = [n for a in adata_st_list_raw for n in a.obs_names]
    if len(all_names) == len(adata_st.obs_names):
        adata_st.obs_names = all_names

    X_arr = adata_st.X.toarray() if scipy.sparse.issparse(adata_st.X) else adata_st.X
    X = torch.from_numpy(X_arr).float()
    raw_coords = np.array(adata_st.obsm['3D_coor'])
    c_min, c_max = raw_coords.min(axis=0), raw_coords.max(axis=0)
    norm_coords = 2.0 * (raw_coords - c_min) / (c_max - c_min + 1e-7) - 1
    coord = torch.from_numpy(norm_coords).float()

    return coord, X, adata_st, adata_st_list_raw


# ==========================================
# 3. 训练 INR-only（只有 MSE + TV loss，无 SAA/Decoder）
#    + ARI 早停（与原 Model 一致）
# ==========================================
def evaluate_ari_inr(net, coord, X, adata_st, adata_st_list_raw,
                     slice_idx, device, n_clusters=5, pca_dims=50, seed=112):
    """用当前的 INR 特征做 PCA+GMM，计算 ARI（用于早停）"""
    net.eval()
    with torch.no_grad():
        mid_fea = net(coord.to(device)).cpu().numpy()
    net.train()

    pca = PCA(n_components=pca_dims, random_state=seed)
    mid_pca = pca.fit_transform(mid_fea)
    gm = GaussianMixture(n_components=n_clusters, covariance_type='tied',
                         reg_covar=1e-4, init_params='kmeans', random_state=seed)
    labels = gm.fit_predict(mid_pca)
    labels_s = pd.Series([str(z) for z in labels], index=adata_st.obs_names)

    ari_list = []
    for i in range(len(adata_st_list_raw)):
        idx = adata_st_list_raw[i].obs_names.tolist()
        pred = labels_s.loc[idx].values
        sid = str(slice_idx[i])
        try:
            gt_path = f'DLPFC_annotations/{sid}_truth.txt'
            if os.path.exists(gt_path):
                gt = pd.read_csv(gt_path, sep='\t', header=None, index_col=0)
                clean = [n.rsplit('-', 1)[0] if '-slice' in n else n for n in idx]
                gt_mapped = pd.Series(clean).map(gt.iloc[:, 0])
                df = pd.DataFrame({'Pred': pred, 'GT': gt_mapped.values}).dropna()
                if len(df) > 0:
                    ari_list.append(adjusted_rand_score(df['Pred'], df['GT']))
        except Exception:
            pass
    return np.mean(ari_list) if ari_list else 0.0


def train_inr_only(net, coord, X, dataloader, device,
                   adata_st, adata_st_list_raw, slice_idx,
                   tv_weight=1e-5, lr=0.001, weight_decay=1e-5,
                   training_steps=10000, patience_steps=1500,
                   eval_interval=200, n_clusters=5, pca_dims=50, seed=112):
    optimizer = torch.optim.Adamax(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()

    best_ari = -1.0
    patience_counter = 0
    best_state = None
    step = 0
    stop = False

    while step < training_steps and not stop:
        for batch_coord, batch_X in dataloader:
            if step >= training_steps:
                stop = True; break

            batch_coord = batch_coord.to(device).requires_grad_(True)
            batch_X = batch_X.to(device)

            mid_fea = net(batch_coord)
            loss_mse = torch.nn.functional.mse_loss(mid_fea, batch_X)

            grads = torch.autograd.grad(
                outputs=mid_fea, inputs=batch_coord,
                grad_outputs=torch.ones_like(mid_fea),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            loss_tv = torch.mean(torch.abs(grads))
            total_loss = loss_mse + tv_weight * loss_tv

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            step += 1

            if step % 500 == 0:
                print(f"  Step {step:5d} | MSE: {loss_mse.item():.4f} | TV: {loss_tv.item():.6f} | Total: {total_loss.item():.4f}")

            # ARI 早停
            if step % eval_interval == 0:
                current_ari = evaluate_ari_inr(
                    net, coord, X, adata_st, adata_st_list_raw, slice_idx,
                    device, n_clusters=n_clusters, pca_dims=pca_dims, seed=seed)
                if current_ari > best_ari:
                    best_ari = current_ari
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                else:
                    patience_counter += eval_interval
                if patience_counter >= patience_steps:
                    stop = True; break

    print(f"  训练结束 (step {step})")
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_ari


# ==========================================
# 4. 主实验
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=112)
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=1500)
    parser.add_argument('--inr_width', type=int, default=160)
    parser.add_argument('--inr_depth', type=int, default=3)
    parser.add_argument('--tv_weight', type=float, default=1e-05)
    parser.add_argument('--omega_0', type=float, default=20.0)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--pca_dims', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--slice_idx', type=int, nargs='+',
                        default=[151673, 151674, 151675, 151676])
    parser.add_argument('--slice_count', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- 加载数据 ----
    print("=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)
    coord, X, adata_st, adata_st_list_raw = load_data(args.slice_count, args.slice_idx)
    coord, X = coord.to(device), X.to(device)
    gene_dim = X.shape[1]
    print(f"  Spots: {coord.shape[0]}, Genes: {gene_dim}, Coords: {coord.shape[1]}")

    dataset = SimpleDataset(coord.cpu(), X.cpu())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ---- 训练 INR-only ----
    print("\n" + "=" * 60)
    print("Step 2: 训练 INR-only (只有 MSE + TV，无 SAA/Decoder)")
    print("=" * 60)
    net = INR_Only(gene_dim=gene_dim, inr_width=args.inr_width,
                   inr_depth=args.inr_depth, omega_0=args.omega_0,
                   coord_dim=coord.shape[1]).to(device)
    print(f"  参数量: {sum(p.numel() for p in net.parameters()):,}")

    start = time.time()
    net, best_ari_during_train = train_inr_only(
        net, coord, X, dataloader, device,
        adata_st, adata_st_list_raw, args.slice_idx,
        tv_weight=args.tv_weight, lr=args.lr,
        weight_decay=args.weight_decay,
        training_steps=args.training_steps,
        patience_steps=args.patience,
        eval_interval=200,
        n_clusters=args.n_clusters,
        pca_dims=args.pca_dims,
        seed=args.seed)
    train_time = time.time() - start
    print(f"  耗时: {train_time:.1f}s")

    # ---- 提取 INR 特征 ----
    print("\n" + "=" * 60)
    print("Step 3: 提取 INR 特征 (mid_fea)")
    print("=" * 60)
    net.eval()
    with torch.no_grad():
        mid_fea = net(coord).cpu().numpy()  # [N, gene_dim]
    print(f"  mid_fea shape: {mid_fea.shape}")

    # ---- PCA → 50 维 ----
    print("\n" + "=" * 60)
    print(f"Step 4: PCA → {args.pca_dims} 维")
    print("=" * 60)
    pca = PCA(n_components=args.pca_dims, random_state=args.seed)
    mid_pca = pca.fit_transform(mid_fea)
    print(f"  累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

    # ---- GMM 聚类 ----
    print("\n" + "=" * 60)
    print("Step 5: GMM 聚类")
    print("=" * 60)
    gm = GaussianMixture(n_components=args.n_clusters, covariance_type='tied',
                         reg_covar=1e-4, init_params='kmeans', random_state=args.seed)
    labels = gm.fit_predict(mid_pca)
    print(f"  各类样本数: {np.bincount(labels)}")

    # ---- 计算 ARI ----
    print("\n" + "=" * 60)
    print("Step 6: 计算逐切片 ARI")
    print("=" * 60)
    adata_st.obs["INR_PCA"] = pd.Series([str(z) for z in labels], index=adata_st.obs_names)

    # Ground Truth
    all_gt = pd.Series(index=adata_st.obs_names, dtype='object')
    for i, raw_ad in enumerate(adata_st_list_raw):
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
        except Exception:
            pass
    adata_st.obs['Ground Truth'] = all_gt.fillna('nan').astype(str).values

    ari_list = []
    for i, sec_raw in enumerate(adata_st_list_raw):
        sid = str(args.slice_idx[i])
        idx = sec_raw.obs_names
        pred = adata_st.obs.loc[idx, "INR_PCA"].values
        gt = adata_st.obs.loc[idx, 'Ground Truth'].values
        df = pd.DataFrame({'Pred': pred, 'GT': gt})
        df = df[df['GT'] != 'nan']
        if len(df) > 0:
            ari = adjusted_rand_score(df['Pred'], df['GT'])
            ari_list.append(ari)
    avg_ari = np.mean(ari_list) if ari_list else 0.0

    # ---- 准备输出目录 ----
    exp_tag = f"INR_PCA{args.pca_dims}_W{args.inr_width}_D{args.inr_depth}_TV{args.tv_weight}_W0{args.omega_0}"
    save_dir = os.path.join("./inr_pca_results", exp_tag)
    os.makedirs(save_dir, exist_ok=True)

    # ---- 可视化：聚类图 ----
    print("\n" + "=" * 60)
    print("Step 7: 生成聚类图")
    print("=" * 60)
    plt.rcParams['font.family'] = 'serif'

    for i, sec_raw in enumerate(adata_st_list_raw):
        sid = str(args.slice_idx[i])
        slice_dir = os.path.join(save_dir, sid)
        os.makedirs(slice_dir, exist_ok=True)

        sec = sec_raw.copy()
        sec.var_names_make_unique()
        lib_id = list(sec.uns['spatial'].keys())[0]
        sec.obs["INR_PCA"] = adata_st.obs.loc[sec.obs_names, "INR_PCA"].values
        sec.obs['Ground Truth'] = adata_st.obs.loc[sec.obs_names, 'Ground Truth'].values

        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.spatial(sec, color='INR_PCA', ax=ax, show=False, title='',
                      legend_loc='right margin', library_id=lib_id)
        ax.set_title(f'INR+PCA (Slice {sid})', fontsize=14)
        fig.savefig(os.path.join(slice_dir, "pred_spatial.pdf"), bbox_inches='tight')
        plt.close(fig)

        print(f"  Slice {sid} 聚类图已保存")

    # ---- 保存数据 ----
    print("\n" + "=" * 60)
    print("Step 8: 保存结果")
    print("=" * 60)

    result_df = pd.DataFrame({
        'Slice': [str(args.slice_idx[i]) for i in range(len(ari_list))],
        'ARI': ari_list
    })
    result_df.to_csv(os.path.join(save_dir, "ari_per_slice.csv"), index=False)

    summary = pd.DataFrame([{
        'Experiment': f'INR+PCA-{args.pca_dims}',
        'Slices': '_'.join(str(s) for s in args.slice_idx),
        'Mean_ARI': round(avg_ari, 4),
        'PCA_Explained_Var': round(pca.explained_variance_ratio_.sum(), 4),
        'Train_Time_s': round(train_time, 1),
        'INR_Width': args.inr_width,
        'INR_Depth': args.inr_depth,
        'TV': args.tv_weight,
        'W0': args.omega_0,
    }])
    summary_path = os.path.join(save_dir, "..", "inr_pca_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n结果已保存到: {os.path.abspath(summary_path)}")


if __name__ == '__main__':
    main()
