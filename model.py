import os, torch, numpy as np, pandas as pd, scipy.sparse
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
from network import DeconvNet_INR_Recon_Attn2
import scanpy as sc
import anndata as ad
from torch.utils.data import Dataset, DataLoader
import copy

class CustomDataset(Dataset):
    def __init__(self, coord, X):
        self.coord = coord
        self.X = X

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        return self.coord[idx], self.X[idx]


class Model:
    def __init__(self, slice_count, slice_idx_gt,
                 hidden_dims=[512, 64], training_steps=10000, lr=0.001, seed=112,
                 n_clusters=7, batch_size=512, weight_decay=1e-5, nhead=8,
                 save_path="./results", recon_weight=2.0,
                 # 网络结构参数
                 inr_width=200, inr_depth=3,
                 dec_width=200, dec_depth=3):

        self.slice_idx = slice_idx_gt
        self.seed = seed
        self.n_clusters = n_clusters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.training_steps = training_steps

        self.net_params = {
            "inr_width": inr_width, "inr_depth": inr_depth,
            "dec_width": dec_width, "dec_depth": dec_depth,
            "nhead": nhead, "recon_weight": recon_weight
        }
        data_dir = "Data"

        print("Model.__init__: Loading data...")
        try:
            self.adata_st_list_raw = [ad.read_h5ad(os.path.join(data_dir, f'adata_st_list_raw{i}.h5ad')) for i in range(slice_count)]
            self.adata_st = ad.read_h5ad(os.path.join(data_dir, 'adata_st_DLPFC.h5ad'))
        except FileNotFoundError as e:
            print(f"错误: 在 {data_dir} 文件夹中找不到数据文件 {e.filename}。请检查目录结构。")
            pass

        # 确保 obs_names 唯一性处理
        for i, a in enumerate(self.adata_st_list_raw):
            if not any('-slice' in str(name) for name in a.obs.index[:5]):
                clean = [n.split('-slice')[0].split('_')[0] for n in a.obs_names]
                a.obs_names = [f"{c}-slice{i}" for c in clean]

        all_names = [n for a in self.adata_st_list_raw for n in a.obs_names]
        if len(all_names) == len(self.adata_st.obs_names):
            self.adata_st.obs_names = all_names

        X = self.adata_st.X.toarray() if scipy.sparse.issparse(self.adata_st.X) else self.adata_st.X
        self.X = torch.from_numpy(X).float().to(self.device)
        raw_coords = np.array(self.adata_st.obsm['3D_coor'])

        # 2. 计算最大最小值
        c_min = raw_coords.min(axis=0)
        c_max = raw_coords.max(axis=0)

        # 3. 执行归一化

        norm_coords = 2.0*(raw_coords - c_min) / (c_max - c_min+ 1e-7)-1


        # 4. 转换为 Tensor 并移动到设备
        self.coord = torch.from_numpy(norm_coords).float().to(self.device)

        self.hidden_dims = [self.adata_st.shape[1], hidden_dims[1]]

        dataset = CustomDataset(self.coord.cpu(), self.X.cpu())
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.net = DeconvNet_INR_Recon_Attn2(
            hidden_dims=self.hidden_dims,
            **self.net_params
        ).to(self.device)

        self.optimizer = torch.optim.Adamax(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, report_loss=True, eval_interval=100, patience_steps=2000):
        self.net.train()

        step = 0
        best_ari = -1.0
        best_loss = float('inf')

        patience_counter = 0

        # 初始化最优状态存储
        best_state = {
            'step': 0,
            'avg_ari': -1.0,
            'gmm_labels': None,
            'latent': None,
            'model_state': None
        }

        print(f"Training started. Max steps: {self.training_steps}")

        stop_training = False
        while step < self.training_steps and not stop_training:
            epoch_loss = 0.0
            for batch_coord, batch_X in self.dataloader:
                if step >= self.training_steps:
                    stop_training = True
                    break

                # 1. 正常的梯度下降更新
                batch_coord = batch_coord.to(self.device)
                batch_X = batch_X.to(self.device)

                loss, _, _, _, _ = self.net(batch_coord, batch_X)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step += 1
                epoch_loss += loss.item()

                # 2. 打印 Loss
                if report_loss and step % 100 == 0:
                    print(f"Step {step}, Loss: {loss.item():.4f}")

                # 3. 周期性评估 ARI (每 eval_interval 步一次)
                if step % eval_interval == 0:
                    current_ari = self.evaluate_ari()

                    if current_ari > best_ari:

                        best_ari = current_ari
                        patience_counter = 0  # 重置早停计数器

                        # 提取当前最优状态
                        with torch.no_grad():
                            latent_full, _, _ = self.net.evaluate(self.coord, 2048)
                            latent_full_np = latent_full.cpu().numpy()

                        gm = GaussianMixture(n_components=self.n_clusters, covariance_type='tied',
                                             reg_covar=1e-4, init_params='kmeans', random_state=self.seed)
                        y_pred = gm.fit_predict(latent_full_np)

                        # 更新 best_state 字典
                        best_state['step'] = step
                        best_state['avg_ari'] = best_ari
                        best_state['gmm_labels'] = y_pred.copy()
                        best_state['latent'] = latent_full_np.copy()
                        best_state['model_state'] = copy.deepcopy(self.net.state_dict())

                        # 磁盘备份
                        os.makedirs(self.save_path, exist_ok=True)
                        torch.save(best_state['model_state'], os.path.join(self.save_path, "best_model.pth"))

                    else:
                        # 没有刷新记录，增加早停计数
                        patience_counter += eval_interval
                        print(
                            f"  Step {step} | ARI:  ({best_ari:.4f}) ")

                    # 检查是否触发早停
                    if patience_counter >= patience_steps:

                        stop_training = True
                        break  # 跳出当前的 dataloader 循环

        # ======================================================
        # 重要：以下代码必须在 while/for 循环彻底结束后执行 (缩进归位)
        # ======================================================

        print(f"\nTraining finished at step {step}.")

        if best_state['model_state'] is not None:
            print(f"Loading best model from step {best_state['step']} (ARI: {best_state['avg_ari']:.4f}) for output.")
            self.net.load_state_dict(best_state['model_state'])

            self.net.eval()
            with torch.no_grad():
                final_latent_tensor, _, final_recon_tensor = self.net.evaluate(self.coord, 2048)
                final_latent_np = final_latent_tensor.cpu().numpy()
                final_recon_np = final_recon_tensor.cpu().numpy()

            # 统一使用 self.seed
            final_gm = GaussianMixture(n_components=self.n_clusters, covariance_type='tied',
                                       reg_covar=1e-4, init_params='kmeans', random_state=self.seed)
            final_labels = final_gm.fit_predict(final_latent_np)

            # 确保返回的是这一组新鲜生成的、完全匹配最优权重的标签
            return {
                'adata_st': self.adata_st,
                'adata_st_list_raw': self.adata_st_list_raw,
                'latent': final_latent_np,
                'recon': final_recon_np, # 增加这一行，获取去噪后的表达矩阵
                'gmm_labels': final_labels,
                'avg_ari': best_state['avg_ari']
            }
    def evaluate_ari(self):
        """计算当前权重的全量平均 ARI"""
        self.net.eval()
        with torch.no_grad():
            # 使用 network.py 中的分批 evaluate 接口
            latent, _, _ = self.net.evaluate(self.coord, 2048)
            latent_np = latent.cpu().numpy()
        self.net.train()

        gm = GaussianMixture(n_components=self.n_clusters, covariance_type='tied',
                             reg_covar=1e-4, init_params='kmeans', random_state=self.seed)
        y = gm.fit_predict(latent_np)

        ari_list = []
        gm_series = pd.Series(y.astype(str), index=self.adata_st.obs_names)

        for i in range(len(self.adata_st_list_raw)):
            idx = self.adata_st_list_raw[i].obs_names.tolist()
            mapped_pred = pd.Series(idx).map(gm_series)

            if mapped_pred.dropna().empty: continue

            try:
                sid = str(self.slice_idx[i])
                gt_file_path = os.path.join("Data", "DLPFC_annotations", f'{sid}_truth.txt')
                if os.path.exists(gt_file_path):
                    gt = pd.read_csv(gt_file_path, sep='\t', header=None, index_col=0)
                    gt.columns = ['Ground Truth']
                    # 对齐 slice ID
                    clean_idx = [n.rsplit('-', 1)[0] if '-slice' in n else n for n in idx]
                    gt_mapped = pd.Series(clean_idx).map(gt['Ground Truth'])

                    df_eval = pd.DataFrame({'Pred': mapped_pred.values, 'GT': gt_mapped.values}).dropna()
                    if len(df_eval) > 0:
                        ari_list.append(adjusted_rand_score(df_eval['Pred'], df_eval['GT']))
            except Exception as e:
                # 评估出错不中断训练
                pass

        return np.mean(ari_list) if ari_list else 0.0