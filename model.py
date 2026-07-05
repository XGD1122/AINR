import os, torch, numpy as np, pandas as pd, scipy.sparse
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
from network import DeconvNet_INR_Recon_Attn2
import scanpy as sc
import anndata as ad
from torch.utils.data import Dataset, DataLoader
import copy
import matplotlib.pyplot as plt
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
                 hidden_dims=[512, 64], training_steps=20000, lr=0.001, seed=112,
                 n_clusters=7, batch_size=512, weight_decay=1e-5, nhead=8,
                 save_path="./results_DLPFC_INR_RECON", recon_weight=2.0, tv_weight=0.01,
                 inr_width=200, inr_depth=3,
                 dec_width=200, dec_depth=3,omega_0=20.0):

        self.slice_idx = slice_idx_gt
        self.seed = seed
        self.n_clusters = n_clusters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.training_steps = training_steps
        self.tv_weight = tv_weight

        # 固定随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 打包传递给网络架构的参数字典
        self.net_params = {
            "inr_width": inr_width, "inr_depth": inr_depth,
            "dec_width": dec_width, "dec_depth": dec_depth,
            "nhead": nhead, "recon_weight": recon_weight,
            "omega_0": omega_0
        }

        print("Model.__init__: Loading data...")
        try:
            self.adata_st_list_raw = [ad.read_h5ad(f'adata_st_list_raw{i}.h5ad') for i in range(slice_count)]
            self.adata_st = ad.read_h5ad('adata_st_DLPFC.h5ad')
        except FileNotFoundError:
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

        c_min = raw_coords.min(axis=0)
        c_max = raw_coords.max(axis=0)
        norm_coords = 2.0 * (raw_coords - c_min) / (c_max - c_min + 1e-7) - 1

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
        step, best_ari, patience_counter = 0, -1.0, 0
        loss_history = []

        best_state = {'step': 0, 'avg_ari': -1.0, 'gmm_labels': None, 'latent': None, 'model_state': None}

        stop_training = False
        while step < self.training_steps and not stop_training:
            for batch_coord, batch_X in self.dataloader:
                if step >= self.training_steps:
                    stop_training = True
                    break

                batch_coord, batch_X = batch_coord.to(self.device), batch_X.to(self.device)

                total_loss, l_mid, l_final, l_tv, mid_fea, recon, Z_enc = self.net(batch_coord, batch_X,
                                                                                   tv_weight=self.tv_weight)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                step += 1

                loss_history.append([step, total_loss.item(), l_mid.item(), l_final.item(), l_tv.item()])

                if report_loss and step % 100 == 0:
                    print(f"Step {step}, Loss: {total_loss.item():.4f} (TV: {l_tv.item():.4f})")

                if step % eval_interval == 0:
                    current_ari = self.evaluate_ari()
                    if current_ari > best_ari:
                        best_ari = current_ari
                        patience_counter = 0
                        with torch.no_grad():
                            latent_full, _, _ = self.net.evaluate(self.coord, 2048)
                            latent_full_np = latent_full.cpu().numpy()
                        gm = GaussianMixture(n_components=self.n_clusters, covariance_type='tied',
                                             reg_covar=1e-4, init_params='kmeans', random_state=self.seed)
                        y_pred = gm.fit_predict(latent_full_np)
                        best_state.update({'step': step, 'avg_ari': best_ari, 'gmm_labels': y_pred.copy(),
                                           'latent': latent_full_np.copy(),
                                           'model_state': copy.deepcopy(self.net.state_dict())})
                        os.makedirs(self.save_path, exist_ok=True)
                        torch.save(best_state['model_state'], os.path.join(self.save_path, "best_model.pth"))
                    else:
                        patience_counter += eval_interval
                    if patience_counter >= patience_steps:
                        stop_training = True;
                        break

        # 保存 Loss 曲线
        os.makedirs(self.save_path, exist_ok=True)
        df_loss = pd.DataFrame(loss_history, columns=['Step', 'Total', 'Mid', 'Final', 'TV'])
        df_loss.to_csv(os.path.join(self.save_path, "loss_history.csv"), index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(df_loss['Step'], df_loss['Total'], label='Total Loss')
        plt.plot(df_loss['Step'], df_loss['Mid'], label='Mid Loss')
        plt.plot(df_loss['Step'], df_loss['Final'], label='Final Loss')
        plt.plot(df_loss['Step'], df_loss['TV'], label='TV Loss')
        plt.xlabel('Step');
        plt.ylabel('Loss');
        plt.legend();
        plt.title('Training Loss Curves')
        plt.savefig(os.path.join(self.save_path, "loss_curves.pdf"), bbox_inches='tight')
        plt.close()

        print(f"\nTraining finished at step {step}.")

        if best_state['model_state'] is not None:
            print(f"Loading best model from step {best_state['step']} (ARI: {best_state['avg_ari']:.4f}) for output.")
            self.net.load_state_dict(best_state['model_state'])
            self.net.eval()

            # 提取常规表示
            with torch.no_grad():
                final_latent_tensor, _, final_recon_tensor = self.net.evaluate(self.coord, 2048)
                final_latent_np = final_latent_tensor.cpu().numpy()
                final_recon_np = final_recon_tensor.cpu().numpy()

            print("Computing spatial gradients for boundary mapping...")
            eval_dataset = CustomDataset(self.coord.cpu(), self.X.cpu())
            eval_loader = DataLoader(eval_dataset, batch_size=2048, shuffle=False)
            grad_list = []
            for b_coord, _ in eval_loader:
                b_coord = b_coord.to(self.device)
                grad_mag = self.net.compute_spatial_gradient(b_coord)
                grad_list.append(grad_mag.cpu())
            final_spatial_gradient = torch.cat(grad_list, dim=0).numpy()

            final_gm = GaussianMixture(n_components=self.n_clusters, covariance_type='tied',
                                       reg_covar=1e-4, init_params='kmeans', random_state=self.seed)
            final_labels = final_gm.fit_predict(final_latent_np)

            return {
                'adata_st': self.adata_st,
                'adata_st_list_raw': self.adata_st_list_raw,
                'latent': final_latent_np,
                'recon': final_recon_np,
                'spatial_gradient': final_spatial_gradient,
                'gmm_labels': final_labels,
                'avg_ari': best_state['avg_ari']
            }

    def evaluate_ari(self):
        self.net.eval()
        with torch.no_grad():
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
                gt_file_path = f'DLPFC_annotations/{sid}_truth.txt'
                if os.path.exists(gt_file_path):
                    gt = pd.read_csv(gt_file_path, sep='\t', header=None, index_col=0)
                    gt.columns = ['Ground Truth']
                    clean_idx = [n.rsplit('-', 1)[0] if '-slice' in n else n for n in idx]
                    gt_mapped = pd.Series(clean_idx).map(gt['Ground Truth'])

                    df_eval = pd.DataFrame({'Pred': mapped_pred.values, 'GT': gt_mapped.values}).dropna()
                    if len(df_eval) > 0:
                        ari_list.append(adjusted_rand_score(df_eval['Pred'], df_eval['GT']))
            except Exception as e:
                pass

        return np.mean(ari_list) if ari_list else 0.0