import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DenseLayer(nn.Module):
    def __init__(self, c_in, c_out, zero_init=False):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out)
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        return self.linear(x)


class SineLayer(nn.Module):
    def __init__(self, c_in, c_out, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = c_in
        self.linear = nn.Linear(c_in, c_out, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class DeconvNet_INR_Recon_Attn2(nn.Module):
    def __init__(self, hidden_dims, nhead=8, recon_weight=2.0,
                 inr_width=200, inr_depth=3,
                 dec_width=200, dec_depth=3):
        super().__init__()

        self.X_dim, self.Z_dim = hidden_dims  # 这里的 Z_dim 通常是 64 或 128
        self.recon_weight = recon_weight

        # 1. INR: 坐标 -> 基因初步表达 (mid_fea)
        inr_layers = [SineLayer(3, inr_width, is_first=True, omega_0=30.0)]
        for _ in range(inr_depth - 1):
            inr_layers.append(SineLayer(inr_width, inr_width, is_first=False, omega_0=1))
        inr_layers.append(DenseLayer(inr_width, self.X_dim))
        self.coord_encoder_layer0 = nn.Sequential(*inr_layers)

        # 2. 投影到潜在空间 Z
        if self.Z_dim % nhead != 0:
            raise ValueError(f"Z_dim ({self.Z_dim}) 必须能被 nhead ({nhead}) 整除")
        self.latent_proj = DenseLayer(self.X_dim, self.Z_dim)

        # 3. Transformer 直接处理 Z
        # 注意：这里 d_model 直接等于 self.Z_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.Z_dim,
            nhead=nhead,
            dim_feedforward=self.Z_dim * 4,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 4. 解码重构: Z -> 基因表达 (recon)
        dec_layers = [SineLayer(self.Z_dim, dec_width)]
        for _ in range(dec_depth - 1):
            dec_layers.append(SineLayer(dec_width, dec_width))
        dec_layers.append(DenseLayer(dec_width, self.X_dim))
        self.feature_decoder = nn.Sequential(*dec_layers)

    def forward(self, coord, node_feats):
        # A. INR 映射
        mid_fea = self.coord_encoder_layer0(coord)

        # B. 投影到 Z 并直接进入 Transformer
        Z = self.latent_proj(mid_fea)

        # 增加维度以适配 Transformer [Seq, Batch, Feature]
        # 这里将整个 Batch 视为一个序列，Batch 维设为 1
        Z_in = Z.unsqueeze(1)
        Z_enc = self.encoder(Z_in).squeeze(1)  # [N, Z_dim]

        # C. 解码
        recon = self.feature_decoder(Z_enc)

        # D. Loss: 两个 MSE
        loss_mid = F.mse_loss(mid_fea, node_feats)
        loss_final = F.mse_loss(recon, node_feats)
        total_loss = loss_mid + self.recon_weight * loss_final

        return total_loss, mid_fea, recon, Z_enc, torch.tensor(0.0, device=coord.device)

    @torch.no_grad()
    def evaluate(self, coord, batch_size=2048):
        device = coord.device
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(coord)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        latent_list, mid_fea_list, recon_list = [], [], []

        for (batch_coord,) in loader:
            batch_coord = batch_coord.to(device)

            mid_fea = self.coord_encoder_layer0(batch_coord)
            Z = self.latent_proj(mid_fea)

            # Transformer 评估
            Z_in = Z.unsqueeze(1)
            Z_enc = self.encoder(Z_in).squeeze(1)
            recon = self.feature_decoder(Z_enc)

            latent_list.append(Z_enc.cpu())
            mid_fea_list.append(mid_fea.cpu())
            recon_list.append(recon.cpu())

        return (
            torch.cat(latent_list, dim=0).to(device),
            torch.cat(mid_fea_list, dim=0).to(device),
            torch.cat(recon_list, dim=0).to(device),
        )