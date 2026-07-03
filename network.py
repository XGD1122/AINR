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


# ========================================================
# [NEW] 自定义空间感知注意力层 (Spatial-Aware Attention)
# ========================================================
class SpatiallyAwareAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead

        # Q, K, V 投影
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # [核心] 可学习的距离权重系数，控制距离对注意力的影响强度
        self.dist_weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, Z, coord):
        # Z: [N, d_model], coord: [N, 3]
        N = Z.shape[0]

        # 1. 线性投影并切分为多头 [nhead, N, d_k]
        Q = self.q_linear(Z).view(N, self.nhead, self.d_k).transpose(0, 1)
        K = self.k_linear(Z).view(N, self.nhead, self.d_k).transpose(0, 1)
        V = self.v_linear(Z).view(N, self.nhead, self.d_k).transpose(0, 1)

        # 2. 计算传统的 Dot-Product Attention Score [nhead, N, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 3. [核心实现] 注入物理距离偏差 (Spatial Distance Bias)
        # 计算所有点对之间的欧几里得距离 [N, N]
        # dist_matrix[i, j] = ||coord[i] - coord[j]||_2
        dist_matrix = torch.cdist(coord, coord, p=2)

        # 将距离矩阵广播到所有 head，并减去它（距离越大，权重越低）
        # scores: [nhead, N, N], dist_matrix: [N, N]
        scores = scores - self.dist_weight * dist_matrix.unsqueeze(0)

        # 4. Softmax 归一化
        attn = F.softmax(scores, dim=-1)

        # 5. 加权求和 [nhead, N, d_k] -> [N, d_model]
        out = torch.matmul(attn, V)
        out = out.transpose(0, 1).contiguous().view(N, -1)
        return out


class SpatiallyAwareTransformerLayer(nn.Module):
    """
    完整的 Transformer 层，包含：
    SpatiallyAwareAttention -> LayerNorm -> FFN -> LayerNorm (含残差连接)
    """

    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = SpatiallyAwareAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈神经网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, Z, coord):
        # 1. 自注意力 + 残差连接 + 层归一化
        attn_out = self.attention(Z, coord)
        Z = self.norm1(Z + attn_out)

        # 2. 前馈网络 + 残差连接 + 层归一化
        ffn_out = self.ffn(Z)
        Z = self.norm2(Z + ffn_out)
        return Z


# ========================================================
# AINR 主网络
# ========================================================
class DeconvNet_INR_Recon_Attn2(nn.Module):
    def __init__(self, hidden_dims, nhead=8, recon_weight=2.0,
                 inr_width=200, inr_depth=3,
                 dec_width=200, dec_depth=3,
                 omega_0=30.0):
        super().__init__()

        self.X_dim, self.Z_dim = hidden_dims
        self.recon_weight = recon_weight

        # 1. INR: 坐标 -> 基因初步表达 (mid_fea)
        inr_layers = [SineLayer(3, inr_width, is_first=True, omega_0=omega_0)]
        for _ in range(inr_depth - 1):
            inr_layers.append(SineLayer(inr_width, inr_width, is_first=False, omega_0=1))
        inr_layers.append(DenseLayer(inr_width, self.X_dim))
        self.coord_encoder_layer0 = nn.Sequential(*inr_layers)

        # 2. 投影到潜在空间 Z
        if self.Z_dim % nhead != 0:
            raise ValueError(f"Z_dim ({self.Z_dim}) 必须能被 nhead ({nhead}) 整除")
        self.latent_proj = DenseLayer(self.X_dim, self.Z_dim)

        # 3. [UPDATE] 使用自定义的空间感知 Transformer 层
        # 如果需要更深，可以使用 nn.Sequential([SpatiallyAwareTransformerLayer(...) for _ in range(num_layers)])
        self.spatial_encoder = SpatiallyAwareTransformerLayer(self.Z_dim, nhead)

        # 4. Cross-Attention 用于超分辨率预测
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.Z_dim, num_heads=nhead, batch_first=False)

        # 5. 解码重构: Z -> 基因表达 (recon)
        dec_layers = [SineLayer(self.Z_dim, dec_width)]
        for _ in range(dec_depth - 1):
            dec_layers.append(SineLayer(dec_width, dec_width))
        dec_layers.append(DenseLayer(dec_width, self.X_dim))
        self.feature_decoder = nn.Sequential(*dec_layers)

    def forward(self, coord, node_feats, tv_weight=0.01):
        if not coord.requires_grad:
            coord = coord.detach().requires_grad_(True)

        mid_fea = self.coord_encoder_layer0(coord)

        # TV Loss 计算
        grads = torch.autograd.grad(outputs=mid_fea, inputs=coord,
                                    grad_outputs=torch.ones_like(mid_fea),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        loss_tv = torch.mean(torch.abs(grads))

        Z = self.latent_proj(mid_fea)

        # [UPDATE] 将坐标 coord 传入空间感知 Transformer
        Z_enc = self.spatial_encoder(Z, coord)

        recon = self.feature_decoder(Z_enc)

        loss_mid = F.mse_loss(mid_fea, node_feats)
        loss_final = F.mse_loss(recon, node_feats)
        total_loss = loss_mid + self.recon_weight * loss_final + tv_weight * loss_tv

        return total_loss, loss_mid, loss_final, loss_tv, mid_fea, recon, Z_enc

    def compute_spatial_gradient(self, coord):
        is_training = self.training
        self.eval()
        coord = coord.detach().clone().requires_grad_(True)
        mid_fea = self.coord_encoder_layer0(coord)
        Z = self.latent_proj(mid_fea)

        # 注意：这里的梯度是基于 INR 的 Z，不包含 Transformer 聚合后的结果
        # 这样能得到最纯粹的局部几何边界
        grads = torch.autograd.grad(outputs=Z, inputs=coord,
                                    grad_outputs=torch.ones_like(Z),
                                    create_graph=False)[0]
        grad_magnitude = torch.norm(grads, p=2, dim=1)
        self.train(is_training)
        return grad_magnitude.detach()

    @torch.no_grad()
    def super_resolve(self, coord_obs, coord_new):
        is_training = self.training
        self.eval()
        mid_obs = self.coord_encoder_layer0(coord_obs)
        Z_obs = self.latent_proj(mid_obs)
        mid_new = self.coord_encoder_layer0(coord_new)
        Z_new = self.latent_proj(mid_new)

        attn_out, _ = self.cross_attn(query=Z_new.unsqueeze(1),
                                      key=Z_obs.unsqueeze(1),
                                      value=Z_obs.unsqueeze(1))
        Z_new_ctx = attn_out.squeeze(1)
        recon_new = self.feature_decoder(Z_new_ctx)
        self.train(is_training)
        return Z_new_ctx, recon_new

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

            # [UPDATE] 评估时也需要传入坐标
            Z_enc = self.spatial_encoder(Z, batch_coord)
            recon = self.feature_decoder(Z_enc)

            latent_list.append(Z_enc.cpu())
            mid_fea_list.append(mid_fea.cpu())
            recon_list.append(recon.cpu())

        return (
            torch.cat(latent_list, dim=0).to(device),
            torch.cat(mid_fea_list, dim=0).to(device),
            torch.cat(recon_list, dim=0).to(device),
        )