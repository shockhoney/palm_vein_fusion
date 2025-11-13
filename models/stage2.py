import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 工具：特征对齐（Linear -> same dim）
# -------------------------------
class FeatureAlign(nn.Module):
    """
    将输入向量线性投影到统一维度，便于注意力/加法/拼接。
    """
    def __init__(self, in_dim: int, out_dim: int, use_bn: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=not use_bn)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='linear')
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        if isinstance(self.bn, nn.BatchNorm1d):
            x = self.bn(x)
        return x


# ---------------------------------------------------------
# 空间注意力：对两个空间特征图使用空间注意力进行自适应融合
# ---------------------------------------------------------
class SpatialAttentionFusion(nn.Module):
    """
    对两个空间特征图使用空间注意力进行自适应融合。
    输入：两路同尺寸特征图 a, b (N, C, H, W)；输出同尺寸 (N, C, H, W)。
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hid = max(in_channels // reduction, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 2, 1, bias=True)  # 2 路权重
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        assert a.shape == b.shape, "SpatialAttentionFusion: a/b 尺寸必须一致"
        x = torch.cat([a, b], dim=1)              # (N, 2C, H, W)
        logits = self.conv(x)                      # (N, 2, H, W)
        weights = F.softmax(logits, dim=1)         # 在 2 路上做 softmax
        w_a, w_b = weights[:, 0:1], weights[:, 1:2]
        fused = w_a * a + w_b * b
        return fused, (w_a, w_b)


class ConvAlign2d(nn.Module):
    """
    1x1 卷积 + BN：将通道数对齐到目标维度，用于空间特征图对齐。
    """
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        self.proj = nn.Sequential(*layers)
        # init
        for m in self.proj:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------
# 通道注意力（Channel Attention）对"二路向量"做逐通道加权融合
# ---------------------------------------------------------
class PairwiseChannelAttentionFusion(nn.Module):
    """
    输入：两路同维度向量 a, b （形状 (N, C)）
    思路：
      1) 拼接 [a, b] -> (N, 2C)
      2) MLP 产生每个通道在 a/b 两路上的权重 logits -> (N, 2C)
      3) 通道维度 reshape -> (N, 2, C)，在"路维"上做 softmax 得到 w_a, w_b
      4) 融合：fused = w_a * a + w_b * b
    优点：逐通道、权重和为 1，稳定可解释。
    """
    def __init__(self, dim: int, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        hid = max(dim // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hid, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hid, 2 * dim, bias=True)
        )
        # 初始化
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        assert a.shape == b.shape, "PairwiseChannelAttentionFusion: a/b 维度必须一致"
        x = torch.cat([a, b], dim=1)                # (N, 2C)
        logits = self.mlp(x)                        # (N, 2C)
        logits = logits.view(x.size(0), 2, -1)      # (N, 2, C)
        weights = F.softmax(logits, dim=1)          # (N, 2, C) -> 两路权重逐通道相加为 1
        wa, wb = weights[:, 0, :], weights[:, 1, :] # (N, C), (N, C)
        fused = wa * a + wb * b                     # (N, C)
        return fused, (wa, wb)


# ---------------------------------------------------------
# ArcFace 头（标准 ArcMarginProduct）
# ---------------------------------------------------------
class ArcMarginProduct(nn.Module):
    """
    ArcFace / CosFace 类似的角度间隔分类头（默认 ArcFace）：
    输入向量与分类权重都做 L2 归一化，基于 cos(theta + m) 计算。
    """
    def __init__(self, in_features: int, out_features: int,
                 s: float = 64.0, m: float = 0.50,
                 easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor):
        # 归一化特征与权重
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # (N, out)
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # 输出：真实类用 phi，其余用原 cos
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# ---------------------------------------------------------
# 第二阶段：整合"全局-全局、局部-局部"的融合（支持空间特征）
# ---------------------------------------------------------
class Stage2FusionCA(nn.Module):
    """
    第二阶段融合模型（改进版）：
    - 支持向量特征融合（通道注意力）
    - 支持空间特征融合（空间注意力）

    参数：
      - in_dim_global_palm / vein：第一阶段全局向量维度（默认 ViT embed_dim[-1]）
      - in_dim_local_palm / vein：第一阶段局部向量维度（默认 ConvNeXt dims[-1]）
      - out_dim_global / out_dim_local：对齐后的通道数（建议相同，如 256）
      - use_spatial_fusion：是否对局部特征使用空间注意力融合（保留空间信息）
      - final_l2norm：融合后是否 L2 归一化（建议 True，以便 ArcFace）
      - with_arcface：是否启用 ArcFace 分类头（可选）

    前向：
      forward_features(...) -> fused_feat（拼接后的融合向量）
      forward(..., labels)  -> ArcFace logits（若 with_arcface=True）
    """
    def __init__(self,
                 in_dim_global_palm: int = 192,
                 in_dim_global_vein: int = 192,
                 in_dim_local_palm: int = 768,
                 in_dim_local_vein: int = 768,
                 out_dim_global: int = 256,
                 out_dim_local: int = 256,
                 use_spatial_fusion: bool = False,
                 final_l2norm: bool = True,
                 with_arcface: bool = False,
                 num_classes: int = 0,
                 arcface_s: float = 64.0,
                 arcface_m: float = 0.50):
        super().__init__()

        self.use_spatial_fusion = use_spatial_fusion

        # 1) 全局特征对齐（向量 -> 向量）
        self.g_align_palm = FeatureAlign(in_dim_global_palm, out_dim_global, use_bn=False)
        self.g_align_vein = FeatureAlign(in_dim_global_vein, out_dim_global, use_bn=False)

        # 2) 局部特征对齐：根据是否使用空间融合选择不同的对齐方式
        if use_spatial_fusion:
            # 空间特征融合：用卷积对齐通道数
            self.l_align_palm = ConvAlign2d(in_dim_local_palm, out_dim_local, use_bn=True)
            self.l_align_vein = ConvAlign2d(in_dim_local_vein, out_dim_local, use_bn=True)
            self.l_fuse = SpatialAttentionFusion(out_dim_local, reduction=16)
        else:
            # 向量特征融合：用全连接对齐维度
            self.l_align_palm = FeatureAlign(in_dim_local_palm, out_dim_local, use_bn=False)
            self.l_align_vein = FeatureAlign(in_dim_local_vein, out_dim_local, use_bn=False)
            self.l_fuse = PairwiseChannelAttentionFusion(dim=out_dim_local, reduction=4, dropout=0.0)

        # 3) 全局特征融合（通道注意力）
        self.g_fuse = PairwiseChannelAttentionFusion(dim=out_dim_global, reduction=4, dropout=0.0)

        # 4) 输出拼接 + 规范化
        self.final_dim = out_dim_global + out_dim_local
        self.final_l2norm = final_l2norm

        # 5) （可选）ArcFace 头
        self.with_arcface = with_arcface
        if with_arcface:
            assert num_classes > 0, "with_arcface=True 时必须指定 num_classes"
            self.arcface = ArcMarginProduct(self.final_dim, num_classes,
                                            s=arcface_s, m=arcface_m)

    @staticmethod
    def _l2(x):
        return F.normalize(x, dim=1)

    def forward_features(self,
                         palm_global: torch.Tensor, vein_global: torch.Tensor,
                         palm_local:  torch.Tensor, vein_local:  torch.Tensor):
        """
        输入：
            palm_global, vein_global: (N, C_g) 全局向量
            palm_local, vein_local:
                - 若 use_spatial_fusion=False: (N, C_l) 向量
                - 若 use_spatial_fusion=True: (N, C_l, H, W) 空间特征图
        返回：
            fused_feat: (N, out_dim_global + out_dim_local)
            details: 字典，包含中间权重/中间向量，便于可视化与消融
        """
        # ---- 1) 全局特征对齐与融合
        g_p = self.g_align_palm(palm_global)  # (N, G)
        g_v = self.g_align_vein(vein_global)  # (N, G)
        g_fused, (g_wa, g_wb) = self.g_fuse(g_p, g_v)  # (N, G)

        # ---- 2) 局部特征对齐与融合
        if self.use_spatial_fusion:
            # 空间特征融合
            l_p = self.l_align_palm(palm_local)  # (N, L, H, W)
            l_v = self.l_align_vein(vein_local)  # (N, L, H, W)
            l_fused_spatial, (l_wa, l_wb) = self.l_fuse(l_p, l_v)  # (N, L, H, W)
            # 池化为向量
            l_fused = F.adaptive_avg_pool2d(l_fused_spatial, 1).flatten(1)  # (N, L)
        else:
            # 向量特征融合
            l_p = self.l_align_palm(palm_local)   # (N, L)
            l_v = self.l_align_vein(vein_local)   # (N, L)
            l_fused, (l_wa, l_wb) = self.l_fuse(l_p, l_v)  # (N, L)

        # ---- 3) 拼接全局和局部融合特征
        fused_feat = torch.cat([g_fused, l_fused], dim=1)  # (N, G+L)
        if self.final_l2norm:
            fused_feat = self._l2(fused_feat)

        # 便于排查/可视化
        details = {
            "global": {"palm": g_p, "vein": g_v, "fused": g_fused, "w_palm": g_wa, "w_vein": g_wb},
            "local":  {"palm": l_p, "vein": l_v, "fused": l_fused, "w_palm": l_wa, "w_vein": l_wb},
        }
        return fused_feat, details

    def forward(self,
                palm_global: torch.Tensor, vein_global: torch.Tensor,
                palm_local:  torch.Tensor, vein_local:  torch.Tensor,
                labels: torch.Tensor = None):
        fused_feat, details = self.forward_features(palm_global, vein_global,
                                                    palm_local, vein_local)
        if self.with_arcface:
            assert labels is not None, "使用 ArcFace 前向时需要 labels"
            logits = self.arcface(fused_feat, labels)
            return logits, fused_feat, details
        else:
            return fused_feat, details
