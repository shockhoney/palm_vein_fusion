import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_feature(x: torch.Tensor):
    """
    将任意形状的特征展开成 (B, N, C) 的序列形式，便于送入注意力模块。
    返回序列以及重建所需的元信息。
    """
    if x.dim() == 4:
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).transpose(1, 2)  # (B, HW, C)
        meta = {"type": "spatial", "shape": (h, w)}
    elif x.dim() == 3:
        # 默认输入已经是 (B, N, C)
        seq = x
        meta = {"type": "sequence", "length": x.size(1)}
    elif x.dim() == 2:
        # 退化为长度为 1 的序列
        seq = x.unsqueeze(1)
        meta = {"type": "vector"}
    else:
        raise ValueError(f"Unsupported feature dimension: {x.shape}")
    return seq, meta


def _recover_feature(seq: torch.Tensor, meta, channel_dim: int):
    """
    将序列恢复到与输入相同的形状。`channel_dim` 代表当前序列的通道数（注意力输出维度）。
    """
    if meta["type"] == "spatial":
        h, w = meta["shape"]
        return seq.transpose(1, 2).reshape(seq.size(0), channel_dim, h, w)
    if meta["type"] == "sequence":
        return seq
    # vector
    return seq.squeeze(1)


def _ensure_rank4(x: torch.Tensor) -> torch.Tensor:
    """
    统一输出到 NCHW，便于后续卷积融合。
    非空间特征会被视作 1x1 或 1xL 的伪特征图。
    """
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        # (B, N, C) -> (B, C, N, 1)
        return x.transpose(1, 2).unsqueeze(-1)
    if x.dim() == 2:
        return x.unsqueeze(-1).unsqueeze(-1)
    raise ValueError(f"Cannot convert tensor with shape {x.shape} to NCHW format")


def _match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    将张量 x 的空间尺寸调整为与 ref 一致。
    """
    if x.dim() != 4 or ref.dim() != 4:
        return x
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)


class CrossAttentionUnit(nn.Module):
    """
    基于 Multi-Head Attention 的跨模态交互模块：
    - 查询来自主分支
    - 键和值来自另一分支
    - 之后跟随一个前馈网络
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(q_feat)
        kv = self.norm_kv(kv_feat)
        attn_out, _ = self.attn(q, kv, kv)
        out = q_feat + attn_out
        out = out + self.ffn(out)
        return out


class CrossFusionBlock(nn.Module):
    """
    将两路特征映射到统一维度后，进行双向跨注意力交互。
    """

    def __init__(self, in_dim_a: int, in_dim_b: int, hidden_dim: int,
                 num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.proj_a = nn.Linear(in_dim_a, hidden_dim)
        self.proj_b = nn.Linear(in_dim_b, hidden_dim)
        self.cross_a = CrossAttentionUnit(hidden_dim, num_heads, mlp_ratio, dropout)
        self.cross_b = CrossAttentionUnit(hidden_dim, num_heads, mlp_ratio, dropout)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        seq_a, meta_a = _flatten_feature(feat_a)
        seq_b, meta_b = _flatten_feature(feat_b)

        seq_a = self.proj_a(seq_a)
        seq_b = self.proj_b(seq_b)

        updated_a = self.cross_a(seq_a, seq_b)
        updated_b = self.cross_b(seq_b, seq_a)

        out_a = _recover_feature(updated_a, meta_a, updated_a.size(-1))
        out_b = _recover_feature(updated_b, meta_b, updated_b.size(-1))
        return out_a, out_b


class Stage2Fusion(nn.Module):
    """
    第二阶段融合模块：
    - global_blocks：用于 EfficientViT (全局特征) 的跨模态交互
    - local_blocks：用于 ConvNeXt (局部/细节特征) 的跨模态交互
    - 最终输出：global_fused、local_fused 以及二者拼接后的 fused_all
      方便连接 ArcFace 等自定义分类头
    """

    def __init__(self,
                 vit_dim: int = 192,
                 cnn_dim: int = 768,
                 fusion_dim: int = 256,
                 num_heads: int = 4,
                 depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()

        self.global_blocks = nn.ModuleList()
        dim_a = dim_b = vit_dim
        for _ in range(depth):
            self.global_blocks.append(
                CrossFusionBlock(dim_a, dim_b, fusion_dim, num_heads=num_heads, dropout=dropout)
            )
            dim_a = dim_b = fusion_dim

        self.local_blocks = nn.ModuleList()
        dim_a = dim_b = cnn_dim
        for _ in range(depth):
            self.local_blocks.append(
                CrossFusionBlock(dim_a, dim_b, fusion_dim, num_heads=num_heads, dropout=dropout)
            )
            dim_a = dim_b = fusion_dim

        self.global_merge = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU(),
        )
        self.local_merge = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU(),
        )
        self.out_bn = nn.BatchNorm2d(fusion_dim * 2)
        self.out_conv = nn.Conv2d(fusion_dim * 2, fusion_dim * 2, kernel_size=1)

    def forward(self,
                vit_palm: torch.Tensor,
                vit_vein: torch.Tensor,
                cnn_palm: torch.Tensor,
                cnn_vein: torch.Tensor):
        # 全局分支（ViT）
        g_palm, g_vein = vit_palm, vit_vein
        for block in self.global_blocks:
            g_palm, g_vein = block(g_palm, g_vein)

        # 细节分支（CNN）
        l_palm, l_vein = cnn_palm, cnn_vein
        for block in self.local_blocks:
            l_palm, l_vein = block(l_palm, l_vein)

        global_fused = self._merge_pair(g_palm, g_vein, self.global_merge)
        local_fused = self._merge_pair(l_palm, l_vein, self.local_merge)
        local_fused = _match_spatial(local_fused, global_fused)

        fused_all = torch.cat([global_fused, local_fused], dim=1)
        fused_all = self.out_bn(self.out_conv(fused_all))
        return global_fused, local_fused, fused_all

    def _merge_pair(self, feat_a: torch.Tensor, feat_b: torch.Tensor, projector: nn.Module):
        feat_a = _ensure_rank4(feat_a)
        feat_b = _ensure_rank4(feat_b)
        feat_b = _match_spatial(feat_b, feat_a)
        fused = torch.cat([feat_a, feat_b], dim=1)
        return projector(fused)


__all__ = ["Stage2Fusion", "CrossFusionBlock", "CrossAttentionUnit"]
