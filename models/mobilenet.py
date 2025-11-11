import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- 轻量 Transformer 块 -------------------
class LiteTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, expansion=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_ = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(self.norm1(x_), self.norm1(x_), self.norm1(x_))
        x_ = x_ + attn_out
        x_ = x_ + self.ffn(self.norm2(x_))
        x = x_.permute(0, 2, 1).view(B, C, H, W)
        return x


# ------------------- 轻量 MobileNet 块 -------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# ------------------- Hybrid Encoder -------------------
class HybridEncoder(nn.Module):
    def __init__(self, inp_channels=1, dim=64):
        super(HybridEncoder, self).__init__()

        # Transformer 分支（基础特征）
        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1)
        self.transformer = nn.Sequential(
            LiteTransformerBlock(dim=dim, num_heads=4, expansion=2),
            LiteTransformerBlock(dim=dim, num_heads=4, expansion=2)
        )

        # MobileNet 分支（细节特征）
        self.detail = nn.Sequential(
            DepthwiseSeparableConv(inp_channels, 16, stride=1),
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=1)
        )

        # 投影层保证输出维度一致
        self.proj_base = nn.Conv2d(dim, 64, 1)
        self.proj_detail = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        # Transformer 分支
        base_feat = self.patch_embed(x)

        # ⚡ 下采样降低 Transformer 序列长度，避免 OOM
        base_feat = F.adaptive_avg_pool2d(base_feat, output_size=(16,16))

        base_feat = self.transformer(base_feat)
        base_feat = self.proj_base(base_feat)

        # MobileNet 分支
        detail_feat = self.detail(x)
        detail_feat = self.proj_detail(detail_feat)

        # 对齐空间尺寸
        if base_feat.shape[2:] != detail_feat.shape[2:]:
            base_feat = F.interpolate(base_feat, size=detail_feat.shape[2:], mode='bilinear', align_corners=False)

        out_enc_level1 = (base_feat + detail_feat) / 2
        return base_feat, detail_feat, out_enc_level1
