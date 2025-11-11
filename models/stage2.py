import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 基础层：跨特征注意力模块 (Cross Attention) ----
class CrossAttention(nn.Module):
    def __init__(self, dim=64, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, base_feat, detail_feat):
        B, C, H, W = base_feat.shape

        q = self.q(base_feat).reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = self.k(detail_feat).reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = self.v(detail_feat).reshape(B, self.num_heads, C // self.num_heads, H * W)

        attn = torch.softmax((q.transpose(-2, -1) @ k) * self.scale, dim=-1)  # (B, heads, HW, HW)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, H, W)
        return self.proj(out)

# ---- 多尺度融合模块 ----
class MultiScaleFusion(nn.Module):
    def __init__(self, dim=64):
        super(MultiScaleFusion, self).__init__()
        self.down1 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.up1 = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        self.conv = nn.Conv2d(dim * 3, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.down1(x)
        x_up = self.up1(x)
    
        #  统一尺寸
        x_down = F.interpolate(x_down, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_up = F.interpolate(x_up, size=x.shape[2:], mode='bilinear', align_corners=False)
    
        x = torch.cat([x_down, x_up, x], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        return x


# ---- 主体：细致融合 Transformer ----
class FineFusionTransformer(nn.Module):
    def __init__(self, dim=64, num_heads=4, num_blocks=2):
        super(FineFusionTransformer, self).__init__()
        self.cross_blocks = nn.ModuleList([CrossAttention(dim, num_heads) for _ in range(num_blocks)])
        self.ms_blocks = nn.ModuleList([MultiScaleFusion(dim) for _ in range(num_blocks)])
        self.final_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, base_feat, detail_feat):
        if base_feat.shape[2:] != detail_feat.shape[2:]:
            base_feat = F.interpolate(base_feat, size=detail_feat.shape[2:], mode='bilinear', align_corners=False)

        x = base_feat
        for cross, ms in zip(self.cross_blocks, self.ms_blocks):
            # 跨注意力融合：细节引导语义
            cross_out = cross(x, detail_feat)
            # 多尺度增强：引入局部与全局尺度上下文
            ms_out = ms(x)
            x = x + cross_out + ms_out

        fused = self.final_conv(self.act(self.norm(x)))
        return fused
