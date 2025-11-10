import torch
import torch.nn as nn 
import numbers  
import torch.nn.functional as F 
from einops import rearrange 
from torchvision.ops import DeformConv2d
import numpy as np

# ==================== 定义 AttentionBase 类（基础注意力机制）====================
class AttentionBase(nn.Module):
    def __init__(self,
                 dim,  
                 num_heads=8,  
                 qkv_bias=False,):  
        super(AttentionBase, self).__init__()  
        
        self.num_heads = num_heads  
        

        head_dim = dim // num_heads  
        
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))  
        
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)  
        
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)  
        
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)  

    def forward(self, x): 
        b, c, h, w = x.shape  
        
        qkv = self.qkv2(self.qkv1(x))  
        
        q, k, v = qkv.chunk(3, dim=1)  
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)  
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)  
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)  
        
        q = torch.nn.functional.normalize(q, dim=-1)  
        k = torch.nn.functional.normalize(k, dim=-1)  
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  
        
        attn = attn.softmax(dim=-1)  

        out = (attn @ v)  

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)  

        out = self.proj(out)  
        return out  
    
# ==================== 定义 Mlp 类（多层感知机）====================
class Mlp(nn.Module):
    def __init__(self, 
                 in_features,  
                 hidden_features=None,  
                 ffn_expansion_factor = 2,  
                 bias = False):  
        super().__init__()  
        
        hidden_features = int(in_features*ffn_expansion_factor)  

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)  

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)  
    
    def forward(self, x):  
        x = self.project_in(x)  
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  
        x = F.gelu(x1) * x2  
        x = self.project_out(x)  
        return x  

# ==================== 定义 BaseFeatureExtraction 类（基础特征提取）====================
class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,  
                 num_heads,  
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):  
        super(BaseFeatureExtraction, self).__init__()  
        
        self.norm1 = LayerNorm(dim, 'WithBias')  
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        
        self.norm2 = LayerNorm(dim, 'WithBias')  
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    
    def forward(self, x):  
        x = x + self.attn(self.norm1(x))  
        x = x + self.mlp(self.norm2(x))  
        return x  

# ==================== 定义 InvertedResidualBlock 类（倒置残差块）====================
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):  
        super(InvertedResidualBlock, self).__init__()
        
        hidden_dim = int(inp * expand_ratio)  
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),  
            nn.ReLU6(inplace=True),  
            nn.ReflectionPad2d(1),  
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),  
            nn.ReLU6(inplace=True),  
            nn.Conv2d(hidden_dim, oup, 1, bias=False),  
        )
    
    def forward(self, x):  
        return self.bottleneckBlock(x)  

# ==================== 定义 DetailNode 类（细节节点）====================
class DetailNode(nn.Module):
    def __init__(self):  
        super(DetailNode, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)  
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)  
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)  
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,  # 64通道 → 64通道
                                    stride=1, padding=0, bias=True)
    
    def separateFeature(self, x):  
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]  
        return z1, z2  
    
    def forward(self, z1, z2):  
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))  
        z2 = z2 + self.theta_phi(z1)  
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)  
        return z1, z2  

# ==================== 定义 DetailFeatureExtraction 类（细节特征提取）====================
class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):  
        super(DetailFeatureExtraction, self).__init__()
        
        INNmodules = [DetailNode() for _ in range(num_layers)]  
        self.net = nn.Sequential(*INNmodules)  
    
    def forward(self, x):  
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        
        for layer in self.net:
            z1, z2 = layer(z1, z2)  
        return torch.cat((z1, z2), dim=1)  


# ==================== 维度转换辅助函数 ====================
# 这两个函数用于在4D图像格式和3D序列格式之间转换

def to_3d(x):  
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):  
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# ==================== 定义 BiasFree_LayerNorm 类（无偏置的层归一化）====================
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):  
        super(BiasFree_LayerNorm, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):  
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)  
        assert len(normalized_shape) == 1  
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape  

    def forward(self, x):  
        sigma = x.var(-1, keepdim=True, unbiased=False)  
        return x / torch.sqrt(sigma+1e-5) * self.weight  


# ==================== 定义 WithBias_LayerNorm 类（带偏置的层归一化）====================
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):   
        super(WithBias_LayerNorm, self).__init__()
        
        # 与BiasFree版本相同，处理输入参数
        if isinstance(normalized_shape, numbers.Integral):  
            normalized_shape = (normalized_shape,)  
        normalized_shape = torch.Size(normalized_shape)  
        assert len(normalized_shape) == 1  
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape  

    def forward(self, x):  
        mu = x.mean(-1, keepdim=True)  
        sigma = x.var(-1, keepdim=True, unbiased=False)  
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias  

# ==================== 定义 LayerNorm 类（层归一化统一接口）====================
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):  
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
        
    def forward(self, x):  
        h, w = x.shape[-2:]  
        return to_4d(self.body(to_3d(x)), h, w)  


# ==================== 定义 FeedForward 类（前馈神经网络）====================
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):  
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):  
        x = self.project_in(x)  
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        
        x = F.gelu(x1) * x2
        
        x = self.project_out(x)  
        return x  

# ==================== 定义 Attention 类（改进版注意力机制）====================
# 这个类和前面的AttentionBase类似，但使用temperature代替scale
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):  
        super(Attention, self).__init__()
        self.num_heads = num_heads  
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):  
        b, c, h, w = x.shape  

        qkv = self.qkv_dwconv(self.qkv(x))
        
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out  


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        
        self.attn = Attention(dim, num_heads, bias)
        
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):  
        x = x + self.attn(self.norm1(x))
        
        x = x + self.ffn(self.norm2(x))

        return x  

# 这个类用于将输入图像转换为特征嵌入
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):  
        x = self.proj(x)  
        return x  

# ==================== 定义 Restormer_Encoder类====================
class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,  
                 out_channels=1,  
                 dim=64,  
                 num_blocks=[4, 4],  
                 heads=[8, 8, 8],  
                 ffn_expansion_factor=2,  
                 bias=False,  
                 LayerNorm_type='WithBias',  
                 ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        
        self.detailFeature = DetailFeatureExtraction()
             
    def forward(self, inp_img):  
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        base_feature = self.baseFeature(out_enc_level1)
        
        detail_feature = self.detailFeature(out_enc_level1)
        
        return base_feature, detail_feature, out_enc_level1


class ArcFaceClassifier(nn.Module):
    def __init__(self, feature_dim=64, num_classes=600, s=30.0, m=0.50):
        super(ArcFaceClassifier, self).__init__()
        self.s = s  
        self.m = m  
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.gap = nn.AdaptiveAvgPool2d(1)  

    def forward(self, fused_feature, labels=None):
        """
        fused_feature: 融合后的特征 (batch, feature_dim, H, W)
        labels: 标签 (batch,) 仅在训练时需要;
        """
        x = self.gap(fused_feature)  # (batch, feature_dim, 1, 1)
        feature_vector = x.view(x.size(0), -1)  # (batch, feature_dim)

        # L2 normalize
        normed_feat = F.normalize(feature_vector, dim=1)     # (batch, feature_dim)
        normed_weight = F.normalize(self.weight, dim=1)      # (num_classes, feature_dim)
        cosine = torch.matmul(normed_feat, normed_weight.t())  # (batch, num_classes)

        if labels is not None:
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            target_logits = torch.cos(theta + self.m)
            one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
            output = cosine * (1 - one_hot) + target_logits * one_hot
            logits = self.s * output
        else:
            logits = self.s * cosine

        return logits, feature_vector

class DeformableAlignment(nn.Module):
    def __init__(self, in_channels=64):
        super(DeformableAlignment, self).__init__()
        
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        )
        
        self.deform_conv = DeformConv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)
    
    def forward(self, ref_feat, target_feat):
        concat_feat = torch.cat([ref_feat, target_feat], dim=1)
        offset = self.offset_conv(concat_feat)
        aligned_feat = self.deform_conv(target_feat, offset)
        return aligned_feat

if __name__ == '__main__':
    height = 128
    width = 128
    encoder = Restormer_Encoder().cuda()
    



