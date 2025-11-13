# 掌纹掌静脉融合识别 - 代码改进说明

## 改进总览

### 1. **代码清理**
- ✅ 清理了 `stage1.py` 中大量注释代码（300+ 行），提高可读性
- ✅ 统一了代码风格和注释规范

### 2. **Stage 1 架构改进**

#### EfficientViT (全局特征提取)
- ✅ **新增参数 `pool`**：支持输出空间特征图或全局向量
  ```python
  # 输出全局向量 (N, 192)
  global_feat = vit_model(x, pool=True)

  # 输出空间特征图 (N, 192, H, W)
  global_spatial = vit_model(x, pool=False)
  ```

#### ConvNeXt (局部细节提取)
- ✅ **新增参数 `return_spatial`**：支持保留空间信息
  ```python
  # 输出全局向量 (N, 768)
  local_feat = cnn_model(x, return_spatial=False)

  # 输出空间特征图 (N, 768, H, W) - 保留空间细节
  local_spatial = cnn_model(x, return_spatial=True)
  ```

**改进意义**：
- 原代码直接全局池化丢失空间信息
- 现在可以在 Stage2 使用空间注意力融合，保留位置信息
- 对掌纹掌静脉的细节特征（纹理、血管分布）更有效

---

### 3. **Stage 2 架构重大改进** ⭐

创建了 **`stage2_improved.py`**，核心改进：

#### 3.1 新增空间注意力融合模块
```python
class SpatialAttentionFusion(nn.Module):
    """
    对两个空间特征图进行自适应加权融合
    输入：两路特征图 (N, C, H, W)
    输出：融合后特征图 (N, C, H, W)
    """
```

**优势**：
- 逐像素自适应权重，保留空间细节
- 比简单相加或通道注意力更精细
- 特别适合掌纹/掌静脉的纹理融合

#### 3.2 双模式支持

**模式 1：向量融合（原方案优化版）**
```python
model = Stage2FusionCA(
    use_spatial_fusion=False,  # 使用通道注意力
    ...
)
```

**模式 2：空间特征融合（推荐）**
```python
model = Stage2FusionCA(
    use_spatial_fusion=True,   # 使用空间注意力
    ...
)
```

#### 3.3 特征对齐改进
- **全局特征**：`FeatureAlign` (Linear + 可选BN)
- **局部特征（向量模式）**：`FeatureAlign` (Linear)
- **局部特征（空间模式）**：`ConvAlign2d` (1x1 Conv + BN)

#### 3.4 ArcFace 分类头集成
- 内置标准 ArcFace 实现
- 支持可调 margin 和 scale 参数
- 自动 L2 归一化

---

## 使用方法

### 完整两阶段训练流程

```python
import torch
import torch.nn as nn
from models.stage1 import EfficientViT, ConvNeXt
from models.stage2_improved import Stage2FusionCA

# ============================================
# Stage 1: 预训练两个分支（单独训练）
# ============================================

# 1.1 定义模型
vit_palm = EfficientViT(
    img_size=224, in_chans=1,  # 掌纹单通道
    embed_dim=[64, 128, 192],
    depth=[1, 2, 3]
)

vit_vein = EfficientViT(
    img_size=224, in_chans=1,  # 掌静脉单通道
    embed_dim=[64, 128, 192],
    depth=[1, 2, 3]
)

cnn_palm = ConvNeXt(
    in_chans=1,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768]
)

cnn_vein = ConvNeXt(
    in_chans=1,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768]
)

# 1.2 Stage1 训练（可以用简单的分类损失预训练）
# 训练 vit_palm, vit_vein, cnn_palm, cnn_vein...
# 保存预训练权重

# ============================================
# Stage 2: 多模态融合训练
# ============================================

# 2.1 加载 Stage1 预训练权重
vit_palm.load_state_dict(torch.load('vit_palm.pth'))
vit_vein.load_state_dict(torch.load('vit_vein.pth'))
cnn_palm.load_state_dict(torch.load('cnn_palm.pth'))
cnn_vein.load_state_dict(torch.load('cnn_vein.pth'))

# 2.2 创建 Stage2 融合模型（推荐空间融合模式）
fusion_model = Stage2FusionCA(
    in_dim_global_palm=192,    # ViT 输出维度
    in_dim_global_vein=192,
    in_dim_local_palm=768,     # ConvNeXt 输出维度
    in_dim_local_vein=768,
    out_dim_global=256,        # 对齐后维度
    out_dim_local=256,
    use_spatial_fusion=True,   # ⭐ 使用空间注意力融合
    final_l2norm=True,
    with_arcface=True,
    num_classes=100,           # 你的类别数
    arcface_s=64.0,
    arcface_m=0.50
)

# 2.3 完整前向传播
palm_img = torch.randn(8, 1, 224, 224)  # Batch=8
vein_img = torch.randn(8, 1, 224, 224)
labels = torch.randint(0, 100, (8,))

# Stage1: 特征提取
with torch.no_grad():  # 可选：冻结 Stage1
    palm_global = vit_palm(palm_img, pool=True)           # (8, 192)
    vein_global = vit_vein(vein_img, pool=True)           # (8, 192)
    palm_local  = cnn_palm(palm_img, return_spatial=True) # (8, 768, H, W)
    vein_local  = cnn_vein(vein_img, return_spatial=True) # (8, 768, H, W)

# Stage2: 融合与分类
logits, fused_feat, details = fusion_model(
    palm_global, vein_global,
    palm_local, vein_local,
    labels
)

# 2.4 计算损失
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()
```

---

## 训练策略建议

### 方案 A：两阶段独立训练（推荐初学者）

**Stage 1**：
```python
# 单独训练 4 个网络（可以用简单的分类任务）
# 目标：让每个分支学会提取有效特征
optimizer_vit_palm = Adam(vit_palm.parameters(), lr=1e-4)
optimizer_cnn_palm = Adam(cnn_palm.parameters(), lr=1e-4)
# ... 训练到收敛
```

**Stage 2**：
```python
# 冻结 Stage1，只训练融合层
for param in [*vit_palm.parameters(), *vit_vein.parameters(),
              *cnn_palm.parameters(), *cnn_vein.parameters()]:
    param.requires_grad = False

optimizer_fusion = Adam(fusion_model.parameters(), lr=1e-3)
# 训练融合模型
```

### 方案 B：端到端微调（推荐有经验者）

```python
# Stage1 预训练后，解冻并端到端微调
for param in [*vit_palm.parameters(), *vit_vein.parameters(),
              *cnn_palm.parameters(), *cnn_vein.parameters()]:
    param.requires_grad = True

# 使用差异化学习率
optimizer = Adam([
    {'params': fusion_model.parameters(), 'lr': 1e-3},  # 融合层高学习率
    {'params': vit_palm.parameters(), 'lr': 1e-5},      # Stage1 低学习率
    {'params': vit_vein.parameters(), 'lr': 1e-5},
    {'params': cnn_palm.parameters(), 'lr': 1e-5},
    {'params': cnn_vein.parameters(), 'lr': 1e-5}
])
```

---

## 关键改进点对比

| 特性 | 原代码 | 改进后 |
|------|--------|--------|
| **代码可读性** | ❌ 300+行注释代码 | ✅ 清理干净 |
| **局部特征** | ❌ 直接全局池化，丢失空间信息 | ✅ 支持保留空间特征图 |
| **融合方式** | ⚠️ 只有通道注意力 | ✅ 通道+空间双模式 |
| **灵活性** | ❌ 固定输出格式 | ✅ 可选输出向量/特征图 |
| **可视化** | ❌ 无中间结果 | ✅ 返回所有注意力权重 |
| **文档** | ❌ 缺少说明 | ✅ 完整注释+使用示例 |

---

## 消融实验建议

为了验证改进效果，建议进行以下对比实验：

1. **Baseline**: 原 `stage2.py`（通道注意力）
2. **Improved**: `stage2_improved.py` 空间注意力模式
3. **Ablation**:
   - 只用全局特征（关闭局部分支）
   - 只用局部特征（关闭全局分支）
   - 简单拼接 vs 注意力融合

---

## 可视化建议

```python
# 获取注意力权重进行可视化
logits, fused_feat, details = fusion_model(...)

# 全局特征权重
g_w_palm = details['global']['w_palm']  # (N, C)
g_w_vein = details['global']['w_vein']  # (N, C)

# 局部特征权重（空间模式下是空间图）
l_w_palm = details['local']['w_palm']   # (N, 1, H, W) 或 (N, C)
l_w_vein = details['local']['w_vein']   # (N, 1, H, W) 或 (N, C)

# 可视化哪些区域被更多关注
import matplotlib.pyplot as plt
plt.imshow(l_w_palm[0, 0].detach().cpu())  # 掌纹权重热力图
plt.title('Palm Attention Map')
plt.show()
```

---

## 常见问题

### Q1: 为什么要用空间注意力？
**A**: 掌纹和掌静脉的纹理是局部的、位置相关的。空间注意力可以让模型学习"在哪里更关注掌纹，在哪里更关注掌静脉"，比全局池化后再融合更精细。

### Q2: 如何选择 `use_spatial_fusion`？
**A**:
- 如果数据集较小（<5000张）：先用 `False`（向量模式）
- 如果数据集较大：推荐 `True`（空间模式），效果更好
- 可以两种都试，对比精度

### Q3: Stage1 的四个网络必须分开训练吗？
**A**: 不一定。可以：
- **独立训练**：4个网络单独预训练（更稳定）
- **同时训练**：掌纹和掌静脉共享权重（参数更少）
- **端到端**：直接训练整个流程（需要大数据集）

### Q4: ArcFace 参数如何调整？
**A**:
- `s` (scale): 建议 32-64，越大分类边界越硬
- `m` (margin): 建议 0.3-0.5，越大类间距离越大
- 数据集小时降低 `m`，避免过拟合

---

## 下一步工作建议

1. ✅ **已完成**：代码重构与架构改进
2. 🔲 **实现数据加载器**：读取掌纹+掌静脉图像对
3. 🔲 **实现完整训练脚本**：两阶段训练流程
4. 🔲 **添加评估指标**：精度、ROC、EER 等
5. 🔲 **实验对比**：消融实验验证改进效果
6. 🔲 **可视化工具**：注意力图可视化

---

## 文件结构

```
palm_vein_fusion/
├── models/
│   ├── stage1.py           # ✅ 已改进：ViT + ConvNeXt
│   ├── stage2.py           # 原版本（保留作对比）
│   └── stage2_improved.py  # ⭐ 新版本（推荐使用）
├── train_stage1.py         # 待实现
├── train_stage2.py         # 待实现
├── evaluate.py             # 待实现
└── README_improvements.md  # 本文档
```

---

如有任何问题，欢迎随时询问！🚀
