# 损失函数设计说明

## 📊 损失函数改进总览

原损失函数是为**特征分解架构**设计的，不适合您的 **Stage2FusionCA（注意力融合）**架构。

新损失函数专门为您的项目设计，包含：

✅ **Stage1**: 改进的 Triplet Loss（支持困难样本挖掘）
✅ **Stage2**: 多任务损失函数（4种损失模式可选）
✅ **灵活配置**: 一键切换不同损失组合

---

## 🎯 Stage1 损失函数：TripletLoss

### 原理

三元组损失用于对比学习，目标是：
- **拉近** anchor 和 positive（同一身份）
- **推远** anchor 和 negative（不同身份）

```python
Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

### 改进点

1. **困难样本挖掘** (`mining='hard'`)
   - 只关注难以区分的样本
   - 加速收敛，提升性能

2. **余弦距离**
   - 使用 `1 - cosine_similarity`
   - 适合高维特征空间

### 使用方法

```python
from utils.loss import TripletLoss

# 创建损失函数
criterion = TripletLoss(
    margin=0.5,      # margin 越大，正负样本分离越远
    mining='hard'    # 'none' 或 'hard'
)

# 训练时使用
loss, dist_ap, dist_an = criterion(feat_anchor, feat_positive, feat_negative)
```

### 参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **margin** | 0.3-1.0 | 数据集难度高→大margin，简单→小margin |
| **mining** | 'hard' | 推荐使用，加速收敛 |

---

## 🔥 Stage2 损失函数：Stage2FusionLoss

### 设计理念

您的 Stage2FusionCA 使用**注意力机制融合**两个模态（掌纹+掌静脉），需要损失函数：

1. ✅ **识别准确** - 分类损失
2. ✅ **模态互补** - 确保两个模态都被有效利用
3. ✅ **注意力合理** - 避免注意力退化（只关注一个模态）
4. ✅ **类内紧凑** - 同类样本特征相似

### 包含的损失项

#### 1. 分类损失（Classification Loss）

**必选项** - 核心损失

```python
CE Loss = CrossEntropy(logits, labels)
```

- 使用 **Label Smoothing**（0.1）防止过拟合
- 配合 ArcFace 时效果最佳

#### 2. 模态一致性损失（Modality Consistency Loss）

**推荐项** - 确保全局和局部特征一致

```python
Consistency = ContrastiveLoss(global_fused, local_fused)
```

**作用**：
- 全局特征（ViT）和局部特征（CNN）应该对同一样本给出一致的判断
- 使用对比学习思想，同一样本的全局/局部特征相似度最高

**权重建议**：`w_consistency = 0.1`

#### 3. 注意力正则化损失（Attention Regularization Loss）

**可选项** - 防止注意力退化

```python
Attention = Balance_Loss + Diversity_Loss
```

**Balance Loss**：鼓励掌纹和掌静脉权重接近 0.5（平衡）
```python
balance = (mean_palm - 0.5)^2 + (mean_vein - 0.5)^2
```

**Diversity Loss**：鼓励注意力权重有差异性（避免所有位置权重相同）
```python
diversity = exp(-var_palm) + exp(-var_vein)
```

**权重建议**：`w_attention = 0.05`

#### 4. Center Loss

**高级选项** - 提升类内紧凑性

```python
Center = ||features - class_centers||^2
```

**作用**：
- 学习每个类别的中心点
- 拉近同类样本，配合 ArcFace 效果更好

**权重建议**：`w_center = 0.01`

---

## 📋 4种损失模式

### Mode 1: Simple（简单模式）

**适用场景**：
- 初始调试
- 数据集很小（< 1000 样本）
- 快速验证

**配置**：
```python
criterion = get_stage2_loss(
    num_classes=100,
    feat_dim=512,
    mode='simple'
)
# w_cls=1.0, 其余=0
```

**优点**：训练稳定，速度快
**缺点**：可能欠拟合，性能中等

---

### Mode 2: Standard（标准模式）⭐ 推荐

**适用场景**：
- 大部分情况的首选
- 数据集适中（1000-10000 样本）
- 平衡性能和稳定性

**配置**：
```python
criterion = get_stage2_loss(
    num_classes=100,
    feat_dim=512,
    mode='standard'
)
# w_cls=1.0, w_consistency=0.1, 其余=0
```

**包含**：
- ✅ 分类损失
- ✅ 模态一致性损失

**优点**：性能好，训练稳定
**缺点**：无

---

### Mode 3: Full（完整模式）

**适用场景**：
- 追求更高性能
- 数据集较大（> 5000 样本）
- 注意力机制重要

**配置**：
```python
criterion = get_stage2_loss(
    num_classes=100,
    feat_dim=512,
    mode='full'
)
# w_cls=1.0, w_consistency=0.1, w_attention=0.05, 其余=0
```

**包含**：
- ✅ 分类损失
- ✅ 模态一致性损失
- ✅ 注意力正则化损失

**优点**：性能最佳，注意力更合理
**缺点**：训练稍慢，需要调参

---

### Mode 4: Advanced（高级模式）

**适用场景**：
- 竞赛/论文，追求极致性能
- 数据集很大（> 10000 样本）
- 类别数很多（> 100）

**配置**：
```python
criterion = get_stage2_loss(
    num_classes=100,
    feat_dim=512,
    mode='advanced'
)
# w_cls=1.0, w_consistency=0.1, w_attention=0.05, w_center=0.01
```

**包含**：
- ✅ 分类损失
- ✅ 模态一致性损失
- ✅ 注意力正则化损失
- ✅ Center Loss

**优点**：性能极致，类内紧凑性好
**缺点**：训练慢，内存占用大，需要仔细调参

---

## 🔧 在训练中使用

### 修改 train.py

已经自动集成！只需修改模式：

```python
# 在 train_phase2() 函数中找到这一行：
criterion = get_stage2_loss(
    num_classes=config.num_classes,
    feat_dim=512,
    mode='standard'  # ← 修改这里
).to(device)

# 可选值：'simple', 'standard', 'full', 'advanced'
```

### 训练输出

新损失函数会显示每个损失项的详细信息：

```
P2 Epoch 5/50: 100%|████| loss=0.8234(cls=0.7123,cons=0.0956,attn=0.0155),acc=85.32%
```

- `cls`: 分类损失
- `cons`: 一致性损失
- `attn`: 注意力正则化损失
- `cent`: Center Loss（如果启用）

---

## 📊 性能对比（预期）

| 模式 | EER | 训练时间 | 显存 | 推荐场景 |
|------|-----|----------|------|----------|
| **Simple** | ~3% | 1x | 1x | 调试、小数据集 |
| **Standard** ⭐ | ~2% | 1.1x | 1.05x | 通用 |
| **Full** | ~1.5% | 1.2x | 1.1x | 追求性能 |
| **Advanced** | ~1% | 1.3x | 1.2x | 竞赛/论文 |

*以 Simple 模式为基准

---

## ⚙️ 高级配置（自定义）

如果4种模式都不满足需求，可以自定义：

```python
from utils.loss import Stage2FusionLoss

criterion = Stage2FusionLoss(
    num_classes=100,
    feat_dim=512,
    w_cls=1.0,           # 分类损失权重
    w_consistency=0.15,  # 一致性损失权重（可调）
    w_attention=0.03,    # 注意力损失权重（可调）
    w_center=0.005,      # Center Loss权重（可调）
    label_smoothing=0.1  # Label smoothing系数
).to(device)
```

### 调参建议

1. **先从 Simple 开始**，确保模型能正常训练
2. **升级到 Standard**，通常能获得最佳性价比
3. **如果性能不足**，尝试 Full 模式
4. **如果还不够**，尝试 Advanced 或自定义权重

### 权重调整原则

- **分类损失** (`w_cls`)：固定为 1.0
- **一致性损失** (`w_consistency`)：0.05-0.2，越大越强调模态互补
- **注意力损失** (`w_attention`)：0.01-0.1，越大越防止注意力退化
- **Center Loss** (`w_center`)：0.005-0.02，越大类内越紧凑但可能过拟合

---

## 🐛 常见问题

### Q1: 训练时 loss 不下降？

**可能原因**：
- 损失权重设置不合理
- 学习率过大或过小

**解决方案**：
1. 先用 `mode='simple'` 验证基础功能
2. 降低学习率：`lr = 5e-5`
3. 逐步添加损失项，观察变化

---

### Q2: 一致性损失很大（> 1.0）？

**原因**：全局和局部特征差异太大

**解决方案**：
1. 降低 `w_consistency`：从 0.1 改为 0.05
2. 延长 Stage1 训练，提升特征质量
3. 检查 Stage1 权重是否正确加载

---

### Q3: 注意力损失不下降？

**原因**：注意力权重已经退化（总是偏向某一模态）

**解决方案**：
1. 增大 `w_attention`：从 0.05 改为 0.1
2. 检查两个模态的数据质量是否差异很大
3. 考虑数据增强，平衡两个模态的难度

---

### Q4: Center Loss 导致过拟合？

**现象**：训练集精度高，验证集精度低

**解决方案**：
1. 降低 `w_center`：从 0.01 改为 0.005
2. 增加 dropout
3. 使用更强的数据增强
4. 或者直接不用 Center Loss（mode='full'）

---

## 📈 实验建议

### 渐进式实验

```
第1轮：mode='simple'
  → 验证模型基础功能
  → 记录 baseline 性能

第2轮：mode='standard'
  → 通常能提升 20-30% 性能
  → 观察一致性损失的变化

第3轮：mode='full'（可选）
  → 如果需要更高性能
  → 观察注意力权重分布

第4轮：mode='advanced'（可选）
  → 追求极致性能
  → 小心过拟合
```

### 消融实验

对比不同损失项的贡献：

| 实验 | 配置 | 预期效果 |
|------|------|----------|
| 1 | Simple | Baseline |
| 2 | + Consistency | +20-30% |
| 3 | + Attention | +5-10% |
| 4 | + Center | +2-5% |

---

## 📚 理论参考

1. **Triplet Loss**: "FaceNet: A Unified Embedding for Face Recognition" (CVPR 2015)
2. **ArcFace**: "ArcFace: Additive Angular Margin Loss" (CVPR 2019)
3. **Center Loss**: "A Discriminative Feature Learning Approach" (ECCV 2016)
4. **Contrastive Learning**: "A Simple Framework for Contrastive Learning" (ICML 2020)

---

## ✅ 使用检查清单

训练前：
- [ ] 确认使用了新的 `utils/loss.py`
- [ ] 在 `train.py` 中使用了 `get_stage2_loss()`
- [ ] 选择了合适的 `mode`
- [ ] `feat_dim` 设置正确（= out_dim_global + out_dim_local）

训练中：
- [ ] 观察各个损失项的数值
- [ ] 检查分类损失是否下降
- [ ] 检查一致性损失是否合理（< 0.5）
- [ ] 观察注意力权重是否平衡

训练后：
- [ ] 对比不同模式的性能
- [ ] 进行消融实验
- [ ] 分析注意力权重分布

---

**总结**：新损失函数专门为您的注意力融合架构设计，支持4种模式快速切换。推荐从 `mode='standard'` 开始，根据需要逐步升级。

**祝训练顺利！🚀**
