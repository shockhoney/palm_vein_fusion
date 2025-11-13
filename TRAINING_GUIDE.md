# 训练代码使用说明

## 📋 训练脚本改进总结

您的 `train.py` 已经完全重构并改进，现在支持：

✅ **完整的两阶段训练流程**
✅ **Stage1**: 分别预训练 4 个模型（ViT-Palm, ViT-Vein, CNN-Palm, CNN-Vein）
✅ **Stage2**: 使用改进的空间注意力融合模型 `Stage2FusionCA`
✅ **自动权重加载与保存**
✅ **TensorBoard 可视化支持**
✅ **Early Stopping 防止过拟合**
✅ **灵活的训练策略**（冻结/微调 Stage1）

---

## 🚀 快速开始

### 1. 检查数据路径

确保 `train.py` 中的数据路径正确：

```python
class Config:
    # Stage1 对比学习数据（CASIA）
    palm_dir1 = 'C:/Users/admin/Desktop/palm_vein_fusion/data/CASIA_dataset/vi'
    vein_dir1 = 'C:/Users/admin/Desktop/palm_vein_fusion/data/CASIA_dataset/ir'

    # Stage2 识别任务数据（PolyU）
    palm_dir2 = 'C:/Users/admin/Desktop/palm_vein_fusion/data/PolyU/NIR'
    vein_dir2 = 'C:/Users/admin/Desktop/palm_vein_fusion/data/PolyU/Red'
```

### 2. 运行训练

```bash
python train.py
```

训练会自动按顺序执行：
- **Stage1**: 预训练 4 个特征提取器（ViT×2 + CNN×2）
- **Stage2**: 训练多模态融合模型

### 3. 查看训练日志

使用 TensorBoard 实时监控：

```bash
tensorboard --logdir runs
```

---

## 📂 输出文件结构

训练完成后，`outputs/models/` 目录下会生成：

```
outputs/models/
├── vit_palm_phase1_best.pth    # Stage1: ViT 掌纹模型
├── vit_vein_phase1_best.pth    # Stage1: ViT 掌静脉模型
├── cnn_palm_phase1_best.pth    # Stage1: CNN 掌纹模型
├── cnn_vein_phase1_best.pth    # Stage1: CNN 掌静脉模型
└── stage2_best.pth              # Stage2: 融合模型（包含所有权重）
```

---

## ⚙️ 训练参数配置

### Stage1 参数（对比学习）

```python
class Config:
    p1_epochs = 1          # Stage1 训练轮数（建议 30-50）
    p1_batch = 8           # Batch size
    p1_lr = 1e-4           # 学习率
    p1_patience = 8        # Early stopping 耐心值
```

### Stage2 参数（融合识别）

```python
class Config:
    p2_epochs = 50         # Stage2 训练轮数
    p2_batch = 8           # Batch size
    p2_lr = 1e-4           # 融合层学习率
    p2_enc_lr = 1e-5       # Stage1 微调学习率（如果不冻结）
    p2_patience = 15       # Early stopping 耐心值
    num_classes = 100      # 类别数（根据数据集修改）
```

### 重要参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `p1_epochs` | 30-50 | Stage1 预训练轮数，数据集小时可减少 |
| `p2_epochs` | 50-100 | Stage2 融合训练轮数 |
| `num_classes` | 根据数据 | **必须修改**为你的数据集类别数 |
| `freeze_stage1` (代码中) | `True` | `True`=冻结Stage1，`False`=端到端微调 |
| `use_spatial_fusion` (代码中) | `True` | **推荐开启**空间注意力融合 |

---

## 🎯 训练策略选择

### 方案 A：冻结 Stage1（推荐初学者）

**适用场景**：
- 数据集较小（< 5000 张）
- 显存不足
- 想快速验证 Stage2 效果

**设置方法**：
在 `train_phase2()` 函数中：
```python
freeze_stage1 = True  # 只训练融合层
```

**优点**：
- 训练速度快
- 显存占用小
- 不容易过拟合

**缺点**：
- Stage1 特征不会针对任务优化

---

### 方案 B：端到端微调（推荐有经验者）

**适用场景**：
- 数据集较大（> 5000 张）
- 显存充足
- 追求最佳性能

**设置方法**：
在 `train_phase2()` 函数中：
```python
freeze_stage1 = False  # 端到��微调
```

**优点**：
- 特征提取器会针对任务优化
- 理论上性能更好

**缺点**：
- 训练时间长
- 需要更多显存
- 可能过拟合

---

## 🔧 跳过 Stage1（如果已训练）

如果您已经训练过 Stage1，可以直接跳到 Stage2：

在 `main()` 函数中设置：
```python
skip_stage1 = True  # 跳过 Stage1，直接加载权重
```

这样会直接加载已保存的 Stage1 权重，进入 Stage2 训练。

---

## 📊 监控训练过程

### TensorBoard 可视化

启动 TensorBoard：
```bash
tensorboard --logdir runs
```

然后在浏览器打开 `http://localhost:6006`

**可查看指标**：
- `Phase1_xxx/BatchLoss`: Stage1 各模型的 batch loss
- `Phase1_xxx/EpochLoss`: Stage1 各模型的 epoch loss
- `Phase2/TrainLoss`, `Phase2/TrainAcc`: Stage2 训练损失和精度
- `Phase2/ValLoss`, `Phase2/ValAcc`: Stage2 验证损失和精度
- `Phase2/LearningRate`: 学习率变化

---

## 🐛 常见问题与解决

### 问题 1: 显存不足 (CUDA Out of Memory)

**解决方案**：
```python
# 在 Config 中减小 batch size
p1_batch = 4  # 从 8 改为 4
p2_batch = 4
```

或者在 `train_phase2()` 中设置：
```python
freeze_stage1 = True  # 冻结 Stage1 减少显存占用
```

---

### 问题 2: 数据集类别数不匹配

**错误信息**：
```
RuntimeError: dimension out of range
```

**解决方案**：
修改 `Config` 中的 `num_classes`：
```python
num_classes = 你的实际类别数  # 例如 50, 100, 200
```

---

### 问题 3: Stage1 预训练效果不好

**症状**：
- Triplet loss 不下降
- d(a,p) 和 d(a,n) 差距不明显

**解决方案**：
1. 增加训练轮数：
   ```python
   p1_epochs = 50  # 从 30 增加到 50
   ```

2. 调整 Triplet margin：
   在 `train_phase1_vit()` 和 `train_phase1_cnn()` 中：
   ```python
   criterion = TripletLoss(margin=1.0)  # 从 0.5 增加到 1.0
   ```

3. 检查数据增强是否过强：
   ```python
   get_transforms(strong=False)  # 使用弱增强
   ```

---

### 问题 4: Stage2 训练精度低

**解决方案**：

1. **确保 Stage1 已充分训练**
   ```python
   # 检查 Stage1 checkpoint 是否存在
   ls outputs/models/
   ```

2. **尝试端到端微调**
   ```python
   freeze_stage1 = False  # 允许 Stage1 参数更新
   ```

3. **调整学习率**
   ```python
   p2_lr = 5e-4  # 增大融合层学习率
   p2_enc_lr = 1e-6  # 降低编码器学习率
   ```

4. **使用空间注意力融合**（已默认开启）
   确认 `train_phase2()` 中：
   ```python
   use_spatial_fusion=True  # ✓ 确保为 True
   ```

---

## 📈 预期训练时间

基于单张 NVIDIA RTX 3090（24GB）的估算：

| 阶段 | 数据量 | Batch=8 | Batch=4 |
|------|--------|---------|--------|
| **Stage1 单模型** | 10k 张 | ~30 分钟/epoch | ~45 分钟/epoch |
| **Stage1 全部（4个）** | 10k 张 | ~2 小时 | ~3 小时 |
| **Stage2 (冻结)** | 5k 张 | ~15 分钟/epoch | ~25 分钟/epoch |
| **Stage2 (微调)** | 5k 张 | ~30 分钟/epoch | ~50 分钟/epoch |

**完整训练（Stage1 + Stage2）**：
- 冻结 Stage1 策略：约 **3-5 小时**
- 端到端微调策略：约 **6-10 小时**

---

## 🎓 下一步建议

### 1. 验证数据加载

在运行完整训练前，先测试数据加载：

```python
from utils.dataset import ContrastDataset, PairDataset
import torchvision.transforms as transforms

# 测试 Stage1 数据集
dataset1 = ContrastDataset(config.palm_dir1, config.vein_dir1, get_transforms())
print(f"Stage1 dataset size: {len(dataset1)}")
anchor, pos, neg, _ = dataset1[0]
print(f"Image shapes: {anchor.shape}, {pos.shape}, {neg.shape}")

# 测试 Stage2 数据集
dataset2 = PairDataset(config.palm_dir2, config.vein_dir2, get_transforms(), split='train')
print(f"Stage2 train size: {len(dataset2)}")
palm, vein, label = dataset2[0]
print(f"Pair shapes: {palm.shape}, {vein.shape}, label: {label}")
```

### 2. 测试单个模型前向传播

```python
import torch
from models.stage1 import EfficientViT, ConvNeXt

# 测试 ViT
vit = EfficientViT(img_size=224, in_chans=1).cuda()
x = torch.randn(2, 1, 224, 224).cuda()
out = vit(x, pool=True)
print(f"ViT output: {out.shape}")  # 应该是 (2, 192)

# 测试 CNN
cnn = ConvNeXt(in_chans=1).cuda()
out_vec = cnn(x, return_spatial=False)
out_spatial = cnn(x, return_spatial=True)
print(f"CNN vector: {out_vec.shape}")  # (2, 768)
print(f"CNN spatial: {out_spatial.shape}")  # (2, 768, H, W)
```

### 3. 运行完整训练

确认一切正常后：
```bash
python train.py
```

### 4. 评估与分析

训练完成后，可以：
- 在验证集上评估性能
- 可视化注意力权重
- 进行消融实验（对比不同融合策略）

---

## 📞 需要帮助？

如果遇到问题，请检查：
1. ✅ 数据路径是否正确
2. ✅ `num_classes` 是否匹配数据集
3. ✅ 显存是否充足（可以降低 batch size）
4. ✅ 依赖包是否安装完整（`torch`, `timm`, `tqdm` 等）

---

**祝训练顺利！🚀**
