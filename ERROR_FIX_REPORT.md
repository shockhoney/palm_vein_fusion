# 训练错误修复报告

## 🔴 错误信息

```
RuntimeError: CUDA error: device-side assert triggered
Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed
```

发生位置：`utils/loss.py:326` → `F.cross_entropy(logits, labels)`

---

## 🎯 问题根源

### 标签越界错误

**问题描述**：
配置文件中硬编码 `num_classes=100`，但实际数据集的类别数可能 > 100，导致标签超出范围。

### 触发条件

```python
# train.py Line 28
num_classes = 100  # 固定值

# 但 PairDataset 会加载所有子文件夹
# 如果 data/PolyU/NIR/ 下有 120 个子文件夹
# 那么 labels 的范围是 [0, 119]
# 但 CrossEntropyLoss 期望 labels ∈ [0, 99]
# → 越界错误！
```

### 为什么会发生

1. **PairDataset.py** (Line 118-134):
   ```python
   for idx, cls in enumerate(sorted(os.listdir(self.palm_dir))):
       # idx 会从 0 递增到 文件夹数量-1
       # 如果有120个文件夹，idx 最大为119
       labels.append(idx)
   ```

2. **loss.py** (Line 326):
   ```python
   cls_loss = self.ce_loss(logits, labels)
   # CrossEntropyLoss 要求：labels < num_classes
   # logits.shape = (batch_size, 100)
   # labels 可能包含 119 → 越界！
   ```

---

## ✅ 解决方案

### 修改内容

**修改文件**: `train.py`

#### 1. 动态获取类别数

```python
# 修改前 (错误)
train_ds = PairDataset(...)
fusion_model = Stage2FusionCA(
    num_classes=config.num_classes  # 固定100
)
criterion = get_stage2_loss(
    num_classes=config.num_classes  # 固定100
)

# 修改后 (正确)
train_ds = PairDataset(...)
actual_num_classes = train_ds.num_classes  # 从数据集获取
num_classes_for_model = actual_num_classes  # 使用实际值

fusion_model = Stage2FusionCA(
    num_classes=num_classes_for_model  # 动态值
)
criterion = get_stage2_loss(
    num_classes=num_classes_for_model  # 动态值
)
```

#### 2. 添加检查和警告

```python
if actual_num_classes != config.num_classes:
    print(f"⚠ Warning: Config num_classes={config.num_classes}, but dataset has {actual_num_classes} classes")
    print(f"  Using actual_num_classes={actual_num_classes}")
```

---

## 🔍 验证方法

### 训练前检查

```python
# 添加到 train_phase2 开始处
print(f"✓ Dataset loaded: {len(train_ds)} train, {len(val_ds)} val, {actual_num_classes} classes")
print(f"  Model num_classes: {num_classes_for_model}")
print(f"  Label range: [{train_ds.labels.min().item()}, {train_ds.labels.max().item()}]")
assert train_ds.labels.max().item() < num_classes_for_model, "Labels exceed num_classes!"
```

### 手动验证数据集

```bash
# 检查数据集类别数
cd data/PolyU/NIR
ls -d */ | wc -l  # Linux/Mac
# 或
dir /b /ad | find /c /v ""  # Windows
```

---

## 📊 修复前后对比

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| **num_classes来源** | 配置文件硬编码 | 数据集动态获取 |
| **类别数匹配** | ❌ 可能不匹配 | ✅ 自动匹配 |
| **错误提示** | ❌ CUDA assert（难懂） | ✅ 明确警告信息 |
| **灵活性** | ❌ 需手动修改config | ✅ 自动适应数据 |

---

## 🎯 其他潜在问题

### 问题1：数据集结构不符合预期

**PairDataset 期望结构**：
```
data/PolyU/NIR/
  ├── 001/       # 每个类别一个文件夹
  │   ├── img1.jpg
  │   └── img2.jpg
  ├── 002/
  └── ...
```

**如果结构错误**：
```
data/PolyU/NIR/
  ├── img1.jpg   # 扁平结构（错误！）
  ├── img2.jpg
  └── ...
```
→ 会导致类别数 = 0 或其他错误

### 问题2：train/val 划分边界情况

```python
# dataset.py Line 128-129
split_idx = int(num_samples * 0.8)
start, end = (0, split_idx) if split == 'train' else (split_idx, num_samples)
```

如果某个类别只有1个样本：
- `split_idx = int(1 * 0.8) = 0`
- train: `[0, 0)` → 空！
- val: `[0, 1)` → 1个样本

**建议**：在 PairDataset 中添加最小样本数检查

---

## ✅ 最终检查清单

在重新训练前确认：

- [x] 修改了 `train_phase2` 使用 `num_classes_for_model`
- [x] 模型创建使用动态类别数
- [x] 损失函数使用动态类别数
- [ ] 验证数据集目录结构正确（子文件夹格式）
- [ ] 确认每个类别至少有2个样本（保证train/val都有数据）
- [ ] 运行前先打印类别数信息

---

## 🚀 重新训练

```bash
python train.py
```

**预期输出**：
```
Loading Stage1 pretrained weights...
✓ Loaded vit_palm
✓ Loaded vit_vein
✓ Loaded cnn_palm
✓ Loaded cnn_vein
✓ Dataset loaded: 4500 train, 1125 val, 150 classes  # 会显示实际类别数
⚠ Warning: Config num_classes=100, but dataset has 150 classes
  Using actual_num_classes=150
...
```

现在应该可以正常训练了！🎉
