# 生物识别指标使用说明

## 📊 已集成的指标

您的项目现在已经完整集成了生物识别领域的标准评估指标：

### 核心指标

1. **EER (Equal Error Rate)** - 等错误率
   - **定义**: FAR (False Accept Rate) = FRR (False Reject Rate) 时的错误率
   - **范围**: [0, 1]，越低越好
   - **意义**: 系统在假接受和假拒绝之间的平衡点
   - **典型值**: 优秀系统 < 1%

2. **AUC (Area Under Curve)** - ROC 曲线下面积
   - **定义**: ROC 曲线下的面积
   - **范围**: [0, 1]，越高越好
   - **意义**: 系统整体性能的综合指标
   - **典型值**: 优秀系统 > 0.99

3. **TAR @ FAR** - 指定 FAR 下的 TAR
   - **定义**: 在固定 FAR 下的 True Accept Rate
   - **常用 FAR**: 0.1%, 1%, 10%
   - **意义**: 实际应用中的性能（如安全性要求高时设置 FAR=0.1%）

### 辅助指标

4. **FAR (False Accept Rate)** - 假接受率
   - 冒名者被错误接受的比例

5. **FRR (False Reject Rate)** - 假拒绝率
   - 真实用户被错误拒绝的比例

6. **ACC (Accuracy)** - 准确率
   - 在某个阈值下的分类准确率

---

## 🔧 使用方式

### 1. 训练阶段（train.py）

训练过程中会**自动计算**生物识别指标：

```bash
python train.py
```

**计算时机**：
- 每 5 个 epoch 计算一次（可修改 `train.py` 中的 `if epoch % 5 == 0`）
- 最后一个 epoch 也会计算
- 为了节省时间，只使用 300 个样本（可通过 `max_samples` 参数调整）

**输出示例**：
```
Epoch 10/50:
  Train Loss: 0.8234, Train Acc: 85.32%
  Val Loss: 0.9123, Val Acc: 82.15%
  EER: 0.0234, AUC: 0.9876
  TAR@1%FAR: 0.9512
  LR: 0.000050
```

**TensorBoard 可视化**：
```bash
tensorboard --logdir runs
```

在 TensorBoard 中可以查看：
- `Phase2/EER`: EER 曲线
- `Phase2/AUC`: AUC 曲线
- `Phase2/TAR@1%FAR`: TAR@1%FAR 曲线
- `Phase2/GenuineMean`: Genuine 分数均值
- `Phase2/ImpostorMean`: Impostor 分数均值

---

### 2. 测试阶段（test.py）

完整的测试评估：

```bash
python test.py
```

**测试流程**：
1. 加载训练好的模型
2. 提取测试集特征向量
3. 计算所有 genuine 和 impostor 配对
4. 计算完整的生物识别指标
5. 生成可视化图表（ROC 曲线、分数分布）
6. 保存详细结果

**输出示例**：
```
======================================================================
测试结果
======================================================================

【核心指标】
  EER (Equal Error Rate):        0.0123 (1.23%)
  EER Threshold:                 0.8234
  AUC (Area Under Curve):        0.9945

【EER 阈值处的性能】
  FAR (False Accept Rate):       0.0125
  FRR (False Reject Rate):       0.0121
  ACC (Accuracy):                0.9877 (98.77%)

【不同 FAR 下的 TAR (True Accept Rate)】
FAR          | TAR          | Threshold
----------------------------------------
0.00001      | 0.6234       | 0.9512
0.00010      | 0.8123       | 0.9123
0.00100      | 0.9234       | 0.8512
0.01000      | 0.9812       | 0.7923
0.10000      | 0.9956       | 0.6234
```

**输出文件**：
- `outputs/test_results/test_results.txt`: 详细测试结果
- `outputs/test_results/roc_curve.png`: ROC 曲线图
- `outputs/test_results/score_distribution.png`: 分数分布图

---

## 📈 指标解读

### 如何评价系统性能？

| 指标 | 优秀 | 良好 | 一般 | 较差 |
|------|------|------|------|------|
| **EER** | < 1% | 1-3% | 3-5% | > 5% |
| **AUC** | > 0.99 | 0.95-0.99 | 0.90-0.95 | < 0.90 |
| **TAR@1%FAR** | > 95% | 85-95% | 70-85% | < 70% |

### 实际应用场景

**高安全场景（如银行、军事）**：
- 关注 **TAR@0.01%FAR** 或更低
- 要求 FAR 极低，可以容忍较高的 FRR

**便捷场景（如手机解锁）**：
- 关注 **TAR@1%FAR**
- 平衡用户体验和安全性

**大规模识别（如机场）**：
- 关注 **EER** 和 **AUC**
- 需要整体性能优秀

---

## 🔍 分数分析

### Genuine vs Impostor 分数

**理想情况**：
```
Genuine 均值: 0.95 ± 0.03
Impostor 均值: 0.15 ± 0.08
```

- **分数分离良好**: 两个分布不重叠或重叠极少 → EER 低
- **分数重叠严重**: 两个分布高度重叠 → EER 高

### 可视化分析

**ROC 曲线**：
- 曲线越接近左上角越好
- AUC 越大越好

**分数分布图**：
- Genuine（绿色）应该集中在高分区
- Impostor（红色）应该集中在低分区
- 两者交叉越少越好

---

## ⚙️ 自定义配置

### 修改测试参数

在 `test.py` 中：

```python
class Config:
    # ... 其他配置
    batch_size = 16          # 批次大小（根据显存调整）
    save_roc_curve = True    # 是否保存 ROC 曲线
    output_dir = 'outputs/test_results'  # 输出目录
```

### 修改训练中的指标计算频率

在 `train.py` 的 `train_phase2()` 函数中：

```python
# 每 5 个 epoch 计算一次 → 改为每 3 个 epoch
if epoch % 3 == 0 or epoch == config.p2_epochs - 1:
    ...

# 使用更多样本（更准确但更慢）
bio_metrics = compute_biometric_metrics(
    ...
    max_samples=500  # 从 300 改为 500
)
```

### 添加更多 FAR 阈值

在 `test.py` 或 `train.py` 中：

```python
target_fars = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1]  # 添加 20%
```

---

## 📝 指标计算原理

### 1. 特征提取

对每个样本提取融合特征向量：
```
特征向量 = Stage2_Fusion(ViT_global, CNN_local)
维度: (N, 512)  # out_dim_global + out_dim_local
```

### 2. 相似度计算

使用**余弦相似度**：
```python
similarity = cosine_similarity(feature1, feature2)
范围: [-1, 1]，通常在 [0, 1]
```

### 3. 配对生成

- **Genuine pairs**: 同一身份的所有组合
- **Impostor pairs**: 不同身份的采样组合

### 4. EER 计算

找到 FAR = FRR 的阈值点，使用线性插值获得精确 EER。

---

## 🐛 常见问题

### Q1: 训练时指标计算很慢？

**A**: 减少 `max_samples` 或增加计算间隔：
```python
# 从每 5 epoch 改为每 10 epoch
if epoch % 10 == 0:
    ...
```

### Q2: EER 很高（> 10%）怎么办？

**A**: 可能的原因：
1. 模型未充分训练
2. Stage1 特征提取器质量不佳
3. 数据质量问题
4. 类别数太多/样本太少

**解决方案**：
- 延长 Stage1 训练
- 使用端到端微调 (`freeze_stage1=False`)
- 检查数据预处理

### Q3: Genuine 和 Impostor 分数分布重叠？

**A**: 说明特征判别性不足：
- 增强数据增强
- 调整 ArcFace margin (增大 `m`)
- 使用空间注意力融合 (`use_spatial_fusion=True`)

### Q4: 测试时显存不足？

**A**: 减小 batch size：
```python
config.batch_size = 8  # 从 16 改为 8
```

---

## 📚 参考文献

1. **EER**: ISO/IEC 19795-1 - Biometric Performance Testing
2. **ArcFace**: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
3. **ROC Analysis**: "An introduction to ROC analysis" (Pattern Recognition Letters, 2006)

---

## ✅ 检查清单

训练前：
- [ ] 确认 `utils/metrics.py` 存在且可导入
- [ ] 安装 `sklearn`: `pip install scikit-learn`
- [ ] 确认数据路径配置正确

训练中：
- [ ] 观察 TensorBoard 中的 EER/AUC 曲线
- [ ] 检查 Genuine/Impostor 均值是否分离

测试前：
- [ ] 确认模型权重文件存在
- [ ] 修改 `test.py` 中的数据路径

测试后：
- [ ] 检查 `outputs/test_results/` 下的输出文件
- [ ] 分析 ROC 曲线和分数分布图
- [ ] 对比不同模型/配置的指标

---

**祝测试顺利！如有任何问题，请检查输出日志或查看生成的可视化图表。**
