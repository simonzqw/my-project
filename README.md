# scERso (Single-cell Expression Response Specificity Optimizer)

这是一个简单的单细胞扰动特异性建模框架，采用 MLP 作为判别器（Discriminator），用于学习“扰动-细胞背景”的匹配关系。

## 项目结构
- `models/mlp.py`: 定义了 `SpecificityMLP` 模型，支持 RNA 特征与扰动 Embedding 的融合。
- `utils/dataset.py`: PyTorch `Dataset` 和 `DataLoader` 实现。
- `utils/synthetic_data.py`: 合成数据生成脚本，模拟 HVGs 和特异性生物学反应。
- `train.py`: 完整的训练、验证和评估流程。

## 快速开始
运行以下命令即可跑通整个流程：
```bash
python train.py
```

## 后续接入真实大规模数据集的建议

老师提到的 "pretrained dataset" 暗示了我们可以利用现有的单细胞大模型（Foundation Models）来提升性能。以下是接入建议：

### 1. 替换 RNA 输入为 Pretrained Embedding
目前的输入是原始基因表达量（HVGs）。你可以使用以下模型提取细胞层面的特征向量（Cell Embedding）：
- **scGPT / Geneformer**: 提取出的 Embedding 包含更丰富的生物学语义，能极大减轻 MLP 的学习负担。
- **接口修改**: 只需在 `train.py` 中修改 `n_genes` 为 Embedding 的维度（通常是 512 或 768），并将输入数据替换为大模型的输出。

### 2. 扰动特征的语义化
目前扰动是用简单的 `nn.Embedding` (ID 映射) 处理的。
- **改进**: 如果扰动是基因敲除，可以使用该基因在大模型中的 **Gene Embedding** 作为输入，而不是随机初始化的 ID 向量。这样模型可以学习到不同基因扰动之间的关联性。

### 3. 负采样策略 (Negative Sampling)
在老师给出的 `label=0` 样本构造中：
- **建议**: 在大规模数据集中，通过“随机洗牌（Shuffle）”扰动与细胞的配对来生成负样本。确保每个 Batch 中正负样本比例均衡，有助于稳定判别器的学习。

### 4. 损失函数优化
- 如果任务演变为预测“响应强度”而不仅仅是“有无响应”，可以将 `nn.BCELoss` 替换为 `nn.MSELoss`，并将 Label 设定为连续的响应分值。
