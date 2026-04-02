# scERso (Single-cell Expression Response Specificity Optimizer)

用于单细胞扰动预测的训练与评估仓库，当前包含两条主线：

1. **Transformer 主体的生成式预测器**（`train.py`）  
2. **Diffusion 方向实验分支**（`train_diffusion.py`）

当前默认推荐先跑 `train.py` 主线，再用 `evaluate_metrics.py` 与 `visualize.py` 进行评估与可视化。

---

## 1. 项目结构

- `train.py`：主训练脚本（Transformer predictor + EMA/AMP/断点续训/滚动 checkpoint）。
- `evaluate_metrics.py`：离线评估脚本，输出 single-cell 与 perturbation-level 指标 JSON。
- `visualize.py`：测试集可视化与报告图生成。
- `train_diffusion.py`：Diffusion 分支训练脚本（可选方向，非当前主线）。
- `models/reasoning_mlp.py`：当前主模型 `PerturbationPredictor`（Transformer 主体）。
- `models/scerso_diffusion.py`：Diffusion 模型定义。
- `utils/data_processor.py`：h5ad 数据读取、划分、control 采样、dose 处理。
- `utils/emb_loader.py`：预训练基因向量（Gene2Vec）读取与对齐。

---

## 2. 依赖与数据

### 2.1 依赖（建议）
- Python 3.8+
- PyTorch
- scanpy / anndata
- numpy / scipy / pandas
- scikit-learn
- matplotlib / seaborn
- rdkit（仅当需要 SMILES 药物特征时）

### 2.2 数据要求
输入为 `.h5ad` 文件，`DataProcessor` 会尝试自动适配 Adamson/SCI-Plex 等常见列名。  
核心字段包括：
- 扰动：`perturbation`（或可映射字段）
- 细胞背景：`cell_line`（或 `source_batch`）
- 剂量（可选）：`dose`
- 药物特征（可选）：`smiles`

---

## 3. 主线训练（Transformer）

> 如果 `--split_strategy perturbation`（zero-shot 按扰动划分），建议并通常需要提供 `--pretrained_emb` 以提升未见扰动泛化。

```bash
export OMP_NUM_THREADS=1

python train.py \
  --data_path "/path/to/perturb_processed.h5ad" \
  --pretrained_emb "/path/to/gene2vec_dim_200_iter_9.txt" \
  --save_dir "./checkpoints_adamson_tf" \
  --split_strategy perturbation \
  --batch_size 512 \
  --epochs 50 \
  --lr 1e-4 \
  --num_workers 4 \
  --amp \
  --ema_decay 0.999 \
  --eval_ema \
  --keep_last_n 5 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 1024 \
  --n_ctrl_tokens 8
```

### 3.1 断点续训
```bash
python train.py \
  --data_path "/path/to/perturb_processed.h5ad" \
  --pretrained_emb "/path/to/gene2vec_dim_200_iter_9.txt" \
  --save_dir "./checkpoints_adamson_tf" \
  --split_strategy perturbation \
  --batch_size 512 \
  --epochs 50 \
  --lr 1e-4 \
  --num_workers 4 \
  --amp \
  --ema_decay 0.999 \
  --eval_ema \
  --keep_last_n 5 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 1024 \
  --n_ctrl_tokens 8 \
  --resume_path "./checkpoints_adamson_tf/latest.pth"
```

---

## 4. 评估（推荐）

```bash
export OMP_NUM_THREADS=1

python evaluate_metrics.py \
  --data_path "/path/to/perturb_processed.h5ad" \
  --model_path "./checkpoints_adamson_tf/best_model.pth" \
  --split_strategy perturbation \
  --batch_size 256 \
  --num_workers 0 \
  --use_ema \
  --de_mode quantile \
  --de_quantile 0.9 \
  --output_json "./checkpoints_adamson_tf/test_metrics_q90.json"
```

`evaluate_metrics.py` 当前会同时输出：
- **single-cell 口径**（`test_unseen_single_*`）
- **perturbation-level 口径**（`test_unseen_perturb_*`）

建议在正式对比中同时报告二者，并显式记录 `metric_config`。

---

## 5. 可视化

```bash
export OMP_NUM_THREADS=1

python visualize.py \
  --data_path "/path/to/perturb_processed.h5ad" \
  --model_path "./checkpoints_adamson_tf/best_model.pth" \
  --save_path "./checkpoints_adamson_tf/v9_report.png" \
  --split_strategy perturbation \
  --batch_size 512 \
  --top_n 50 \
  --heatmap_gene "POLR3K"
```

输出：
- 综合报告图（AUC 分布、ROC、Top-DE 对比等）
- 同目录下 `test_genes_list.txt`

---

## 6. Diffusion 分支（可选）

`train_diffusion.py` 已支持 `dose_dim/time_dim` 与 dose 条件输入。  
该分支目前用于研究探索，不作为默认主训练路径。

---

## 7. 常见问题

### Q1: `libgomp: Invalid value for environment variable OMP_NUM_THREADS`
设置为有效整数：
```bash
export OMP_NUM_THREADS=1
```

### Q2: 为什么 `visualize.py` 的 Pearson 往往比 `evaluate_metrics.py` 高？
`visualize.py` 使用的是按 perturbation 聚合均值后的报告口径；`evaluate_metrics.py` 的 single-cell 指标更严格，数值通常更低但更接近真实单细胞预测难度。
