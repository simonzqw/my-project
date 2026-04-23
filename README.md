# scERso 条件扩散模型（Conditional Diffusion for Single-cell Response）

本仓库当前**主线已切换为条件扩散模型**，不再推荐使用旧的 MLP/Transformer 训练路线（`train.py`）。

当前建议统一使用：
- 训练：`train_diffusion.py`
- 评估：`evaluate_diffusion.py`
- 推理/组合/插值：`predict_diffusion.py`
- 可视化：`visualize_diffusion.py`

---

## 1. 当前主线能力

### 1.1 条件扩散主干
- 背景-效应解耦：`z_bg`（background）与 `z_eff`（effect）分离编码。
- 目标模式：`target_mode = target | delta`。
- 采样与引导：支持 classifier-free guidance、DDIM 步数控制、EMA 权重评估。

### 1.2 双任务模式（重点）
通过 `--task_mode` 区分两类任务，避免“任务定义与数据不匹配”：

1. `single_gene`
   - 适用于 Adamson 等单基因扰动任务
   - 条件字段：`perturb_gene_idx`、`is_control`

2. `translation`
   - 适用于 two-condition translation（如 day4 -> day6）
   - 条件字段：`condition_id`、`source_flag`

### 1.3 数据与 control/reference 机制
- 支持 `split_strategy = random | perturbation | custom`
- 在 perturbation zero-shot 设置下，val/test 复用 train control bank，避免无 control split 崩溃
- 支持 `control_match_mode`、`control_prototype_mode`、`control_prototype_temp`

---

## 2. 项目结构（与当前主线相关）

- `train_diffusion.py`：条件扩散训练入口（主入口）
- `evaluate_diffusion.py`：评估入口（single-cell + perturbation-level 指标）
- `predict_diffusion.py`：单扰动/组合扰动预测与 latent 插值轨迹输出
- `visualize_diffusion.py`：组合扰动分析图与诊断可视化
- `models/scerso_diffusion.py`：条件扩散模型定义
- `models/diffusion_core.py`：扩散过程实现
- `utils/data_processor.py`：h5ad 读取、划分、control pool、条件字段构造
- `docs/diffusion_methodology.md`：方法论说明

> 旧路线文件（如 `train.py`, `evaluate_metrics.py`, `visualize.py`）保留仅供历史对照，不作为当前推荐路径。

---

## 3. 环境与依赖

建议：
- Python 3.8+
- PyTorch
- scanpy / anndata
- numpy / scipy / pandas
- scikit-learn
- matplotlib / seaborn
- rdkit（仅当使用 SMILES 药物特征）

并建议设置：

```bash
export OMP_NUM_THREADS=1
```

---

## 4. 训练（主入口）

## 4.1 Adamson（single_gene）

```bash
python train_diffusion.py \
  --data_path /path/to/adamson/perturb_processed.h5ad \
  --save_dir ./checkpoints_adamson_single_gene \
  --task_mode single_gene \
  --split_strategy perturbation \
  --preset vnext \
  --amp
```

## 4.2 day4/day6（translation）

```bash
python train_diffusion.py \
  --data_path /path/to/day4_to_day6_diffusion.h5ad \
  --save_dir ./checkpoints_day4_day6_translation \
  --task_mode translation \
  --split_strategy custom \
  --split_col split \
  --atac_key atac_feat \
  --preset vnext \
  --amp
```

> 快速冒烟可用 `--preset smoke`。

---

## 5. 评估

```bash
python evaluate_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --task_mode single_gene \
  --split_strategy perturbation \
  --output_json ./checkpoints_xxx/eval_metrics.json
```

对 translation 数据可改为：

```bash
--task_mode translation --split_strategy custom --split_col split
```

---

## 6. 推理与可视化

### 6.1 预测/组合/插值

```bash
python predict_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 \
  --latent_mode adaptive \
  --save_dir ./pred_out
```

### 6.2 可视化

```bash
python visualize_diffusion.py \
  --data_path /path/to/perturb_processed.h5ad \
  --model_path ./checkpoints_xxx/best_model.pth \
  --cell_line K562 \
  --perturb_genes FOXA2 GATA6 \
  --save_path ./combo_report.png
```

---

## 7. 常见问题

### Q1: `adata.obs 缺少自定义划分列: split`
你用了 `split_strategy=custom`，但数据里没有 `obs['split']`。可改成：

```bash
--split_strategy perturbation
```

或先在 h5ad 里准备 `split` 列。

### Q2: 明明传了 `--split_strategy perturbation`，日志却显示 custom
`--preset` 现在只会覆盖“未显式设置”的参数；显式传参会保留。若仍异常，请确认命令行没有重复传参。

### Q3: val/test 报 control pool 为空
在 perturbation zero-shot 下，val/test 复用 train control bank。若仍报错，通常是训练集本身没有 control 样本，需要先检查原始数据。

---

## 8. Cross-species（实验脚本）

`scripts/` 下提供：
- `prepare_mouse_context.py`
- `train_cross_species_ctx.py`
- `cross_species_infer_ctx.py`

用于 mouse→human context 预处理、训练与推理实验。
