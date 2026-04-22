# 当前扩散模型方法学（scERso diffusion）

## 1) 任务与条件建模
目标是学习条件分布：

\[
p(\mathbf{x}_{\text{pert}}\mid \mathbf{x}_{\text{ctrl}},\ \text{perturb},\ \text{cell\_line},\ \text{dose},\ \text{ATAC},\ \text{drug})
\]

其中 \(\mathbf{x}_{\text{ctrl}}\) 为对照RNA表达，\(\mathbf{x}_{\text{pert}}\) 为扰动后表达。

## 2) 语义潜变量与上下文
模型先把多模态条件编码为语义潜变量 \(\mathbf{z}_{sem}\)：

- RNA对照投影；
- perturbation embedding（可按dose做缩放）；
- cell-line embedding；
- dose投影；
- 可选 ATAC / drug 特征投影；
- 通过多头自注意力融合，再做残差MLP与LayerNorm得到 \(\mathbf{z}_{sem}\)。
- 并行引入 joint semantic encoder（RNA/perturb/cell-line/dose 拼接后 MLP）并与 attention 路径做门控融合，提升单扰动语义表示稳定性。

随后构造扩散条件向量：

\[
\mathbf{c}=\left[\mathbf{x}_{\text{ctrl}};\mathbf{z}_{sem}\right]
\]

并支持条件dropout（训练时随机将 \(\mathbf{z}_{sem}\) 置零）用于 classifier-free guidance。

## 3) 前向扩散（加噪）
采用高斯扩散前向过程：

\[
q(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N}\left(\sqrt{\bar\alpha_t}\mathbf{x}_0,(1-\bar\alpha_t)\mathbf{I}\right)
\]

实现为：

\[
\mathbf{x}_t=\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\quad \boldsymbol\epsilon\sim\mathcal{N}(0,\mathbf{I})
\]

噪声调度可选 linear 或 cosine（默认 cosine）。

## 4) 反向去噪网络
去噪器是 Squidiff 风格 MLP：

- 输入 \(\mathbf{x}_t\) 与条件 \(\mathbf{c}\)；
- 时间步 \(t\) 先做正弦位置编码并MLP；
- 每个残差块中注入 time embedding 和 \(\mathbf{z}_{sem}\)；
- 输出与输入同维度的表达向量。

当前 objective 配置为 `pred_x0`，即网络直接预测 \(\hat{\mathbf{x}}_0\)。

## 5) 训练目标
每次随机采样时间步 \(t\)，最小化：

\[
\mathcal{L}=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol\epsilon}\left[\lVert f_\theta(\mathbf{x}_t,t,\mathbf{c})-\mathbf{x}_0\rVert_2^2\right]
\]

代码中为按样本求 mean(dim=1) 后再 batch 平均；可选样本权重（配合时间步重采样器）。

## 6) 采样与推理
### DDPM采样
按 \(t=T-1\to0\) 迭代：

1. 用模型得到 \(\hat{\mathbf{x}}_0\)（或先预测噪声再换算）；
2. 通过后验均值方差
\(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\hat{\mathbf{x}}_0)\)
采样 \(\mathbf{x}_{t-1}\)。

### DDIM快速采样
若 `sample_steps < timesteps`，走DDIM子序列更新，支持 \(\eta\) 控制随机性。

### Latent Interpolation
支持在两个语义 latent 之间做线性插值轨迹：
\[
z(\alpha)=(1-\alpha)z_A+\alpha z_B,\ \alpha\in[0,1]
\]
可用于剂量/状态连续过渡分析（`predict_diffusion.py --interpolate_to --interp_steps`）。

### Classifier-Free Guidance
同时计算条件/无条件预测并线性组合：

\[
\hat{y}=\hat{y}_{uncond}+s(\hat{y}_{cond}-\hat{y}_{uncond})
\]

其中 \(s=\) `guidance_scale`。

## 7) 数学意义（直观解释）
1. **把高维基因表达生成转化为“逐步细化”问题**：
   从各向同性高斯噪声出发，逐步收缩到符合条件分布的表达向量。
2. **条件潜变量 \(\mathbf{z}_{sem}\) 是“扰动语义坐标”**：
   把 perturb/cell-line/dose/ATAC/drug 统一到同一潜空间，等价于给逆扩散轨迹施加“场”。
3. **`pred_x0` 目标偏向直接回归生物信号主体**：
   相比纯噪声预测，直接监督 \(\mathbf{x}_0\) 对表达幅值的拟合更直接（但对稳定性和校准依赖调参与归一化）。
4. **CFG 对应条件似然梯度放大**：
   在采样时增强条件项贡献，提高条件一致性（通常会牺牲一些多样性）。
5. **时间步重采样 = 重要性采样思想**：
   loss-second-moment 更关注高损失时间步，近似减少梯度方差并提升样本效率。

## 8) 与组合扰动的关系
该实现支持先编码单扰动 latent，再做组合后采样：

- `sum/mean`：线性叠加（可解释、稳定）；
- `adaptive`：在加权叠加基础上，引入 pairwise 非线性交互项
  \\(\phi([z_i,z_j,z_i\\odot z_j,|z_i-z_j|])\\) 与门控融合：
  \\[
  z_{combo}=g\\odot z_{lin} + (1-g)\\odot (z_{lin}+z_{pair})
  \\]
  其中 \\(g=\\sigma(\\psi([z_{lin},\\bar z]))\\)。

这使组合扰动可显式表达一部分协同/拮抗的非线性效应，同时保持单扰动路径不变。
