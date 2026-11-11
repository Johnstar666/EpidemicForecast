# EpidemicForecast

融合 **SSTKG 影响力机制** 的 **EpiLLM** 区域疫情多步预测实验，为大三数据科学方向实训课的课程设计项目。

本项目在 EpiLLM（以冻结的大语言模型作为时序编码器的疫情预测框架）基础上，把 SSTKG（Spatio-Temporal Knowledge Graph，WWW 2024）中的影响力建模思想引入疫情分支：用可学习的**影响力矩阵**替代原始人口流动邻接矩阵，并在多步自回归预测阶段加入趋势一致性校正，以缓解长程预测中的误差累积。

---

## 一、设计目标与实现状态

| # | 设计目标 | 实现状态 | 对应代码 |
|---|---------|---------|---------|
| 1 | 疫情分支用 SSTKG 影响力矩阵替代原始邻接矩阵 | ✅ 已接入训练与推理主流程 | `layers/influence_matrix.py`、`models/EpiLLM_*.py` 的 `SpatioTemporalEncoder` |
| 2 | 多步预测约束，修正偏离历史规律的预测值 | ⚠️ 部分实现 | `layers/sstkg_constraint.py` 提供了完整实现；`exp/forecasting.py` 当前使用启发式趋势外推校正，由 `--use_sstkg_constraint` 开启 |
| 3 | 生成量化传播图谱，识别高风险传播源 | ⚠️ 模块已实现，尚未接入推理 | `SSTKGConstraintLayer.generate_propagation_graph()` / `analyze_temporal_evolution()` |

> 说明：`SSTKGConstraintLayer` 目前只在 `exp/forecasting.py` 中被导入，尚未实例化；推理阶段的可解释性输出仍是预留项（见源码注释）。因此本仓库的实验结果主要反映**第 1 项 + 第 2 项的启发式版本**带来的影响。

### 影响力矩阵如何工作

`InfluenceMatrixLayer` 对每个时间步生成一张 `[N, N]` 影响力矩阵，融合两类信号：

1. **结构邻接**：归一化后的人口流动邻接矩阵 `adj_norm`；
2. **流行病学相似度**：区域病例特征经投影、L2 归一化后的余弦相似度 `case_sim`。

二者由可学习参数 `alpha` 加权混合，再按 SSTKG 的定义拆分自影响与交叉影响：

```
influence[t] = I ⊙ (1 - Eye) ⊙ (alpha * adj_norm[t] + (1 - alpha) * case_sim[t]) + diag(p) * Eye
```

其中 `p`（自影响，对角）与 `I`（交叉影响缩放，非对角）均为按节点可学习的参数。疫情分支把该矩阵送入 `adj_proj` 生成图嵌入，而空间 GAT 的边仍来自原始邻接矩阵。

---

## 二、模型结构

```
输入: 病例特征 [B, seq_len, N, token_len] + 每日邻接序列 [B, seq_len, N, N]
   │
   ├─ SpatioTemporalEncoder（按 token 切分，逐时间片编码）
   │     ├─ InfluenceMatrixLayer → 影响力矩阵 → adj_proj → 图嵌入（adj 分支）
   │     └─ GATv2Conv（时序边 + 空间边，边权含前后向时间权重）→ 节点嵌入
   │
   ├─ 冻结参数的 LLM（仅接收 inputs_embeds，输出 last_hidden_state）
   │     ├─ GPT-2                                hidden = 768
   │     ├─ Gemma-3-1B-pt                       hidden = 1152
   │     └─ DeepSeek-R1-Distill-Qwen-1.5B       hidden = 1536
   │
   └─ 双 MLP 预测头
         ├─ mlp_head     → 未来 token_len 天的病例数
         └─ adj_mlp_head → 未来 token_len 天的邻接/影响力矩阵
```

- **损失函数**：`RMSE(病例预测, 真实) + RMSE(邻接预测, 真实) / 3000`，第二项作为辅助监督。
- **多步推理**：以 `token_len`（默认 7 天）为单位自回归滚动，把上一段预测拼回输入；`--use_sstkg_constraint` 开启后，从第 2 步起用「阻尼趋势外推（0.8×趋势）＋ 0.6/0.4 融合」修正预测。
- **训练策略**：Adam、早停（patience=7）、可选余弦退火与 AMP，随机种子固定为 2026。
- **评价指标**：RMSE、MAE（另在 `utils/metrics.py` 中提供 MSE / MAPE / MSPE / RSE / CORR）。

---

## 三、目录结构

```
EpidemicForecast/
├── run.py                      # 入口：参数解析、训练/测试调度
├── test_all_models.py          # 批量测试 checkpoints（含硬编码服务器路径）
├── requirements.txt
├── result_long_term_forecast.txt   # 追加写的实验结果日志（RMSE / MAE）
├── data_provider/
│   ├── data_factory.py         # 数据集与 DataLoader 构造
│   └── data_loader.py          # 图/标签读取、特征构造、train/val/test 划分
├── exp/
│   ├── exp_basic.py            # 模型注册表
│   └── forecasting.py          # 训练、验证、测试、训练曲线可视化
├── layers/
│   ├── influence_matrix.py     # ✅ SSTKG 影响力矩阵层（已接入）
│   ├── sstkg_constraint.py     # 多步约束 / 传播图谱（尚未接入主流程）
│   └── mlp.py
├── models/
│   ├── EpiLLM_Gpt2.py          # GPT-2 骨干
│   ├── EpiLLM_Gemma1.py        # Gemma-3-1B 骨干
│   └── EpiLLM_DS3.py           # DeepSeek-R1-Distill-Qwen-1.5B 骨干
├── utils/
│   ├── metrics.py
│   └── tools.py                # 早停、学习率调整、可视化
├── dataset/
│   ├── Italy/                  # 105 个区域，2020-02-24 ~ 2020-04-24
│   │   ├── italy_labels.csv
│   │   └── graphs/IT_<date>.csv
│   └── England/                # 129 个区域，2020-03-13 ~ 2020-05-12
│       ├── england_labels.csv
│       ├── graphs/EN_<date>.csv
│       └── graphs.zip
├── train_results/<setting>/    # 6 组训练记录：曲线 PNG、日志 CSV、摘要 TXT
├── checkpoints/<setting>/      # 训练产物（gitignore）
└── test_results/<setting>/     # 可视化预测对比图（gitignore）
```

> `gpt` 是一个指向原实验机 `/root/autodl-tmp/models/gpt2` 的符号链接，已加入 `.gitignore`，在其它机器上无效，需按下方说明自行准备模型权重。

---

## 四、环境准备

- Python 3.10+，PyTorch 2.5.0，**必须配备 CUDA GPU**（源码中大量硬编码 `.to('cuda')` / `.cuda()`，无 CPU 回退）。
- 安装依赖：

```bash
pip install -r requirements.txt
pip install networkx        # data_loader.py 依赖，但 requirements.txt 中未列出
```

- 准备 LLM 权重（三种骨干任选其一，与 `--model` 对应）：

| `--model` | 权重目录示例 | 隐藏维度 |
|-----------|-------------|---------|
| `EpiLLm_Gpt2` | `gpt2` | 768 |
| `EpiLLm_gemma1` | `gemma-3-1b-pt` | 1152 |
| `EpiLLm_DS3` | `DeepSeek-R1-Distill-Qwen-1.5B` | 1536 |

通过 `--llm_ckp_dir` 指定本地权重路径（Hugging Face 格式，需含 `config.json`）。

- 数据集已随仓库提供。`data_loader.py` 读取的是 `./dataset/<Country>/graphs/<CC>_<date>.csv`，若只有压缩包请先解压到该目录，例如 `dataset/England/graphs.zip` → `dataset/England/graphs/`。

---

## 五、快速开始

> **重要**：`run.py` 的默认 `--root_path` 是原实验服务器的绝对路径 `/root/experiment/EpiLLM-backup/dataset/Italy`，本地运行务必显式覆盖。

**训练（England + GPT-2，7 天预测）：**

```bash
python run.py \
  --is_training 1 \
  --model EpiLLm_Gpt2 \
  --root_path ./dataset/England \
  --data_path england_labels.csv \
  --node_num 129 \
  --llm_ckp_dir ./gpt \
  --seq_len 21 --label_len 14 --token_len 7 --test_pred_len 7 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --use_sstkg_constraint \
  --des sstkg_full
```

**训练（Italy + DeepSeek 蒸馏模型）：**

```bash
python run.py \
  --model EpiLLm_DS3 \
  --root_path ./dataset/Italy \
  --data_path italy_labels.csv \
  --node_num 105 \
  --llm_ckp_dir /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --learning_rate 0.005 \
  --des sstkg_full
```

**仅测试已有 checkpoint：**

```bash
python run.py --is_training 0 --model EpiLLm_Gpt2 \
  --root_path ./dataset/England --data_path england_labels.csv --node_num 129 \
  --llm_ckp_dir ./gpt \
  --test_dir <setting> --test_file_name checkpoint.pth --visualize
```

`test_all_models.py` 可批量跑 `checkpoints/` 下含 `full_training_0` 的实验，但它硬编码了 `/root/experiment/EpiLLM`、`/root/miniconda3/bin/activate epillm` 等服务器路径，使用前需按本机环境修改。

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `EpiLLm_Gpt2` | 骨干模型，见上表 |
| `--root_path` / `--data_path` | 服务器路径 / `italy_labels.csv` | 数据目录与标签文件 |
| `--node_num` | 105 | 区域数：Italy 105、England 129 |
| `--seq_len` / `--label_len` / `--token_len` | 21 / 14 / 7 | 输入长度、标签长度、单步预测 token 长度 |
| `--test_pred_len` | 7 | 测试阶段预测天数 |
| `--learning_rate` | 0.005 | 建议：Italy 0.005，England 0.001 |
| `--train_epochs` / `--patience` | 100 / 7 | 训练轮数与早停耐心值 |
| `--use_sstkg_constraint` | False | 开启多步预测趋势校正 |
| `--use_amp` / `--cosine` / `--use_multi_gpu` | False | 混合精度 / 余弦退火 / 多卡 DDP |
| `--visualize` | False | 测试时输出第 0 号节点的预测对比图 |

每次运行会以 `setting` 命名实验（由任务名、模型、超参拼成），结果写入：

- `checkpoints/<setting>/checkpoint.pth` — 验证集最优权重
- `train_results/<setting>/` — `training_curves.png`、`loss_curves.png`、`training_log.csv`、`training_summary.txt`
- `test_results/<setting>/` — 预测可视化
- `result_long_term_forecast.txt` — 以追加方式记录每次测试的 RMSE / MAE

---

## 六、实验结果

数据来自仓库内的 `result_long_term_forecast.txt` 与 `train_results/*/training_summary.txt`。基线为 `full_training` 系列，改进版为 `sstkg_full` 系列（Gemma 的 England 改进版为 `sstkg_full_step2`）。指标越小越好，预测步长均为 7 天。

**England（129 区域，lr=0.001）**

| 骨干 | 基线 RMSE | 基线 MAE | +SSTKG RMSE | +SSTKG MAE | RMSE 变化 |
|------|----------:|---------:|------------:|-----------:|----------:|
| GPT-2 | 6.2498 | 4.7294 | 6.2473 | 4.7352 | −0.04% |
| Gemma-3-1B | 6.5144 | 4.5624 | 6.2464 | 4.1983 | −4.11% |
| DeepSeek-R1-Distill-Qwen-1.5B | 6.1192 | 4.3832 | **6.0936** | **4.2291** | −0.42% |

**Italy（105 区域，lr=0.005）**

| 骨干 | 基线 RMSE | 基线 MAE | +SSTKG RMSE | +SSTKG MAE | RMSE 变化 |
|------|----------:|---------:|------------:|-----------:|----------:|
| GPT-2 | 23.6129 | 14.8916 | 24.7017 | 14.6506 | +4.61% |
| Gemma-3-1B | 28.9988 | 16.7905 | **23.2106** | 14.9174 | −19.96% |
| DeepSeek-R1-Distill-Qwen-1.5B | 26.1500 | 15.7662 | 23.7582 | **14.3868** | −9.15% |

**观察**

- 融合影响力矩阵在 **Italy + 大容量骨干** 上收益最明显（Gemma RMSE 下降约 20%，DS3 下降约 9%，MAE 同步改善 8%~11%）。
- England 上整体幅度较小，但 MAE 普遍改善；`sstkg_train_viz` 一组还记录到 RMSE 5.9551 / MAE 4.3274。
- GPT-2 骨干基本持平甚至略差，说明融合收益与骨干容量、数据规模有关，不能一概而论。

> **结果使用须知**：`result_long_term_forecast.txt` 是追加写入的实验日志，含重复条目与中间调试记录（原文件用 🌟 标注了有效结果）。上表各组并非严格控制变量的消融实验（训练轮次、可视化配置等可能不同），数值仅用于观察趋势，不宜作为严格结论引用。

**训练过程**：`train_results/` 下的 6 组记录显示早停发生在第 20~59 轮（England 38~59 轮、Italy 20~40 轮），每轮的训练/验证/测试损失与学习率均保存在 `training_log.csv`，可直接绘图复现。

---

## 七、已知限制与后续工作

- **强依赖 GPU**：`models/` 与 `exp/forecasting.py` 中硬编码 `cuda`，无法在纯 CPU 环境运行。
- **路径硬编码**：默认 `--root_path`、`gpt` 符号链接、`test_all_models.py` 的 conda 与数据集路径均来自原实验服务器，迁移时需逐项修改。
- **依赖清单不完整**：`networkx` 未写入 `requirements.txt`；`patool` 等依赖与运行无关。
- **第 2、3 项设计目标待接入**：`SSTKGConstraintLayer` 的影响力一致性校正与传播图谱尚未进入主推理链路；当前 `--use_sstkg_constraint` 是启发式趋势阻尼，且仅在已产生 ≥2 段预测时生效。
- **实验结果可复现性**：日志为追加写、存在重复与调试条目，建议改用结构化记录（如 CSV/JSON）并固定每组实验的完整配置。
- **数据覆盖**：`data_loader.py` 预留了 France（81 区域）与 Spain（34 区域）的读取逻辑，但仓库未提供对应数据。
- **Windows 检出**：`train_results/` 下实验目录名较长，克隆前建议先执行 `git config core.longpaths true`，否则部分文件会检出失败。

---

## 八、后续计划

1. 把 `SSTKGConstraintLayer` 正式接入 `exp/forecasting.py` 的多步推理，替换启发式校正，并做同配置消融对比。
2. 在推理时导出影响力矩阵，生成区域传播量化图谱与时序演化分析（模块已具备接口）。
3. 补齐 `requirements.txt`、去除硬编码设备与路径，增加 CPU/单卡友好开关。
4. 引入 France / Spain 数据集，验证跨区域泛化能力。
