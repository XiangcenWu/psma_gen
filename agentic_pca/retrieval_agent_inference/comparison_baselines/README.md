# 独立对比实验：Qwen3.5-9B、case k-NN 与结构化 SUV

这个目录实现的是独立方法基线，不修改主实验
`trajectory_rag`，也不重复 `trajectory_rag_ablation` 中对历史轨迹模块的
组件消融。

## 推荐的完整方法表

| 方法 | 当前患者输入 | 外部/历史检索 | 模型 | 状态 |
|---|---|---|---|---|
| Majority class | 无 | 无 | 固定预测训练集中多数类 | 报告为最低参照 |
| Structured SUV ML | 全器官 FDG/PSMA SUVmean、SUVmax | 无 | 五折 Logistic Regression | 本目录新增 |
| 3D Swin | FDG/PSMA 三维体数据 | 无 | 3D Swin Tiny | 已在 `DL-only` |
| Direct clinical Qwen | Report、Medical History、PSA | 无 | 单阶段 Qwen3.5-9B | 本目录新增 |
| Direct SUV Qwen | 全器官 FDG/PSMA SUVmean、SUVmax | 无 | 单阶段 Qwen3.5-9B | 本目录新增 |
| Direct clinical+SUV Qwen | 临床信息和完整紧凑 SUV 表 | 无 | 单阶段 Qwen3.5-9B | 本目录新增，主要无检索对照 |
| PET/PDF-RAG agent | 临床信息、agent 选择的 SUV、PDF 段落 | PDF 文献 | 两阶段 Qwen3.5-9B | 已在 `infer` |
| Case k-NN label vote | Report、Medical History、PSA | LOO 相似病例真实标签 | 无 LLM | 本目录新增 |
| Trajectory-RAG | 临床信息、SUV、PDF、历史轨迹和历史标签 | PDF + LOO 病例记忆 | 两阶段 Qwen3.5-9B | 主实验 |

`infer` 不能命名为“literature-only”：它先由 planner 选择 SUV 器官和
文献 query，最终同时使用当前患者 SUV 与 PDF 文献。它可以作为
“PET/PDF-RAG、无病例记忆”的对比方法。

现有消融注册表中的 `no_rag` 只去掉历史 trajectory memory，仍然保留
当前患者 SUV 和 PDF。若完成了受控的 `no_rag` 重跑，应把它作为主实验最匹配的
PET/PDF-RAG 对照，并避免与旧 `infer` 结果重复计数。旧
`outputs/full_run_Qwen3.5_9B` 使用 `invalid-report-policy=fail`，而主实验使用
`warn`，因此旧结果只能作为描述性对照，统计时必须同时报告 coverage/failure。

## 本目录的五个新基线

### 1. `clinical_suv_no_retrieval`（最高 GPU 优先级）

单次 Qwen 调用直接接收当前患者 Report、Medical History、pretreatment PSA，
以及固定顺序的全部共有 ROI 的 FDG/PSMA SUVmean、SUVmax。没有 planner、
PDF、历史病例或任何检索。这是信息量最接近主方法的强无检索对照。

### 2. `clinical_only`

单次 Qwen 调用只接收 Report、Medical History 和 pretreatment PSA。注意：
Report 本身经常包含放射科医生记录的病灶 SUVmax，因此论文中应称为
“report/clinical baseline”，不能称为完全无影像信息。

### 3. `suv_only`

单次 Qwen 调用只接收固定、完整、按器官名排序的双示踪全器官
SUVmean/SUVmax 表；不接收 Report、病史、PSA、文献或历史病例。这里的数据来自
自动解剖 mask，不是病灶分割，也不是 lesion radiomics。

三种 direct Qwen 基线固定使用：

- `Qwen3.5-9B`；
- 249 位患者，每人 5 条独立 trajectory；
- temperature `0.7`，top-p `0.9`，seed `20260725`；
- 相同四分类定义、严格 JSON、最多 3 次 JSON retry；
- `invalid-report-policy=warn`；
- Treatment 和 post-treatment PSA 永不进入 prompt，仅在预测冻结后评估。

本地 Qwen tokenizer 对完整 249 人的实测最大输入长度分别为：

- `clinical_only`: 1015 tokens；
- `suv_only`: 7775 tokens；
- `clinical_suv_no_retrieval`: 8042 tokens。

均低于固定上限 30000，不发生输入截断。

### 4. `case_knn`

完全复用主实验的 `CompletedTrajectoryRetriever`：Report + Medical History
TF-IDF 与 log-PSA 相似度，PSA 权重 0.15，严格排除当前患者全部轨迹，
top-5 且每个来源患者最多一条。预测不调用 LLM，而是对五个历史患者的真实
management label 做相似度加权投票。

这是重要的机制对照：如果它接近或超过 trajectory-RAG，说明病例记忆的收益
可能主要来自标签传播，而不一定来自复杂 trajectory reasoning。它和主方法都属于
监督式病例记忆，不能描述为无监督 RAG。该实现先冻结全部 249 个预测，然后才读取
当前患者 Treatment 做评估。

### 5. `structured_suv_ml`

使用固定 121 个 ROI × 2 个 tracer × `SUVmean/SUVmax`，共 484 个数值特征。
严格复用 `DL-only` manifest 的五折 train/validation/test：

1. 仅在 train 拟合 median imputer、StandardScaler 和 class-balanced
   multinomial Logistic Regression；
2. 仅用 validation 从固定
   `C={0.01,0.1,1,10,100}` 选择超参数；
3. 选定后以新的预处理器和模型在 train+validation 重拟合；
4. held-out test 只预测一次，并汇总完整 OOF 结果。

它提供传统结构化影像特征基线；`DL-only` 则提供原始三维影像深度学习基线。

## 本地检查

```bash
source /share/home/xcwu/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd /share/home/xcwu/projects/psma_gen

for baseline_id in clinical_only suv_only clinical_suv_no_retrieval; do
  python agentic_pca/retrieval_agent_inference/comparison_baselines/infer_baseline.py \
    --baseline-id "${baseline_id}" --dry-run
done

python agentic_pca/retrieval_agent_inference/comparison_baselines/case_knn.py \
  --dry-run

python agentic_pca/retrieval_agent_inference/comparison_baselines/structured_suv_ml.py \
  --dry-run
```

不加载模型的 Slurm 检查：

```bash
sbatch \
  agentic_pca/retrieval_agent_inference/comparison_baselines/slurm/00_validate.slurm
```

三个基线各跑 2 人、1 trajectory 的真实模型 smoke test：

```bash
sbatch \
  agentic_pca/retrieval_agent_inference/comparison_baselines/slurm/01_smoke_array.slurm
```

## 时间有限时的提交顺序

最小但有解释力的组合是：

1. 复用已有 `DL-only`、`infer` 和 trajectory-RAG 结果；
2. CPU 运行 `case_knn` 与 `structured_suv_ml`；
3. GPU 只优先运行最强无检索基线 `clinical_suv_no_retrieval`。

只提交第 3 个 direct-Qwen array task：

```bash
sbatch --array=2-2 \
  agentic_pca/retrieval_agent_inference/comparison_baselines/slurm/10_direct_array.slurm
```

有足够 GPU 时间时，提交三个 direct-Qwen 基线：

```bash
sbatch \
  agentic_pca/retrieval_agent_inference/comparison_baselines/slurm/10_direct_array.slurm
```

CPU 基线：

```bash
sbatch \
  agentic_pca/retrieval_agent_inference/comparison_baselines/slurm/20_case_knn.slurm

sbatch \
  agentic_pca/retrieval_agent_inference/comparison_baselines/slurm/21_structured_suv_ml.slurm
```

一次提交验证、三个 direct Qwen、两个 CPU 基线和最终评估：

```bash
bash \
  agentic_pca/retrieval_agent_inference/comparison_baselines/slurm/submit_all.sh
```

`submit_all.sh` 会真实提交实验，已有作业运行时不要重复执行，否则会重复提交并写入
相同输出目录。

## 输出与统计

Qwen 和 case k-NN 使用与主实验兼容的布局：

```text
outputs/comparison_baselines_Qwen3.5-9B/<baseline-id>/
├── config.json
├── patient_manifest.json
├── summary.json
└── patients/case_*/trajectory_*.json
```

统一比较：

```bash
python \
  agentic_pca/retrieval_agent_inference/comparison_baselines/evaluate_baselines.py
```

默认以主 trajectory-RAG 为 reference，同时评估三个 direct Qwen、case k-NN
和现有 PET/PDF-RAG `infer`，输出 patient-cluster bootstrap、exact McNemar、
trajectory accuracy、macro-F1、balanced accuracy、严格患者多数票和 failure
coverage。

`structured_suv_ml` 输出独立的五折 OOF 指标，格式与 trajectory JSON 不同，
不会被 `evaluate_baselines.py` 混入；其 `summary.json` 应与 `DL-only` 五折 OOF
结果并列报告。

主要统计单位必须是患者，不是 5 条相关 trajectory。还必须披露：

- trajectory-RAG/case k-NN 是同队列严格 LOO 的监督式病例记忆；
- 现有 trajectory retriever 的 TF-IDF 在排除当前患者候选之前由完整 corpus
  拟合，存在轻微 transductive IDF 信息，case k-NN 为公平起见复用了同一实现；
- DL/structured ML 是五折 OOF，训练参考病例数与 LOO 方法不同；
- 旧 infer 与主实验的 invalid-report policy 和 coverage 不完全匹配；
- 必须确认 Report、Medical History 和 “Preoperative”等字段确实在 PET
  检查时已经可用，否则需要作为潜在时间泄漏单独说明；
- 类别高度不均衡，除 accuracy 外必须报告 macro-F1、balanced accuracy 和逐类
  recall。多数类 `systemic_treatment` 为 128/249，恒定多数类 accuracy 约
  51.4%，但 macro-F1 和少数类 recall 很低。
