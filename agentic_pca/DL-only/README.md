# FDG + PSMA PET 影像-only 五折分类基线

这个目录实现一个与 Agent 方法对照的 4 分类影像基线。每位患者的 FDG
和 PSMA PET 被处理成两个 3D channel，默认模型为从头训练的 3D Swin
Transformer。外层 5-fold 的测试预测合并为一份完整 OOF（out-of-fold）
结果，所以 249 位患者每人恰好测试一次。

## 先说明数据审计结果

代码已对目标 JSON 和影像目录做过只读核查：

- JSON 共 249 位患者，均可按中文姓名唯一匹配到
  `/share/home/anyone/Data/RJPCa/batch*/<患者>/`。
- 两种 PET 均为 249/249 存在。
- 标签分布为：
  `radical_prostatectomy=97`、`systemic_treatment=128`、
  `local_treatment=17`、`other_examination=7`。
- 原始 PET 并不是 `128×128×384`。FDG 主要为 `192×192×673`，
  PSMA 也主要为 `192×192×673`，其中 6 位患者的两模态数组尺寸不同。
- 249 位患者中，FDG 与 PSMA 的 affine 没有一例完全相同。不能直接把
  两个原始 NIfTI 数组按 channel 堆叠。
- 现有 `data_h5.h5` 虽然是 `128×128×384`，但只覆盖目标队列中的
  62 位患者，不能和其余 187 位患者混用。

因此，`prepare_dataset.py` 会从全部 249 位患者的原始文件统一重做缓存：

1. 把每种 PET 分别转换到 canonical RAS 方向；
2. 用对应 CT body mask 只确定身体 bounding box；
3. 在每位患者、每个 tracer 内按正值体素的 p1–p99 clip/scale 到 `[0,1]`；
4. 分别以 trilinear interpolation resize 到 `128×128×384`；
5. 按 `[FDG, PSMA]` 堆叠为 float16 的
   `[2,128,128,384]` 缓存。

CT 和 mask 只用于确定 crop，不会作为模型输入。两种 PET 是
independent body-normalized grids，并不是严格的 FDG→PSMA 体素级配准。
论文方法部分必须明确这一点；如果要宣称 voxel-level early fusion，需要
另做经验证的跨示踪剂配准实验。

默认会保存原始 PET/body-mask 和缓存的 SHA256，并拒绝不同 case ID 下
完全相同的预处理 PET pair，防止重复影像跨 fold 泄漏。

## 严格的五折设计

- 标签映射与 Agent 代码完全相同：
  `observed-management-v1`。
- case ID 使用与 Agent 相同的
  `retrieval-agent-inference-v1` salt，可直接逐患者配对。
- 外层：
  `StratifiedKFold(5, shuffle=True, random_state=20260727)`。
- 每个外层训练集合内部再用固定 seed 分层划出 20% validation。
- 约为每折 159/40/50（最后一折 160/40/49）
  train/validation/test。
- 最佳 checkpoint 只由 validation macro-F1 决定，val loss 用于打破平局；
  test 不参与预处理参数拟合、class weight、early stopping 或调参。
- test Dataset 不返回标签。模型预测先写入 blinded JSONL，之后才和
  manifest 的标签连接以计算指标。
- 五折 test 集严格互斥，其并集必须等于全部 249 位患者；聚合器会拒绝
  缺失、重复、未知患者、错误 fold、错误标签、概率不合法或 fingerprint
  不一致的结果。

Treatment 只用于构造标签、分层和最终评价。Report、Medical History、
PSA、Post-treatment PSA、Agent trajectory 和治疗文本都不会进入模型。

## 默认模型和训练

`swin_tiny_3d` 是 MONAI 3D Swin Transformer encoder：

- input channels 2，embed dim 48；
- patch/window 都是 `(4,4,4)`；
- depths `(2,2,6,2)`，heads `(3,6,12,24)`；
- stochastic depth 0.2；
- global average pooling + LayerNorm + dropout + 4-class linear head；
- 9,701,530 个参数，activation checkpointing；
- 从头训练，不使用 2D ImageNet 权重。

默认 AdamW、effective-number class-weighted cross entropy、label smoothing
0.05、bf16、batch size 1、gradient accumulation 4、warmup + cosine schedule。
增强只在 train 使用同步左右翻转和轻量 tracer-specific intensity jitter/noise。
另提供 `densenet121_3d`，可作为架构敏感性分析，但应使用不同 OUTPUT
目录，避免覆盖主实验。

## 推荐运行方式

先确保日志目录存在（仓库中已放置 `.gitkeep`），然后可以一键建立依赖：

```bash
agentic_pca/DL-only/slurm/submit_all.sh
```

它会依次提交：

1. CPU 预处理；
2. 依赖预处理成功的 5-fold GPU array；
3. 依赖全部五折成功的汇总作业。
4. 依赖汇总成功的 DL–Agent 配对比较。

如果希望一个一个提交：

```bash
sbatch agentic_pca/DL-only/slurm/00_prepare.slurm

# 上一步完成后，可以按任意顺序分别提交五折
sbatch agentic_pca/DL-only/slurm/10_fold_0.slurm
sbatch agentic_pca/DL-only/slurm/11_fold_1.slurm
sbatch agentic_pca/DL-only/slurm/12_fold_2.slurm
sbatch agentic_pca/DL-only/slurm/13_fold_3.slurm
sbatch agentic_pca/DL-only/slurm/14_fold_4.slurm

# 五折全部完成后
sbatch agentic_pca/DL-only/slurm/20_aggregate.slurm
```

也可以只提交 array：

```bash
sbatch agentic_pca/DL-only/slurm/15_folds_array.slurm
```

五个 fold 相互独立，可以同时跑。默认每折申请一张 A800、6 CPU、80 GB
主存；`%5` 允许最多五折并发。若集群资源紧张，可把 array 文件中的 `%5`
改成 `%2`。这与 Agent 主实验的显卡申请不是同一负载假设：完整 3D
反向传播需要更明确的显存保障。

常用覆盖参数示例：

```bash
MODEL=densenet121_3d \
OUTPUT=agentic_pca/retrieval_agent_inference/outputs/dl_densenet121_fdg_psma_5fold \
sbatch agentic_pca/DL-only/slurm/10_fold_0.slurm
```

## 输出

默认输出目录是：

```text
agentic_pca/retrieval_agent_inference/outputs/dl_swin3d_fdg_psma_5fold/
├── folds/
│   ├── fold_0/
│   │   ├── best_model.pt
│   │   ├── history.csv
│   │   ├── metrics.json
│   │   ├── split.json
│   │   └── test_predictions.jsonl
│   └── fold_1 ... fold_4
├── patients/
│   └── case_xxxxxxxxxxxxxxxx/prediction.json
├── oof_predictions.csv
├── confusion_matrix.csv
└── summary.json
```

`summary.json` 保存 pooled OOF accuracy、balanced accuracy、macro/weighted
F1、每类别 precision/recall/F1、OVR AUROC/AUPRC、混淆矩阵、五折
mean±sample SD、majority-class baseline，以及按真实类别分层的 10,000 次
patient bootstrap 95% CI。`coverage` 必须为 1.0。

缓存预计约 6.3 GB。包含姓名和原始路径的私有 `patient_manifest.json`
只保存在 cache 目录；正式 output 的患者结果只使用匿名 case ID。

## 和 Agent 主结果做配对比较

五折汇总完成后运行：

```bash
python agentic_pca/DL-only/compare_with_agent.py \
  --dl-output agentic_pca/retrieval_agent_inference/outputs/dl_swin3d_fdg_psma_5fold \
  --agent-output agentic_pca/retrieval_agent_inference/outputs/trajectory_rag_full_run_Qwen3.59BonQwen3.59B
```

也可以提交默认比较作业：

```bash
sbatch agentic_pca/DL-only/slurm/30_compare_agent.slurm
```

比较另一个 Agent 主结果时要使用不同文件名：

```bash
AGENT_OUTPUT=agentic_pca/retrieval_agent_inference/outputs/trajectory_rag_qwen3_8b_full_run_on_Qwen3.59B \
COMPARISON_OUTPUT=agentic_pca/retrieval_agent_inference/outputs/dl_swin3d_fdg_psma_5fold/agent_comparison_qwen3_8b.json \
sbatch agentic_pca/DL-only/slurm/30_compare_agent.slurm
```

比较脚本按患者对齐 case ID；Agent 的多条 trajectory 使用严格多数票，
平票或没有超过半数视为 abstention 并计错。输出包括 paired exact McNemar
检验，以及 accuracy、macro-F1、balanced accuracy 差值的分层 paired
bootstrap 95% CI。若某位患者的所有 Agent trajectory 都在写入 evaluation
前失败，脚本使用已经冻结的 DL OOF manifest 真值，并在结果中单独记录
无法从 Agent 文件交叉核对的患者数，不会把失败患者静默删除。

## TMI 写作边界

这是单中心、同队列的内部 5-fold cross-validation，不是 external
validation。标签表示真实记录的 observed management，不表示最佳治疗。
另外，Agent 使用文本、PSA、检索证据和历史 trajectory，而这个模型只使用
PET 影像，所以应称为 image-only comparator，不能把差异解释成纯粹的算法
优劣。`other_examination` 只有 7 例，每个 validation fold 仅 1 例，
macro-F1 和该类 recall 的不确定性必须结合置信区间报告。
