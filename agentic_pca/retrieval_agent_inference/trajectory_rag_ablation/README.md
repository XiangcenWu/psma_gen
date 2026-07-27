# TMI-style trajectory-RAG ablation study

This directory is an independent, pre-registered ablation pipeline. It does
not modify the production `trajectory_rag` implementation or the completed
main experiment.

The fixed study setting is:

- inference model: `Qwen3.5-9B`;
- historical corpus:
  `outputs/full_run_Qwen3.5_9B`;
- cohort: 249 patients, five stochastic trajectories per patient;
- retrieval: strict leave-one-patient-out before every ranking/filter;
- common generation seed: `20260725`, temperature `0.7`, top-p `0.9`;
- invalid XML report policy: `warn`, matching the main trajectory-RAG run;
- one experiment per output directory and one GPU job per experiment.

The historical directory was previously named `outputs/full_run`. Its current
content fingerprint is `4f56504f0fda05ee1d53e2ce1afbc7561ba99c9271dd2aeeab365456601f9b3e`,
which matches the corpus recorded by the completed main experiment
`trajectory_rag_full_run_Qwen3.59BonQwen3.59B`.

The old no-RAG run and the completed main RAG run are not a fully controlled
ablation pair: they used different invalid-report policies and code/prompt
hashes. Therefore this study deliberately reruns both `full` and `no_rag`
under the same runner and inference settings.

## Files

```text
trajectory_rag_ablation/
├── experiments.json          # immutable experiment registry
├── ablation_specs.py         # registry validation
├── ablation_retriever.py     # LOO ranking/filter controls
├── contexts.py               # planner/final information projections
├── infer_ablation.py         # strict-JSON ablation inference
├── validate_ablations.py     # equivalence, leakage and determinism checks
├── validate_run.py           # completed-run structural validation
├── evaluate_ablations.py     # patient-cluster-aware statistical analysis
├── registry_cli.py
└── slurm/
    ├── 00_validate.slurm
    ├── 01_smoke_array.slurm
    ├── 02_primary_array.slurm
    ├── 03_retrieval_array.slurm
    ├── 04_sensitivity_array.slurm
    ├── 05_all_array.slurm
    ├── 06_single.slurm
    ├── 07_validate_outputs.slurm
    ├── 08_evaluate.slurm
    └── run_experiment.sh
```

Every run records the resolved experiment specification, registry and code
hashes, actual planner context, actual final CASE context, prompt hashes,
retrieval fingerprint, and label-exposure flags. Information removed by an
ablation is not retained inside the corresponding model-context object.

## Pre-registered experiments

### Primary group

| ID | Scientific contrast |
|---|---|
| `full` | Full planner and final trajectory RAG; positive control |
| `no_rag` | Neither stage receives historical context |
| `planner_only` | Historical guidance only during evidence selection |
| `final_only` | Historical CASE evidence only during final prediction |
| `patient_context_only` | Final stage sees similar pretreatment inputs only |
| `no_historical_outcomes` | Removes outcome and correctness together |
| `no_historical_prediction` | Keeps outcomes but removes prediction/reason/correctness |
| `outcome_only` | Similar patient input plus observed management label only |
| `outcome_only_permuted` | Outcome-only patient-level label permutation control |
| `no_historical_reason` | Removes historical free-text reasoning |
| `no_strategy_hints` | Planner sees patient context but no historical strategy |

`no_rag`, `planner_only`, `final_only`, and `full` form a matched 2×2 design
over planner memory and predictor memory. The outcome experiments distinguish
case similarity from supervised label memory.

### Retrieval-mechanism group

| ID | Scientific contrast |
|---|---|
| `text_only` | Report + medical history, PSA weight zero |
| `report_only` | Report-only retrieval |
| `history_only` | Medical-history-only retrieval |
| `psa_only` | True PSA-only ranking, no text fallback/tie-break |
| `random_retrieval` | Deterministic SHA256 random negative control |
| `least_similar` | Least-similar negative control |
| `correct_only_corpus` | Outcome-aware oracle-quality diagnostic |
| `incorrect_only_corpus` | Outcome-aware robustness diagnostic |
| `shuffled_outcomes` | Full context with globally permuted patient labels |

The correct/incorrect corpus filters use outcomes from reference patients and
are diagnostic analyses, not deployable retrieval policies.

### Sensitivity group

| ID | Scientific contrast |
|---|---|
| `topk_1`, `topk_3`, `topk_10` | Retrieval-depth sensitivity |
| `no_case_diversity` | Allows five trajectories from one reference patient |
| `organ_hints_only` | Planner receives only aggregated organ hints |
| `literature_hints_only` | Planner receives only literature-query hints |

Patient-level label permutation is global and deterministic: all trajectories
from the same reference patient share one permuted label, and the corpus label
margins are preserved.

## Local validation

List all registered experiments:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python -m \
  agentic_pca.retrieval_agent_inference.trajectory_rag_ablation.registry_cli
```

Run the full equivalence/leakage audit:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/validate_ablations.py
```

This verifies all 249 patients for:

- exact `full` retrieval equivalence with the original retriever;
- leave-one-patient-out exclusion;
- deterministic random retrieval;
- stage-specific context contracts;
- absence of outcomes/predictions in their respective ablations.

Dry-run one experiment without loading the model:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/infer_ablation.py \
  --experiment-id full --dry-run
```

Exercise full orchestration with the mock model:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/infer_ablation.py \
  --experiment-id no_historical_outcomes \
  --mock-model --mock-invalid-first \
  --max-patients 2 --num-trajectories 1 \
  --output-dir /tmp/traj_ablation_mock
```

## Slurm workflow

First validate the code and reference corpus:

```bash
sbatch \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/slurm/00_validate.slurm
```

Then run a real-model smoke test for every condition:

```bash
sbatch \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/slurm/01_smoke_array.slurm
```

For full inference, use either the single all-experiment array:

```bash
sbatch \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/slurm/05_all_array.slurm
```

or the three staged arrays:

```bash
primary_job=$(sbatch --parsable \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/slurm/02_primary_array.slurm)
retrieval_job=$(sbatch --parsable \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/slurm/03_retrieval_array.slurm)
sensitivity_job=$(sbatch --parsable \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/slurm/04_sensitivity_array.slurm)
```

Do not submit `05_all_array.slurm` together with the three staged arrays: they
target the same output directories.

Resume or run one experiment:

```bash
sbatch --export=ALL,ABLATION_ID=full \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/slurm/06_single.slurm
```

The verified Qwen3.5-9B jobs request one A800, 80 GB host RAM, six CPUs and an
18-hour limit. Array concurrency is capped at two. Override `OUTPUT_ROOT`,
`SOURCE_RUN`, or `MODEL_PATH` only when intentionally starting a separately
documented study.

After all staged arrays reach a terminal state, validate every output using
`afterany`, because inference intentionally returns nonzero when even one
trajectory has a recorded JSON-generation failure:

```bash
check_job=$(sbatch --parsable \
  --dependency=afterany:${primary_job}:${retrieval_job}:${sensitivity_job} \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/slurm/07_validate_outputs.slurm)

sbatch --dependency=afterok:${check_job} \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/slurm/08_evaluate.slurm
```

Each full experiment writes:

```text
outputs/trajectory_rag_ablation_Qwen3.5-9B_on_Qwen3.5-9B/<experiment-id>/
├── config.json
├── patient_manifest.json
├── summary.json
└── patients/case_*/trajectory_*.json
```

Never run patient shards concurrently into the same directory. The config,
manifest, and summary updates are atomic per process but are not a
multi-process merge protocol.

## TMI-oriented analysis

The evaluator reports:

- attempted/completed/failed trajectories and coverage;
- trajectory accuracy, macro-F1 and balanced accuracy;
- unweighted patient-mean trajectory accuracy;
- patient strict-majority-vote accuracy, macro-F1, balanced accuracy and
  per-class metrics, with ties/non-majority pluralities treated as abstentions
  and scored incorrect;
- paired common-patient cluster-bootstrap accuracy differences with 95% CIs;
- exact patient-majority McNemar tests with Holm multiplicity correction when
  SciPy is available.

Patients, not stochastic trajectories, are the inferential unit. This avoids
treating five correlated trajectories from one patient as five independent
clinical samples. Report class-wise results because the cohort is imbalanced
and `local_treatment`/`other_examination` are rare.

The analysis is internal leave-one-patient-out validation over a labelled case
memory. It is not an independent external test set, and historical outcomes
supplied to the final model must be described as supervised case memory rather
than unsupervised RAG.
