# Two-stage literature-only trajectory RAG

This directory mirrors the strict-JSON inference and trajectory-RAG pipeline
without reading the current patient's structured whole-organ SUV statistics.

The Slurm workflow is a true two-stage ablation:

1. generate a historical no-SUV corpus without trajectory RAG;
2. use only that stage-1 corpus for leave-one-patient-out trajectory retrieval
   and generate a second set of predictions.

Stage 2 starts only if stage 1 exits successfully. Both stages are resumable,
so resubmitting the same job skips semantically validated completed
trajectories.

The boundary is deliberate:

- the model still receives the original `Report`, `Medical History`, and
  pretreatment `PSA`;
- the report is not text-redacted, so a source report may itself mention an
  `SUVmax`;
- the report or medical history may also describe prior therapy; the
  structured `Treatment` field is hidden, but free-text treatment language is
  not redacted;
- no `fdg_suv_statistics.json` or `psma_suv_statistics.json` file is opened;
- no organ selector, structured SUV payload, or `SUV-*` evidence ID is created;
- stage-1 evidence consists of the patient input and local PDF passages;
- stage-2 evidence additionally includes leave-one-patient-out historical
  trajectories produced by stage 1;
- the source `Treatment` is revealed only after a valid prediction is frozen.

The default Slurm workflow does not read `full_run_Qwen3.5_9B`. This prevents
historical categories produced by the legacy SUV pipeline from entering the
ablation.

## Files

- `infer.py`: literature-only strict-JSON generation and output/resume logic;
- `trajectory_rag.py`: validated leave-one-patient-out trajectory retrieval
  whose model-visible projection contains no structured SUV fields;
- `no_suv_trajectory_rag.slurm`: one GPU allocation that runs stage 1 and then
  stage 2 sequentially.

## Output

The two output roots are separate:

```text
outputs/no_suv_full_run_Qwen3.5-9B/
├── config.json
├── patient_manifest.json
├── summary.json
└── patients/case_*/trajectory_*.json

outputs/no_suv_trajectory_rag_Qwen3.5-9B_on_no_suv_Qwen3.5-9B/
├── config.json
├── patient_manifest.json
├── summary.json
└── patients/case_*/trajectory_*.json
```

## Checks

Validate stage 1 without loading the model or accessing structured SUV files:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/no_suv/infer.py \
  --dry-run
```

Run a CPU-safe end-to-end two-stage smoke test:

```bash
MOCK_MODEL=1 MAX_PATIENTS=2 NUM_TRAJECTORIES=1 \
BASE_OUTPUT_DIR=/tmp/no_suv_base_mock \
RAG_OUTPUT_DIR=/tmp/no_suv_trajectory_rag_mock \
bash agentic_pca/retrieval_agent_inference/no_suv/no_suv_trajectory_rag.slurm
```

Submit the full two-stage run with:

```bash
sbatch \
  agentic_pca/retrieval_agent_inference/no_suv/no_suv_trajectory_rag.slurm
```

Useful environment overrides are `BASE_OUTPUT_DIR`, `RAG_OUTPUT_DIR`,
`NUM_TRAJECTORIES`, `TOP_K`, `MAX_PER_CASE`, `PSA_WEIGHT`, and `BASE_SEED`.
Do not point either output variable at the old
`no_suv_trajectory_rag_Qwen3.5-9B_on_Qwen3.5-9B` directory, whose immutable
configuration records the legacy `full_run_Qwen3.5_9B` source.
