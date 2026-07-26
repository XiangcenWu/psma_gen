# Strict-JSON PET/PDF-RAG inference

`infer.py` processes the 249 patients in
`agentic_pca/agent_dataset/FDG&PSMA双探针_clean_en_with_report.json`.
For every independent trajectory it:

1. gives only `Report`, `Medical History`, and pretreatment `PSA` to an
   evidence-selection agent;
2. requires a strict JSON request containing one or more exact SUV organs and
   one literature-search sentence;
3. retrieves paired FDG/PSMA whole-organ SUV statistics and relevant passages
   from the local PDF collection;
4. requires a strict JSON final prediction with one of four management
   categories, a reason, and valid evidence IDs;
5. reveals the source `Treatment` only after the prediction is frozen and
   computes exact agreement in Python.

Only malformed or schema-invalid model outputs are retried. A clinically
incorrect but valid prediction is retained and recorded as `correct: false`;
it is never regenerated using label feedback.

## Output layout

Each trajectory is a separate valid JSON file inside a pseudonymous patient
folder:

```text
<output-dir>/
├── config.json
├── patient_manifest.json
├── summary.json
└── patients/
    └── case_<salted_hash>/
        ├── trajectory_001.json
        ├── trajectory_002.json
        └── trajectory_003.json
```

`patient_manifest.json` is the private mapping between source patient keys and
pseudonymous case IDs; it must still be handled as identifiable data. Each
trajectory stores the accepted request, retrieved SUV/PDF evidence, accepted
final prediction, all JSON-validation attempts, and the programmatic comparison
with documented treatment. The top-level `result` object is the convenient
final record:

```json
{
  "answer": "systemic_treatment",
  "reason": "Evidence-grounded explanation...",
  "evidence_ids": ["SUV-001", "LIT-001"],
  "observed_treatment": "source treatment text",
  "observed_management_category": "systemic_treatment",
  "correct": true
}
```

## First checks

Validate all input paths, labels, and patient/SUV matches without loading the
model:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/infer.py \
  --dry-run
```

Exercise two-stage JSON validation, forced first-attempt failures, retrieval,
per-trajectory files, and resume logic without loading Qwen:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/infer.py \
  --mock-model --mock-invalid-first \
  --max-patients 2 --num-trajectories 2 \
  --output-dir /tmp/pet_rag_mock
```

## Real-model smoke test

Run one patient and two independent trajectories before the full cohort:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/infer.py \
  --device cuda \
  --max-patients 1 \
  --num-trajectories 2 \
  --output-dir agentic_pca/retrieval_agent_inference/outputs/smoke_test
```

For the complete dataset:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/infer.py \
  --device cuda \
  --num-trajectories 5 \
  --output-dir agentic_pca/retrieval_agent_inference/outputs/full_run
```

Rerunning the same command skips completed files whose input/config
fingerprints match. Use `--overwrite` only when you intentionally want to
regenerate selected trajectories under the same inference configuration.
Changing the dataset, model, PDF corpus, prompts, or inference settings
requires a new output directory; the program refuses to mix configurations.

On this Slurm cluster, submit a one-patient GPU smoke test with:

```bash
sbatch --export=ALL,MAX_PATIENTS=1,NUM_TRAJECTORIES=1,OUTPUT_DIR=agentic_pca/retrieval_agent_inference/outputs/slurm_smoke \
  agentic_pca/retrieval_agent_inference/infer.slurm
```

After inspecting its JSON and log, submit the resumable full run:

```bash
sbatch agentic_pca/retrieval_agent_inference/infer.slurm
```

Useful controls include:

- `--max-json-retries 3`: retry count after the initial generation;
- `--max-suv-organs 6`: maximum exact organs requested per trajectory;
- `--literature-top-k 3`: PDF passages supplied to final prediction;
- `--temperature 0.7`: sampling diversity across trajectories;
- `--patient <exact-key>`: process selected patients (repeatable);
- `--invalid-report-policy fail|warn`: the dataset has one
  `System.Xml.XmlElement` report placeholder; the safe default writes failed
  input-validation trajectories instead of pretending a report was available;
- `--num-shards N --shard-index I`: split patients across independent GPU jobs.
  Give concurrent shards separate output directories to avoid manifest/summary
  write races.

## Leave-one-patient-out trajectory RAG

An existing inference output can be used as a retrieval corpus for a second
prediction pass. The corpus loader indexes every trajectory whose top-level
`status` is `completed`; it does not filter on `evaluation.correct`. Thus both
correct and incorrect valid predictions are available as historical examples,
whereas failed or incomplete trajectories are excluded.

Retrieval uses only the historical pretreatment `Report`, `Medical History`,
and PSA. Before ranking, every completed trajectory belonging to the current
patient is removed using the private patient manifest from the reference run.
The default hybrid score combines normalized word/character TF-IDF similarity
with log-transformed PSA similarity. After retrieval, historical organ choices
and literature queries guide the evidence-selection agent. The current
patient's SUV values and PDF passages are then retrieved again, and the final
agent receives both current evidence and compact historical trajectory
examples.

The default allows at most one retrieved trajectory per reference patient so
that repeated trajectories from one patient cannot fill the entire context.
Within an otherwise identical reference patient, trajectory-specific organ
choices and literature queries provide a label-free secondary tie-break.
Use a larger `--trajectory-rag-max-per-case` only when within-patient trajectory
variation is intentionally required.

The evidence-selection agent sees only historical patient inputs, selected
organs, literature queries, and PDF source/page hints. Historical SUV values
and PDF text are not transferred. The final prediction agent additionally sees
each retrieved trajectory's historical prediction, observed management
category, and whether that prediction was correct. These are labels from other
patients only; the current patient's outcome remains hidden until its new
prediction is frozen.

Validate the full reference corpus without loading the model:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/infer.py \
  --dry-run \
  --trajectory-rag-dir \
    agentic_pca/retrieval_agent_inference/outputs/full_run
```

Run a mock orchestration test:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/infer.py \
  --mock-model --mock-invalid-first \
  --trajectory-rag-dir \
    agentic_pca/retrieval_agent_inference/outputs/full_run \
  --trajectory-rag-top-k 5 \
  --max-patients 2 --num-trajectories 1 \
  --output-dir /tmp/pet_trajectory_rag_mock
```

Run all 249 patients with five new trajectories per patient:

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/retrieval_agent_inference/infer.py \
  --device cuda \
  --trajectory-rag-dir \
    agentic_pca/retrieval_agent_inference/outputs/full_run \
  --trajectory-rag-top-k 5 \
  --trajectory-rag-max-per-case 1 \
  --invalid-report-policy warn \
  --num-trajectories 5 \
  --output-dir \
    agentic_pca/retrieval_agent_inference/outputs/trajectory_rag_full_run
```

`--invalid-report-policy warn` is explicit here because one of the 249 records
contains an XML report placeholder; without it, that patient is retained as a
recorded input-validation failure rather than receiving a new prediction.
Never set `--output-dir` to the reference `--trajectory-rag-dir`.

The equivalent Slurm submission is:

```bash
sbatch agentic_pca/retrieval_agent_inference/trajectory_rag.slurm
```

## Label mapping

The fixed label mapping is versioned as `observed-management-v1`:

1. completed radical prostatectomy (including after neoadjuvant treatment);
2. any management containing ADT/ARPI, hormonal treatment, chemotherapy, or
   immunotherapy is systemic, including combinations with radiotherapy;
3. radiotherapy without a systemic component, focal ablation, and local
   resection are local treatment;
4. follow-up, biopsy, diagnostic/transurethral procedures, and symptomatic
   management are other examination.

The label denotes observed management, not necessarily optimal treatment.

## Imaging limitation

The current SUV files contain statistics from automatic anatomical masks, not
lesion segmentations. They contain no lymph-node lesion ROI. High uptake in
kidneys, bladder, liver, bone masks, or other normal structures must not be
treated as tumour solely from these values.
