# PDF RAG Agent

This local RAG agent searches every PDF under `agentic_pca/agent_dataset/papers`, retrieves the most relevant passages, and asks the local Qwen model to produce one evidence-grounded English paragraph with inline source labels.

The first run extracts and caches the PDF text. Later runs reuse the cache unless a PDF changes or `--rebuild-index` is supplied.

```bash
/share/home/xcwu/miniconda3/envs/gen/bin/python \
  agentic_pca/pdf_rag/pdf_rag_agent.py \
  --device cuda \
  --question "How should PSMA PET findings influence treatment selection in recurrent prostate cancer?" \
  --show-sources
```

The generation prompt is available as `SYSTEM_PROMPT` near the top of `pdf_rag_agent.py`. It requires exactly one English paragraph, restricts the answer to retrieved PDF evidence, requires inline source labels, and explicitly handles insufficient or conflicting evidence.

Useful options:

- `--top-k 8`: number of retrieved passages passed to the agent.
- `--rebuild-index`: force PDF text extraction after changing extraction settings.
- `--device auto`: use CUDA when available and otherwise use the CPU.
- `--show-sources`: print retrieved file names, pages, and retrieval scores before the answer.
