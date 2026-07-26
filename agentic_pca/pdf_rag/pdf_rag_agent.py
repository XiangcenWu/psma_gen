#!/usr/bin/env python3
"""Retrieve evidence from every PDF and generate one evidence-grounded paragraph."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_DIR = ROOT / "agentic_pca/agent_dataset/papers"
DEFAULT_MODEL = ROOT / "llm_models/Qwen3.5-9B"
DEFAULT_CACHE = Path(__file__).resolve().parent / "cache/pdf_chunks.json"

SYSTEM_PROMPT = """You are an evidence-grounded prostate cancer research assistant.
Answer the user's question using only the retrieved passages from the supplied PDF collection.
Synthesize the evidence into exactly one concise English paragraph. Do not use bullet points or
headings. Do not invent facts, recommendations, study results, or citations. When the evidence is
insufficient or conflicting, state that limitation explicitly."""


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int


def pdf_manifest(pdf_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(pdf_dir)),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(pdf_dir.rglob("*.pdf"))
    ]


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_page(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = clean_text(text).split()
    if not words:
        return []
    step = chunk_words - overlap_words
    return [" ".join(words[start : start + chunk_words]) for start in range(0, len(words), step)]


def extract_pdf(path: Path, pdf_dir: Path, chunk_words: int, overlap_words: int) -> list[Chunk]:
    process = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if process.returncode != 0:
        print(f"Warning: could not extract {path}: {process.stderr.strip()}", flush=True)
        return []
    source = str(path.relative_to(pdf_dir))
    chunks = []
    for page_number, page_text in enumerate(process.stdout.split("\f"), 1):
        for text in split_page(page_text, chunk_words, overlap_words):
            chunks.append(Chunk(text=text, source=source, page=page_number))
    return chunks


def load_or_build_chunks(
    pdf_dir: Path,
    cache_path: Path,
    chunk_words: int,
    overlap_words: int,
    rebuild: bool,
) -> list[Chunk]:
    manifest = pdf_manifest(pdf_dir)
    if not manifest:
        raise FileNotFoundError(f"No PDF files were found under {pdf_dir}")
    if cache_path.exists() and not rebuild:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            payload.get("manifest") == manifest
            and payload.get("chunk_words") == chunk_words
            and payload.get("overlap_words") == overlap_words
        ):
            return [Chunk(**item) for item in payload["chunks"]]

    chunks = []
    for index, item in enumerate(manifest, 1):
        path = pdf_dir / str(item["path"])
        print(f"Extracting PDF {index}/{len(manifest)}: {item['path']}", flush=True)
        chunks.extend(extract_pdf(path, pdf_dir, chunk_words, overlap_words))
    if not chunks:
        raise RuntimeError("PDF extraction produced no searchable text.")
    payload = {
        "manifest": manifest,
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "chunks": [chunk.__dict__ for chunk in chunks],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return chunks


class PdfRetriever:
    """A local sparse-vector retriever over chunks from the complete PDF collection."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = FeatureUnion(
            [
                ("words", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
                ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
            ]
        )
        documents = [chunk.text for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        count = min(top_k, len(self.chunks))
        best = np.argpartition(scores, -count)[-count:]
        best = best[np.argsort(scores[best])[::-1]]
        return [(self.chunks[index], float(scores[index])) for index in best]


def load_model(model_path: Path, requested_device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def build_user_prompt(question: str, results: list[tuple[Chunk, float]]) -> str:
    passages = []
    for index, (chunk, _) in enumerate(results, 1):
        passages.append(f"[S{index}] Source: {chunk.source}, page {chunk.page}\n{chunk.text}")
    return f"Question: {question}\n\nRetrieved evidence:\n\n" + "\n\n".join(passages)


def generate_paragraph(tokenizer, model, prompt: str, max_input_tokens: int, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    paragraph = tokenizer.decode(generated, skip_special_tokens=True).strip()
    paragraph = re.sub(r"\s*\[(?:S\d+)(?:\s*(?:,|;|-|–)\s*S?\d+)*\]", "", paragraph)
    paragraph = re.sub(r"\s+([.,;:!?])", r"\1", paragraph)
    return re.sub(r"\s*\n+\s*", " ", paragraph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="The research question, written in English.")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--chunk-words", type=int, default=350)
    parser.add_argument("--overlap-words", type=int, default=70)
    parser.add_argument("--max-input-tokens", type=int, default=12000)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--show-sources", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1 or args.chunk_words <= args.overlap_words:
        raise ValueError("top-k must be positive and chunk-words must exceed overlap-words.")
    chunks = load_or_build_chunks(
        args.pdf_dir,
        args.cache_path,
        args.chunk_words,
        args.overlap_words,
        args.rebuild_index,
    )
    print(f"Indexed {len(chunks)} chunks from {len(pdf_manifest(args.pdf_dir))} PDF files.", flush=True)
    results = PdfRetriever(chunks).search(args.question, args.top_k)
    if args.show_sources:
        for index, (chunk, score) in enumerate(results, 1):
            print(f"S{index}: {chunk.source}, page {chunk.page}, score={score:.4f}")
    tokenizer, model = load_model(args.model_path, args.device)
    answer = generate_paragraph(
        tokenizer,
        model,
        build_user_prompt(args.question, results),
        args.max_input_tokens,
        args.max_new_tokens,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
