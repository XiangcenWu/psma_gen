"""Download BioClinical ModernBERT for local registration conditioning.

Example:
    python llm_Registration/gen/download_bioclinical_modernbert.py ^
        --model_dir D:/models/BioClinical-ModernBERT-large
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_ID = "thomas-sounack/BioClinical-ModernBERT-large"


def download_model(
    model_dir: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    trust_remote_code: bool = False,
) -> Path:
    """Download tokenizer and model weights into ``model_dir``."""
    output_dir = Path(model_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download thomas-sounack/BioClinical-ModernBERT-large."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Local directory where the tokenizer and model weights are saved.",
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = download_model(
        model_dir=args.model_dir,
        model_id=args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Saved {args.model_id} to {output_dir}")


if __name__ == "__main__":
    main()
