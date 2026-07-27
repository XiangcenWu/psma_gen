#!/bin/bash
#SBATCH --job-name=dl_cache_check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_cache_check_%j.out
#SBATCH --error=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_cache_check_%j.err

set -euo pipefail

CACHE="${CACHE:-/data2/xiangcen/cache/pet_128x128x384}"

# The copied manifest contains absolute paths to the original /share cache.
# Rewrite them once on gpu02, where the node-local /data2 mount is available.
python - "${CACHE}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

cache = Path(sys.argv[1]).resolve()
manifest_path = cache / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Cache manifest is missing: {manifest_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
patients = manifest.get("patients")
if not isinstance(patients, list) or not patients:
    raise SystemExit(f"{manifest_path}: missing non-empty patients list")

missing = []
for record in patients:
    volume = cache / "volumes" / f"{record['case_id']}.npy"
    sidecar = volume.with_suffix(".json")
    record["cache_path"] = str(volume)
    for path in (volume, sidecar):
        if not path.is_file():
            missing.append(str(path))

if missing:
    preview = "\n".join(missing[:10])
    raise SystemExit(
        f"Cache is incomplete ({len(missing)} missing files):\n{preview}"
    )

fingerprint_payload = {
    key: value
    for key, value in manifest.items()
    if key not in {"created_at_utc", "manifest_fingerprint"}
}
encoded = json.dumps(
    fingerprint_payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
manifest["manifest_fingerprint"] = hashlib.sha256(encoded).hexdigest()

temporary = manifest_path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, manifest_path)
print(f"Validated and relocated {len(patients)} cached patients at {cache}")
PY
