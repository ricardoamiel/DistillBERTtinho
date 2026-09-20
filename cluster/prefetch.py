"""Download every checkpoint and dataset into the shared cache.

Run this on the login node, which has internet access, so that the compute
node can work offline.
"""

import sys

sys.path.insert(0, "src")

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from config import DATASETS, MODELS

for key, checkpoint in MODELS.items():
    print(f"model {checkpoint}", flush=True)
    AutoTokenizer.from_pretrained(checkpoint)
    AutoModel.from_pretrained(checkpoint)

for key, spec in DATASETS.items():
    print(f"dataset {key} ({spec.path})", flush=True)
    for split in ("train", spec.test_split):
        ds = load_dataset(spec.path, spec.name, split=split)
        print(f"   {split}: {len(ds)} rows", flush=True)

print("prefetch complete")
