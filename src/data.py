"""Dataset loading, subsampling and tokenisation.

Every dataset is turned into three splits with the same interface:

    train : used for gradient updates
    val   : held out slice of train, used only for the loss curves
    test  : the official evaluation split, used for the reported metrics
"""

from __future__ import annotations

from datasets import load_dataset

from config import DATASETS, SEED, DatasetSpec


def _subsample(split, size: int | None, seed: int = SEED):
    """Shuffle and keep at most size examples."""
    if size is None or size >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(size))


def load_splits(dataset_key: str, tokenizer, val_fraction: float = 0.05, seed: int = SEED):
    """Return tokenised train, val and test splits ready for a DataLoader."""
    spec: DatasetSpec = DATASETS[dataset_key]

    raw_train = load_dataset(spec.path, spec.name, split="train")
    raw_test = load_dataset(spec.path, spec.name, split=spec.test_split)

    raw_train = _subsample(raw_train, spec.train_size, seed)
    raw_test = _subsample(raw_test, spec.test_size, seed)

    # Hold out a slice of train for the training curves so the test split is
    # never touched during optimisation.
    split = raw_train.train_test_split(test_size=val_fraction, seed=seed)
    raw_train, raw_val = split["train"], split["test"]

    def tokenize(batch):
        return tokenizer(
            batch[spec.text_field],
            truncation=True,
            max_length=spec.max_length,
            padding="max_length",
        )

    keep = ["input_ids", "attention_mask", "labels"]
    out = {}
    for name, ds in (("train", raw_train), ("val", raw_val), ("test", raw_test)):
        ds = ds.map(tokenize, batched=True, batch_size=1000,
                    desc=f"tokenising {dataset_key}/{name}")
        ds = ds.rename_column("label", "labels")
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        ds.set_format(type="torch", columns=keep)
        out[name] = ds

    return out["train"], out["val"], out["test"], spec


def dataset_summary(dataset_key: str) -> dict:
    """Sizes and label names, used for the dataset table in the report."""
    spec = DATASETS[dataset_key]
    train = load_dataset(spec.path, spec.name, split="train")
    test = load_dataset(spec.path, spec.name, split=spec.test_split)
    return {
        "key": dataset_key,
        "path": spec.path if spec.name is None else f"{spec.path}/{spec.name}",
        "num_labels": spec.num_labels,
        "max_length": spec.max_length,
        "full_train": len(train),
        "full_test": len(test),
        "used_train": min(len(train), spec.train_size or len(train)),
        "used_test": min(len(test), spec.test_size or len(test)),
        "label_names": list(spec.label_names),
    }
