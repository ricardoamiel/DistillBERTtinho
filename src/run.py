"""Command line entry point for every experiment in the project.

Examples:

    python src/run.py --mode main
    python src/run.py --mode ablation
    python src/run.py --mode single --model bert --dataset sst2 --head A_baseline
"""

from __future__ import annotations

import argparse
import json
import os

from config import (ABLATION_BY_KEY, ABLATION_DATASETS, ABLATIONS, DATASETS,
                    EPOCH_DATASETS, EPOCH_STUDY_EPOCHS, MAIN_DATASETS, MODELS,
                    RESULTS_DIR, TrainConfig)
from experiment import run_one


def build_config(args) -> TrainConfig:
    cfg = TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    return cfg


def best_head_key(results_dir: str, model_key: str = "distilbert",
                  datasets: list[str] | None = None) -> str:
    """Pick the ablation variant with the highest mean macro F1 across datasets."""
    datasets = datasets or ABLATION_DATASETS
    scores: dict[str, list[float]] = {}
    for cfg in ABLATIONS:
        values = []
        for ds in datasets:
            path = os.path.join(results_dir, f"{model_key}__{ds}__{cfg.key}.json")
            if os.path.exists(path):
                with open(path) as fh:
                    values.append(json.load(fh)["metrics"]["f1_macro"])
        if len(values) == len(datasets):
            scores[cfg.key] = values
    if len(scores) < len(ABLATIONS):
        raise RuntimeError(
            f"ablation incomplete: {len(scores)} of {len(ABLATIONS)} variants have results "
            f"for {datasets}. Run --mode ablation first.")
    return max(scores, key=lambda k: sum(scores[k]) / len(scores[k]))


def main() -> None:
    parser = argparse.ArgumentParser(description="DistilBERT vs BERT text classification")
    parser.add_argument("--mode",
                        choices=["main", "ablation", "single", "best_on_bert", "epochs"],
                        default="main")
    parser.add_argument("--model", choices=list(MODELS), default="distilbert")
    parser.add_argument("--dataset", choices=list(DATASETS), default="sst2")
    parser.add_argument("--head", default="A_baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = build_config(args)
    baseline = ABLATION_BY_KEY["A_baseline"]
    common = dict(cfg=cfg, device=args.device,
                  results_dir=args.results_dir, overwrite=args.overwrite)

    if args.mode == "single":
        run_one(args.model, args.dataset, ABLATION_BY_KEY[args.head], **common)

    elif args.mode == "main":
        # Headline comparison: both backbones, every dataset, identical head.
        # Every encoder is kept on disk so the interactive demo can show the
        # latent space of each dataset, not just one.
        for dataset in MAIN_DATASETS:
            for model in ["distilbert", "bert"]:
                run_one(model, dataset, baseline, save_encoder=True, **common)

    elif args.mode == "ablation":
        # Classifier ablation on DistilBERT only.
        for dataset in ABLATION_DATASETS:
            for head in ABLATIONS:
                run_one("distilbert", dataset, head, **common)

    elif args.mode == "epochs":
        # How many epochs are actually needed. Devlin et al. recommend 2, 3 or 4,
        # so we train to 4 and score validation and test at every epoch boundary.
        cfg_ep = build_config(args)
        cfg_ep.epochs = EPOCH_STUDY_EPOCHS
        cfg_ep.eval_each_epoch = True
        for dataset in EPOCH_DATASETS:
            for model in ["distilbert", "bert"]:
                run_one(model, dataset, baseline, cfg=cfg_ep, device=args.device,
                        results_dir=args.results_dir, overwrite=args.overwrite,
                        tag="ep4")

    elif args.mode == "best_on_bert":
        # The winning DistilBERT variant, now with BERT as the backbone.
        key = best_head_key(args.results_dir)
        print(f"best ablation variant: {key}", flush=True)
        for dataset in ABLATION_DATASETS:
            run_one("bert", dataset, ABLATION_BY_KEY[key], **common)


if __name__ == "__main__":
    main()
