"""One fine tuning run, from data loading to the saved result file."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from config import (ABLATION_BY_KEY, DATASETS, MODELS, RESULTS_DIR, HeadConfig,
                    TrainConfig)
from data import load_splits
from metrics import (classification_metrics, latency_benchmark,
                     measure_inference_memory, memory_report, predict)
from models import TextClassifier
from train import train_model


def environment_info() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": None,
        "gpu_total_mb": None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    return info


def run_id(model_key: str, dataset_key: str, head_key: str, tag: str = "") -> str:
    base = f"{model_key}__{dataset_key}__{head_key}"
    return f"{base}__{tag}" if tag else base


def run_one(model_key: str, dataset_key: str, head: HeadConfig,
            cfg: TrainConfig, device: str = "cuda",
            results_dir: str = RESULTS_DIR, overwrite: bool = False,
            save_encoder: bool = False, tag: str = "") -> dict:
    """Fine tune one model on one dataset with one classifier head."""
    rid = run_id(model_key, dataset_key, head.key, tag)
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{rid}.json")
    encoder_dir = os.path.join("artifacts", "encoders", rid)
    # A finished run is only complete if the encoder it was asked to keep is there too.
    done = os.path.exists(path) and (not save_encoder or os.path.isdir(encoder_dir))
    if done and not overwrite:
        print(f"[skip] {rid} already done", flush=True)
        with open(path) as fh:
            return json.load(fh)

    print(f"\n=== {rid} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key])
    train_ds, val_ds, test_ds, spec = load_splits(
        dataset_key, tokenizer, val_fraction=cfg.val_fraction, seed=cfg.seed)
    print(f"  train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}", flush=True)

    model = TextClassifier(
        backbone_key=model_key,
        num_labels=spec.num_labels,
        hidden_sizes=head.hidden_sizes,
        dropout=head.dropout,
        frozen_layers=head.frozen_layers,
    )
    params = model.parameter_counts()
    print(f"  params total {params['total_params']:,}  "
          f"trainable {params['trainable_params']:,}", flush=True)

    train_info = train_model(model, train_ds, val_ds, cfg, device=device,
                             test_ds=test_ds if cfg.eval_each_epoch else None)

    logits, labels, test_loss = predict(model, test_ds, cfg.eval_batch_size, device, cfg.amp)
    scores = classification_metrics(logits, labels)

    infer_mb = measure_inference_memory(model, test_ds, cfg.eval_batch_size, device)
    latency = latency_benchmark(model, seq_len=spec.max_length, device=device)

    result = {
        "run_id": rid,
        "model": model_key,
        "checkpoint": MODELS[model_key],
        "dataset": dataset_key,
        "head": asdict(head),
        "train_config": asdict(cfg),
        "dataset_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "num_labels": spec.num_labels,
        "label_names": list(spec.label_names),
        "max_length": spec.max_length,
        "params": params,
        "metrics": scores,
        "test_loss": test_loss,
        "training": train_info,
        "latency": latency,
        "memory": {**memory_report(model, device), "inference_peak_mb": infer_mb},
        "environment": environment_info(),
    }

    if save_encoder:
        out_dir = encoder_dir
        os.makedirs(out_dir, exist_ok=True)
        model.backbone.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        torch.save(model.head.state_dict(), os.path.join(out_dir, "head.pt"))
        result["encoder_dir"] = out_dir

    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  accuracy {scores['accuracy']:.4f}  f1_macro {scores['f1_macro']:.4f}  "
          f"saved {path}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result
