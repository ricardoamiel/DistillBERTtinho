"""Performance and efficiency measurements.

Performance: accuracy, precision, recall and F1.
Efficiency: parameter count, inference latency and GPU memory use.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader


def amp_ok(enabled: bool, device: str) -> bool:
    """Use bfloat16 autocast only where the GPU supports it natively."""
    return bool(enabled) and device == "cuda" and torch.cuda.is_available() \
        and torch.cuda.is_bf16_supported()


@torch.no_grad()
def predict(model, dataset, batch_size: int = 64, device: str = "cuda", amp: bool = True):
    """Run the model over a dataset and return logits and gold labels."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    logits, labels, total_loss, n = [], [], 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_ok(amp, device))
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with autocast:
            out = model(ids, mask)
        out = out.float()
        total_loss += loss_fn(out, y).item()
        n += y.numel()
        logits.append(out.cpu())
        labels.append(y.cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), total_loss / max(n, 1)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy plus macro and weighted precision, recall and F1."""
    preds = logits.argmax(axis=1)
    macro = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "per_class": {
            "precision": per_class[0].tolist(),
            "recall": per_class[1].tolist(),
            "f1": per_class[2].tolist(),
            "support": per_class[3].tolist(),
        },
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def _timed_forward(model, ids, mask, reps: int, device: str) -> list[float]:
    times = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(ids, mask)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def latency_benchmark(model, seq_len: int, device: str = "cuda",
                      reps: int = 200, warmup: int = 20,
                      throughput_batch: int = 32, cpu_reps: int = 30) -> dict:
    """Single example latency and batched throughput, in float32 for a fair clock."""
    model.eval()
    vocab = model.config.vocab_size

    def make(batch):
        ids = torch.randint(0, vocab, (batch, seq_len), device=device)
        mask = torch.ones_like(ids)
        return ids, mask

    ids, mask = make(1)
    _timed_forward(model, ids, mask, warmup, device)
    single = _timed_forward(model, ids, mask, reps, device)

    ids_b, mask_b = make(throughput_batch)
    _timed_forward(model, ids_b, mask_b, warmup, device)
    batched = _timed_forward(model, ids_b, mask_b, max(reps // 4, 10), device)

    result = {
        "seq_len": seq_len,
        "latency_ms_p50": float(np.percentile(single, 50)),
        "latency_ms_p95": float(np.percentile(single, 95)),
        "latency_ms_mean": float(np.mean(single)),
        "batch_size": throughput_batch,
        "batched_latency_ms_p50": float(np.percentile(batched, 50)),
        "throughput_samples_per_s": float(throughput_batch / (np.median(batched) / 1000.0)),
    }

    if device == "cuda" and cpu_reps > 0:
        model_cpu = model.to("cpu")
        ids_c = torch.randint(0, vocab, (1, seq_len))
        mask_c = torch.ones_like(ids_c)
        _timed_forward(model_cpu, ids_c, mask_c, 5, "cpu")
        cpu_times = _timed_forward(model_cpu, ids_c, mask_c, cpu_reps, "cpu")
        result["cpu_latency_ms_p50"] = float(np.percentile(cpu_times, 50))
        model.to(device)

    return result


def memory_report(model, device: str = "cuda") -> dict:
    """Weight footprint and peak allocation recorded so far."""
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    report = {"weights_mb": weight_bytes / 1024 ** 2}
    if device == "cuda" and torch.cuda.is_available():
        report["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        report["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 ** 2
    return report


def measure_inference_memory(model, dataset, batch_size: int, device: str = "cuda") -> float:
    """Peak GPU memory of a short inference pass, in MiB."""
    if device != "cuda" or not torch.cuda.is_available():
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    small = dataset.select(range(min(len(dataset), batch_size * 8)))
    predict(model, small, batch_size=batch_size, device=device, amp=False)
    return torch.cuda.max_memory_allocated() / 1024 ** 2
