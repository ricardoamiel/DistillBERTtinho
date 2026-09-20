"""Training loop.

A plain PyTorch loop, kept short on purpose. It records the training loss and
the held out validation loss at a fixed number of checkpoints so the report can
plot iterations against both curves, and it tracks the peak GPU memory used
during optimisation.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import TrainConfig
from metrics import amp_ok, classification_metrics, predict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model, train_ds, val_ds, cfg: TrainConfig, device: str = "cuda",
                verbose: bool = True, test_ds=None):
    """Fine tune the model and return the loss history and timing information."""
    set_seed(cfg.seed)
    model.to(device)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        drop_last=True, pin_memory=True, num_workers=2)
    total_steps = len(loader) * cfg.epochs
    eval_every = max(total_steps // cfg.eval_points, 1)

    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    use_amp = amp_ok(cfg.amp, device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)

    val_probe = val_ds.select(range(min(len(val_ds), cfg.val_subset)))

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    history = {"step": [], "train_loss": [], "val_loss": [], "epoch": []}
    per_epoch = []
    running, running_n, step = 0.0, 0, 0
    start = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with autocast:
                logits = model(ids, mask)
                loss = loss_fn(logits.float(), y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            running_n += 1
            step += 1

            if step % eval_every == 0 or step == total_steps:
                _, _, val_loss = predict(model, val_probe, cfg.eval_batch_size, device, use_amp)
                model.train()
                history["step"].append(step)
                history["train_loss"].append(running / max(running_n, 1))
                history["val_loss"].append(val_loss)
                history["epoch"].append(step / len(loader))
                if verbose:
                    print(f"  step {step:>5}/{total_steps}  "
                          f"train {running / max(running_n, 1):.4f}  val {val_loss:.4f}",
                          flush=True)
                running, running_n = 0.0, 0

        if cfg.eval_each_epoch:
            # Model selection signal: full validation split, never the test split.
            logits, labels, val_loss = predict(model, val_ds, cfg.eval_batch_size, device, use_amp)
            scores = classification_metrics(logits, labels)
            row = {"epoch": epoch + 1, "step": step, "val_loss": val_loss,
                   "val_accuracy": scores["accuracy"], "val_f1_macro": scores["f1_macro"]}
            if test_ds is not None:
                t_logits, t_labels, t_loss = predict(model, test_ds, cfg.eval_batch_size,
                                                     device, use_amp)
                t_scores = classification_metrics(t_logits, t_labels)
                row.update({"test_loss": t_loss, "test_accuracy": t_scores["accuracy"],
                            "test_f1_macro": t_scores["f1_macro"]})
            per_epoch.append(row)
            if verbose:
                print(f"  [epoch {epoch + 1}] val f1 {row['val_f1_macro']:.4f}"
                      + (f"  test f1 {row.get('test_f1_macro', float('nan')):.4f}"
                         if test_ds is not None else ""), flush=True)
            model.train()

    train_seconds = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if device == "cuda" else float("nan")

    best_epoch = None
    if per_epoch:
        best_epoch = max(per_epoch, key=lambda r: r["val_f1_macro"])["epoch"]

    return {
        "history": history,
        "per_epoch": per_epoch,
        "best_epoch_by_val_f1": best_epoch,
        "train_seconds": train_seconds,
        "total_steps": total_steps,
        "steps_per_epoch": len(loader),
        "train_peak_gpu_mb": peak_mb,
        "effective_batch_size": cfg.batch_size,
    }
