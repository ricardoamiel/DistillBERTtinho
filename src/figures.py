"""Build every figure and LaTeX table used in the report.

Reads the JSON files written by run.py and writes vector graphics (PDF and SVG)
into artifacts/figures plus LaTeX fragments into artifacts/tables.

Colour is used for one thing only, the backbone: blue is DistilBERT, orange is
BERT. Everything else is carried by position, panel or line style.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (ABLATIONS, ABLATION_BY_KEY, ABLATION_DATASETS, DATASET_LABELS, DATASETS,
                    EPOCH_DATASETS, EPOCH_STUDY_EPOCHS, FIGURES_DIR, MAIN_DATASETS,
                    RESULTS_DIR)

TABLES_DIR = "artifacts/tables"
WEB_DIR = "web/data"   # overridden by the dry run so it never touches real data

# Validated categorical slots: blue, orange, aqua, yellow.
C_DISTIL = "#2a78d6"
C_BERT = "#eb6834"
C_AXIS = {"baseline": "#2a78d6", "freezing": "#eb6834", "width": "#1baf7a", "depth": "#eda100"}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"

MODEL_COLOR = {"distilbert": C_DISTIL, "bert": C_BERT}
MODEL_LABEL = {"distilbert": "DistilBERT", "bert": "BERT"}
DATASET_LABEL = DATASET_LABELS   # defined once in config.py
# Which ablation axis each variant probes, used only for colour.
ABLATION_AXIS = {
    "A_baseline": "baseline", "B_frozen_all": "freezing", "C_frozen_half": "freezing",
    "D_narrow": "width", "E_wide": "width", "F_deep": "depth", "G_linear": "depth",
}
ABLATION_SHORT = {
    "A_baseline": "A: 1x768 full",
    "B_frozen_all": "B: frozen backbone",
    "C_frozen_half": "C: 3 blocks frozen",
    "D_narrow": "D: 1x128",
    "E_wide": "E: 1x2048",
    "F_deep": "F: 3x768",
    "G_linear": "G: no hidden layer",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def bar_values(ax, bars, values, fmt="{:.1f}", size=5.8):
    """Write each bar's value above it, turned upright so neighbours cannot collide."""
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", rotation=90,
                    ha="center", va="bottom", fontsize=size, color=INK_SOFT)


def legend_below(fig, labels_colors, ncols=None, y=-0.04, marker="s", size=7):
    """One legend under the whole figure, never inside an axes where it can cover data."""
    handles = [plt.Line2D([], [], marker=marker, ls="", color=c, label=n)
               for n, c in labels_colors]
    fig.legend(handles=handles, ncols=ncols or len(handles), loc="lower center",
               bbox_to_anchor=(0.5, y), fontsize=size, handletextpad=0.4,
               columnspacing=1.4)


def tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)


def save(fig, name: str) -> None:
    """PDF and SVG for the report, PNG so the figures render on GitHub."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"), format=ext)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), format="png", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf, {name}.svg and {name}.png", flush=True)


# --------------------------------------------------------------------------
# result loading
# --------------------------------------------------------------------------

def load_results(results_dir: str = RESULTS_DIR) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        out[r["run_id"]] = r
    return out


def get(results: dict, model: str, dataset: str, head: str = "A_baseline"):
    return results.get(f"{model}__{dataset}__{head}")


def available_main(results: dict) -> list[str]:
    return [d for d in MAIN_DATASETS
            if get(results, "distilbert", d) and get(results, "bert", d)]


def best_head(results: dict, datasets: list[str] | None = None) -> str | None:
    datasets = datasets or ABLATION_DATASETS
    scored = {}
    for cfg in ABLATIONS:
        vals = [get(results, "distilbert", d, cfg.key) for d in datasets]
        if all(vals):
            scored[cfg.key] = float(np.mean([v["metrics"]["f1_macro"] for v in vals]))
    # Only decide once every variant has been measured, otherwise a partially
    # finished ablation would silently crown whichever variant happened to run.
    if len(scored) < len(ABLATIONS):
        return None
    return max(scored, key=scored.get)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_params_accuracy_bubble(results: dict) -> None:
    """Required bubble chart: parameters against accuracy, bubble area is latency.

    Accuracies span roughly 65 to 97 percent, so a single linear axis squashes the
    three easy datasets into one another. The y axis is therefore broken, with the
    same scale in both segments.
    """
    datasets = available_main(results)
    if not datasets:
        return

    points = {}
    for d in datasets:
        points[d] = [(get(results, m, d)["params"]["total_params"] / 1e6,
                      get(results, m, d)["metrics"]["accuracy"] * 100,
                      get(results, m, d)["latency"]["latency_ms_p50"])
                     for m in ("distilbert", "bert")]
    accs = [p[1] for pair in points.values() for p in pair]
    lat = [p[2] for pair in points.values() for p in pair]
    lo, hi = min(lat), max(lat)

    def area(ms):
        return 90 + 520 * (ms - lo) / max(hi - lo, 1e-9)

    # Split the datasets into a high band and a low band if there is a real gap.
    ordered = sorted(accs)
    gaps = [(ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1)]
    widest, at = max(gaps) if gaps else (0, 0)
    split = widest > 8

    if split:
        cut = (ordered[at] + ordered[at + 1]) / 2
        fig, (top, bot) = plt.subplots(
            2, 1, sharex=True, figsize=(5.4, 2.3),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
        axes = [top, bot]
        top.set_ylim(min(a for a in accs if a > cut) - 1.2, max(accs) + 1.2)
        bot.set_ylim(min(accs) - 1.2, max(a for a in accs if a < cut) + 1.2)
        top.spines["bottom"].set_visible(False)
        bot.spines["top"].set_visible(False)
        top.tick_params(labelbottom=False, bottom=False)
        # the diagonal marks that say the axis is broken
        kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=5, linestyle="none",
                  color=INK_SOFT, mec=INK_SOFT, mew=1, clip_on=False)
        top.plot([0, 1], [0, 0], transform=top.transAxes, **kw)
        bot.plot([0, 1], [1, 1], transform=bot.transAxes, **kw)
    else:
        fig, ax = plt.subplots(figsize=(5.4, 2.35))
        axes = [ax]
        cut = -1

    for d, pair in points.items():
        ax = axes[0] if (not split or pair[0][1] > cut) else axes[1]
        ax.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]],
                color=GRID, lw=1.0, zorder=1)
        for (x, y, ms), m in zip(pair, ("distilbert", "bert")):
            ax.scatter(x, y, s=area(ms), color=MODEL_COLOR[m], alpha=0.78,
                       edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(f"{pair[0][1]:.1f}", (pair[0][0], pair[0][1]), xytext=(-25, -2.5),
                    textcoords="offset points", fontsize=6.5, color=INK_SOFT)
        ax.annotate(f"{pair[1][1]:.1f}", (pair[1][0], pair[1][1]), xytext=(10, -8),
                    textcoords="offset points", fontsize=6.5, color=INK_SOFT)
        ax.annotate(DATASET_LABEL[d], (pair[1][0], pair[1][1]), xytext=(16, 3),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK)

    for ax in axes:
        ax.set_xlim(52, 162)
        tidy(ax)
    axes[-1].set_xlabel("Total parameters (millions)")
    fig.supylabel("Test accuracy (%)", fontsize=8, x=0.015)
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                          color=MODEL_COLOR[m], label=MODEL_LABEL[m])
               for m in ("distilbert", "bert")]
    fig.legend(handles=handles, ncols=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.17), fontsize=7.5)
    if split:
        fig.subplots_adjust(left=0.11, right=0.99, top=0.97, bottom=0.17)
    else:
        fig.tight_layout()
    save(fig, "fig1_params_vs_accuracy_bubble")


def fig_performance(results: dict) -> None:
    """Accuracy, precision, recall and F1 side by side for every dataset."""
    datasets = available_main(results)
    if not datasets:
        return
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    names = ["Accuracy", "Precision", "Recall", "F1"]
    ncol = 2
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4, 2.0 * nrow), squeeze=False,
                             layout="constrained")

    for ax, d in zip(axes.flat, datasets):
        x = np.arange(len(keys))
        lo = min(get(results, m, d)["metrics"][k] * 100
                 for m in ("distilbert", "bert") for k in keys)
        for i, m in enumerate(("distilbert", "bert")):
            vals = [get(results, m, d)["metrics"][k] * 100 for k in keys]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=MODEL_COLOR[m])
            bar_values(ax, bars, vals, "{:.1f}")
        # headroom for the upright labels
        ax.set_ylim(max(0, lo - 6), 100 + (100 - max(0, lo - 6)) * 0.16)
        ax.set_xticks(x, names, fontsize=7)
        ax.set_title(DATASET_LABEL[d], fontsize=8.5)
        ax.set_ylabel("%", fontsize=7.5)
        ax.tick_params(labelsize=7)
        tidy(ax)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    legend_below(fig, [(MODEL_LABEL[m], MODEL_COLOR[m]) for m in ("distilbert", "bert")])
    save(fig, "fig2_performance_metrics")


def fig_loss_curves(results: dict) -> None:
    """Iterations against training and validation loss."""
    datasets = available_main(results)
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4, 1.65), squeeze=False,
                             layout="constrained")
    for i, (ax, d) in enumerate(zip(axes.flat, datasets)):
        for m in ("distilbert", "bert"):
            h = get(results, m, d)["training"]["history"]
            ax.plot(h["step"], h["train_loss"], color=MODEL_COLOR[m], ls="-", lw=1.3)
            ax.plot(h["step"], h["val_loss"], color=MODEL_COLOR[m], ls="--", lw=1.3, alpha=0.85)
        ax.set_title(DATASET_LABEL[d], fontsize=8)
        ax.set_xlabel("Iterations", fontsize=7)
        if i == 0:
            ax.set_ylabel("Loss", fontsize=7)
        ax.tick_params(labelsize=6.5)
        tidy(ax)
    handles = [plt.Line2D([], [], color=MODEL_COLOR[m], label=MODEL_LABEL[m])
               for m in ("distilbert", "bert")]
    handles += [plt.Line2D([], [], color=INK_SOFT, ls="-", label="training"),
                plt.Line2D([], [], color=INK_SOFT, ls="--", label="validation")]
    fig.legend(handles=handles, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.17), fontsize=7)
    save(fig, "fig3_loss_curves")


def fig_efficiency(results: dict) -> None:
    """Latency, throughput, GPU memory and training time."""
    datasets = available_main(results)
    if not datasets:
        return
    panels = [
        ("GPU latency\nbatch 1 (ms)", lambda r: r["latency"]["latency_ms_p50"], "{:.1f}"),
        ("Throughput\nbatch 32 (samples/s)", lambda r: r["latency"]["throughput_samples_per_s"], "{:.0f}"),
        ("Peak GPU memory\ntraining (MiB)", lambda r: r["training"]["train_peak_gpu_mb"], "{:.0f}"),
        ("Wall clock\ntraining (min)", lambda r: r["training"]["train_seconds"] / 60, "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(5.4, 2.1), layout="constrained")
    x = np.arange(len(datasets))
    for ax, (title, fn, fmt) in zip(axes.flat, panels):
        top = 0
        for i, m in enumerate(("distilbert", "bert")):
            vals = [fn(get(results, m, d)) for d in datasets]
            top = max(top, max(vals))
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=MODEL_COLOR[m])
            bar_values(ax, bars, vals, fmt)
        ax.set_ylim(0, top * 1.38)          # room for the upright labels
        ax.set_xticks(x, [DATASET_LABEL[d] for d in datasets], rotation=40,
                      ha="right", fontsize=6)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6.5)
        tidy(ax)
    legend_below(fig, [(MODEL_LABEL[m], MODEL_COLOR[m]) for m in ("distilbert", "bert")], y=-0.09)
    save(fig, "fig4_efficiency")


def fig_ablation(results: dict) -> None:
    """Classifier ablation on DistilBERT."""
    datasets = [d for d in ABLATION_DATASETS
                if all(get(results, "distilbert", d, c.key) for c in ABLATIONS)]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets) + 1, figsize=(5.4, 2.15),
                             gridspec_kw={"width_ratios": [1.15] * len(datasets) + [1.0]},
                             layout="constrained")
    axes = np.atleast_1d(axes)
    order = [c.key for c in ABLATIONS]

    for ax, d in zip(axes, datasets):
        vals = [get(results, "distilbert", d, k)["metrics"]["f1_macro"] * 100 for k in order]
        colors = [C_AXIS[ABLATION_AXIS[k]] for k in order]
        y = np.arange(len(order))
        ax.barh(y, vals, color=colors, height=0.68)
        span = max(vals) - min(vals)
        for yi, v in zip(y, vals):
            ax.text(v + span * 0.03, yi, f"{v:.1f}", va="center", fontsize=6.2, color=INK_SOFT)
        ax.set_yticks(y, [ABLATION_SHORT[k] for k in order], fontsize=6.2)
        ax.invert_yaxis()
        ax.set_xlim(min(vals) - span * 0.12, max(vals) + span * 0.26)
        ax.set_xlabel("F1 macro (%)", fontsize=7)
        ax.set_title(DATASET_LABEL[d], fontsize=8.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # cost against quality, with the point labels pushed apart so none is hidden
    ax = axes[-1]
    pts = []
    for k in order:
        r = get(results, "distilbert", datasets[0], k)
        mean_f1 = float(np.mean([get(results, "distilbert", d, k)["metrics"]["f1_macro"] * 100
                                 for d in datasets]))
        pts.append([r["params"]["trainable_params"] / 1e6, mean_f1, k.split("_")[0], k])
    for x, y, _, k in pts:
        ax.scatter(x, y, s=50, color=C_AXIS[ABLATION_AXIS[k]], edgecolor="white", linewidth=1.0)
    # Labels get their own y positions, pushed apart from the bottom up, and a thin
    # leader line back to the point they belong to.
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = max(ys) - min(ys)
    gap = 0.085 * max(span, 1e-6)
    x_off = 0.12 * max(max(xs) - min(xs), 1.0)

    placed, prev = [], None
    for x, y, label, _ in sorted(pts, key=lambda p: p[1]):
        ly = y if prev is None else max(y, prev + gap)
        placed.append((x, y, ly, label))
        prev = ly

    for x, y, ly, label in placed:
        ax.annotate(label, xy=(x, y), xytext=(x + x_off, ly), textcoords="data",
                    fontsize=6.2, color=INK_SOFT, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6,
                                    shrinkA=0, shrinkB=3))
    top = max(max(ys), max(p[2] for p in placed))
    ax.set_ylim(min(ys) - span * 0.12, top + span * 0.12)
    ax.set_xlim(-8, max(xs) * 1.55)
    ax.set_xlabel("Trainable parameters (M)", fontsize=7)
    ax.set_ylabel("Mean F1 macro (%)", fontsize=7)
    ax.set_title("Cost against quality", fontsize=8.5)
    ax.tick_params(labelsize=6.5)
    tidy(ax)

    legend_below(fig, [(n.capitalize(), c) for n, c in C_AXIS.items()], y=-0.11)
    save(fig, "fig5_ablation")


def fig_best_on_bert(results: dict) -> None:
    """The winning classifier variant, DistilBERT against BERT."""
    key = best_head(results)
    if key is None:
        return
    datasets = [d for d in ABLATION_DATASETS
                if get(results, "distilbert", d, key) and get(results, "bert", d, key)]
    if not datasets:
        return
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision"),
               ("recall_macro", "Recall"), ("f1_macro", "F1")]
    fig, axes = plt.subplots(1, len(datasets) + 1, figsize=(5.4, 2.3), layout="constrained")
    axes = np.atleast_1d(axes)

    for ax, d in zip(axes, datasets):
        x = np.arange(len(metrics))
        lo = min(get(results, m, d, key)["metrics"][k] * 100
                 for m in ("distilbert", "bert") for k, _ in metrics)
        for i, m in enumerate(("distilbert", "bert")):
            vals = [get(results, m, d, key)["metrics"][k] * 100 for k, _ in metrics]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=MODEL_COLOR[m])
            bar_values(ax, bars, vals, "{:.1f}")
        base = max(0, lo - 4)
        ax.set_ylim(base, base + (100 - base) * 1.2)
        ax.set_xticks(x, [n for _, n in metrics], rotation=35, ha="right", fontsize=6.5)
        ax.set_title(DATASET_LABEL[d], fontsize=8.5)
        ax.tick_params(axis="y", labelsize=6.5)
        tidy(ax)

    ax = axes[-1]
    d = datasets[0]
    bars_data = [
        ("Latency (ms)", lambda r: r["latency"]["latency_ms_p50"]),
        ("GPU mem (GiB)", lambda r: r["training"]["train_peak_gpu_mb"] / 1024),
        ("Train (min)", lambda r: r["training"]["train_seconds"] / 60),
    ]
    x = np.arange(len(bars_data))
    top = 0
    for i, m in enumerate(("distilbert", "bert")):
        r = get(results, m, d, key)
        vals = [fn(r) for _, fn in bars_data]
        top = max(top, max(vals))
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=MODEL_COLOR[m])
        bar_values(ax, bars, vals, "{:.1f}")
    ax.set_ylim(0, top * 1.4)
    ax.set_xticks(x, [n for n, _ in bars_data], rotation=35, ha="right", fontsize=6.5)
    ax.set_title("Cost on " + DATASET_LABEL[d], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=6.5)
    tidy(ax)

    fig.suptitle(f"Best classifier variant: {ABLATION_SHORT.get(key, key)}", fontsize=8.5)
    legend_below(fig, [(MODEL_LABEL[m], MODEL_COLOR[m]) for m in ("distilbert", "bert")], y=-0.07)
    save(fig, "fig6_best_variant_on_bert")


def fig_confusion(results: dict) -> None:
    """Confusion matrices on AG News, one panel per backbone."""
    if not get(results, "distilbert", "ag_news"):
        return
    names = DATASETS["ag_news"].label_names
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), layout="constrained")
    for ax, m in zip(axes, ("distilbert", "bert")):
        cm = np.array(get(results, m, "ag_news")["metrics"]["confusion_matrix"], dtype=float)
        cm = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=9.5,
                        color="white" if cm[i, j] > 55 else INK)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=9)
        # only the left panel carries the class names, so nothing collides in the middle
        if m == "distilbert":
            ax.set_yticks(range(len(names)), names, fontsize=9)
            ax.set_ylabel("True class", fontsize=9.5)
        else:
            ax.set_yticks(range(len(names)), [""] * len(names))
        ax.set_xlabel("Predicted class", fontsize=9.5)
        ax.set_title(MODEL_LABEL[m], fontsize=11.5)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cb.set_label("Row percentage", fontsize=9.5)
    cb.ax.tick_params(labelsize=9)
    save(fig, "fig7_confusion_ag_news")


def fig_epochs(results: dict) -> None:
    """How many epochs the task actually needs."""
    datasets = [d for d in EPOCH_DATASETS
                if all(get(results, m, d, "A_baseline__ep4") for m in ("distilbert", "bert"))]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4, 2.0), squeeze=False,
                             layout="constrained")
    for ax, d in zip(axes.flat, datasets):
        for m in ("distilbert", "bert"):
            rows = get(results, m, d, "A_baseline__ep4")["training"]["per_epoch"]
            ep = [r["epoch"] for r in rows]
            val = [r["val_f1_macro"] * 100 for r in rows]
            ax.plot(ep, val, color=MODEL_COLOR[m], marker="o", markersize=3.5,
                    label=MODEL_LABEL[m])
            if "test_f1_macro" in rows[0]:
                ax.plot(ep, [r["test_f1_macro"] * 100 for r in rows], color=MODEL_COLOR[m],
                        ls="--", lw=1.2, alpha=0.8)
            best = max(rows, key=lambda r: r["val_f1_macro"])["epoch"]
            ax.scatter([best], [val[ep.index(best)]], s=70, facecolor="none",
                       edgecolor=MODEL_COLOR[m], linewidth=1.4, zorder=5)
        ax.axvline(2, color=GRID, lw=1.0, zorder=0)
        ax.set_xticks(range(1, EPOCH_STUDY_EPOCHS + 1))
        ax.set_xlabel("Epochs trained")
        ax.set_ylabel("F1 macro (%)")
        ax.set_title(DATASET_LABEL[d])
        tidy(ax)
    handles = [plt.Line2D([], [], color=MODEL_COLOR[m], marker="o", markersize=3.5,
                          label=MODEL_LABEL[m]) for m in ("distilbert", "bert")]
    handles += [plt.Line2D([], [], color=INK_SOFT, ls="-", label="validation"),
                plt.Line2D([], [], color=INK_SOFT, ls="--", label="test")]
    fig.legend(handles=handles, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.14), fontsize=7)
    save(fig, "fig9_epoch_study")


def table_epochs(results: dict) -> None:
    datasets = [d for d in EPOCH_DATASETS
                if all(get(results, m, d, "A_baseline__ep4") for m in ("distilbert", "bert"))]
    if not datasets:
        return
    rows = []
    for d in datasets:
        for m in ("distilbert", "bert"):
            r = get(results, m, d, "A_baseline__ep4")
            per = r["training"]["per_epoch"]
            best = r["training"]["best_epoch_by_val_f1"]
            cells = " & ".join(mark_best([x["val_f1_macro"] * 100 for x in per],
                                         "{:.2f}", higher_is_better=True))
            rows.append(f"{DATASET_LABEL[d]} & {MODEL_LABEL[m]} & {cells} & {best} \\\\")
        rows.append("\\addlinespace")
    cols = "ll" + "r" * EPOCH_STUDY_EPOCHS + "r"
    header = " & ".join(str(i) for i in range(1, EPOCH_STUDY_EPOCHS + 1))
    body = (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            f"Dataset & Model & \\multicolumn{{{EPOCH_STUDY_EPOCHS}}}{{c}}{{Validation F1 macro after epoch}} "
            f"& Best \\\\\n\\cmidrule(lr){{3-{2 + EPOCH_STUDY_EPOCHS}}}\n"
            f" &  & {header} &  \\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("table_epochs", body,
                caption="Epoch study. Training runs to four epochs, the top of the range "
                        "recommended by Devlin et al., and validation F1 macro is scored at every "
                        "epoch boundary. Best is the epoch selected by validation F1.",
                label="tab:epochs")


def fig_representation(web_dir: str | None = None) -> None:
    """How close the two latent spaces are, from the exported demo data."""
    path = os.path.join(web_dir or WEB_DIR, "words.json")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        data = json.load(fh)
    meta = data["meta"]
    overlap = np.array([w["overlap"] for w in data["words"]])
    k = meta["top_k"]

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.85), layout="constrained")

    ax = axes[0]
    counts = np.bincount(overlap, minlength=k + 1)
    ax.bar(np.arange(k + 1), counts, color=C_DISTIL, width=0.72)
    ax.axvline(overlap.mean(), color=INK_SOFT, ls="--", lw=1.1)
    ax.annotate(f"mean {overlap.mean():.1f}", (overlap.mean(), counts.max()),
                xytext=(-4, -6), textcoords="offset points", ha="right",
                fontsize=6.5, color=INK_SOFT)
    ax.set_xlabel(f"Shared neighbours out of {k}")
    ax.set_ylabel("Words")
    ax.set_title("Agreement on the closest words", fontsize=8)
    tidy(ax)

    ax = axes[1]
    prof = meta.get("layer_profile")
    if prof:
        x = [r["distil_layer"] for r in prof]
        y = [r["cka"] for r in prof]
        ax.plot(x, y, color=C_DISTIL, marker="o", markersize=3.5)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6, color=INK_SOFT)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x, ["emb"] + [f"{d}/{2 * d}" for d in x[1:]], fontsize=6.5)
        ax.set_xlabel("DistilBERT block / BERT block")
        ax.set_ylabel("Linear CKA")
        ax.set_title("Alignment falls with depth", fontsize=8)
    tidy(ax)
    save(fig, "fig8_representation_similarity")


# --------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------

def mark_best(values: list[float], fmt: str, higher_is_better: bool = True,
              bold: str = "tex") -> list[str]:
    """Format a row or column and emphasise the winning value.

    Ties are all marked, since claiming a single winner on equal numbers would be
    misleading. Returns LaTeX or Markdown depending on bold.
    """
    texts = [fmt.format(v) for v in values]
    # Compare what the reader sees, so two values that print the same are both
    # marked rather than one being crowned on invisible decimals.
    shown = []
    for v, t in zip(values, texts):
        try:
            shown.append(float(t.replace(",", "")))
        except ValueError:
            shown.append(float("nan"))
    clean = [v for v in shown if v == v]
    if not clean:
        return texts
    target = max(clean) if higher_is_better else min(clean)
    out = []
    for v, t in zip(shown, texts):
        if v == v and v == target:
            t = f"\\textbf{{{t}}}" if bold == "tex" else f"**{t}**"
        out.append(t)
    return out


def write_table(name: str, body: str, caption: str = "", label: str = "",
                size: str = "\\footnotesize") -> None:
    """Write a LaTeX fragment. With a caption it becomes a complete table float."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    if caption:
        body = (f"\\begin{{table}}[t]\n\\centering\n{size}\n"
                f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
                f"{body}\\end{{table}}\n")
    with open(os.path.join(TABLES_DIR, f"{name}.tex"), "w") as fh:
        fh.write(body)
    print(f"  wrote {name}.tex", flush=True)


def table_datasets(results: dict) -> None:
    """Sizes actually used, read back from the runs."""
    datasets = available_main(results)
    if not datasets:
        return
    rows = []
    for d in datasets:
        r = get(results, "distilbert", d)
        spec = DATASETS[d]
        sizes = r["dataset_sizes"]
        source = (spec.path if spec.name is None else spec.path + "/" + spec.name).replace("_", r"\_")
        rows.append(f"{DATASET_LABEL[d]} & \\texttt{{{source}}} & "
                    f"{r['num_labels']} & {sizes['train']:,} & {sizes['val']:,} & {sizes['test']:,} & "
                    f"{r['max_length']} \\\\".replace(",", "{,}"))
    body = ("\\begin{tabular}{llrrrrr}\n\\toprule\n"
            "Dataset & Source & Classes & Train & Validation & Test & Max length \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("table_datasets", body,
                caption="Datasets after subsampling. Validation is a five percent slice of train, "
                        "held out for the loss curves only.", label="tab:data")


def table_main(results: dict) -> None:
    datasets = available_main(results)
    if not datasets:
        return
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    rows = []
    for d in datasets:
        cells = {}
        for k in keys:
            vals = [get(results, m, d)["metrics"][k] * 100 for m in ("distilbert", "bert")]
            cells[k] = mark_best(vals, "{:.2f}", higher_is_better=True)
        for i, m in enumerate(("distilbert", "bert")):
            line = " & ".join(cells[k][i] for k in keys)
            rows.append(f"{DATASET_LABEL[d]} & {MODEL_LABEL[m]} & {line} \\\\")
        rows.append("\\addlinespace")
    body = ("\\begin{tabular}{llrrrr}\n\\toprule\n"
            "Dataset & Model & Accuracy & Precision & Recall & F1 \\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("table_main_results", body,
                caption="Performance of both backbones with the identical baseline classifier. "
                        "Precision, recall and F1 are macro averaged. The better value of each "
                        "pair is in bold.", label="tab:main")


def table_efficiency(results: dict) -> None:
    datasets = available_main(results)
    if not datasets:
        return
    rows = []
    for m in ("distilbert", "bert"):
        r0 = get(results, m, datasets[0])
        lat = np.mean([get(results, m, d)["latency"]["latency_ms_p50"] for d in datasets])
        cpu = np.mean([get(results, m, d)["latency"].get("cpu_latency_ms_p50", np.nan)
                       for d in datasets])
        thr = np.mean([get(results, m, d)["latency"]["throughput_samples_per_s"] for d in datasets])
        mem = np.mean([get(results, m, d)["training"]["train_peak_gpu_mb"] for d in datasets])
        inf = np.mean([get(results, m, d)["memory"]["inference_peak_mb"] for d in datasets])
        tim = np.sum([get(results, m, d)["training"]["train_seconds"] for d in datasets]) / 60
        rows.append(f"{MODEL_LABEL[m]} & {r0['params']['total_params']/1e6:.1f} & "
                    f"{lat:.2f} & {cpu:.1f} & {thr:.0f} & {mem/1024:.2f} & {inf:.0f} & {tim:.0f} \\\\")
    body = ("\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "Model & Params (M) & GPU ms & CPU ms & Samples/s & Train GiB & Infer MiB & Total min \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("table_efficiency", body,
                caption="Efficiency averaged over the four datasets. Latency is the median of 200 "
                        "single example passes; total minutes is the sum over the four runs.",
                label="tab:eff")


def table_ablation(results: dict) -> None:
    datasets = [d for d in ABLATION_DATASETS
                if all(get(results, "distilbert", d, c.key) for c in ABLATIONS)]
    if not datasets:
        return
    header = " & ".join(f"{DATASET_LABEL[d]} Acc & {DATASET_LABEL[d]} F1" for d in datasets)
    # one column at a time, so the best variant on each dataset metric stands out
    columns = []
    for d in datasets:
        for k in ("accuracy", "f1_macro"):
            vals = [get(results, "distilbert", d, c.key)["metrics"][k] * 100 for c in ABLATIONS]
            columns.append(mark_best(vals, "{:.2f}", higher_is_better=True))
    rows = []
    for i, cfg in enumerate(ABLATIONS):
        r0 = get(results, "distilbert", datasets[0], cfg.key)
        cells = [col[i] for col in columns]
        rows.append(f"{ABLATION_SHORT[cfg.key]} & {r0['params']['trainable_params']/1e6:.1f} & "
                    + " & ".join(cells) + " \\\\")
    cols = "lr" + "rr" * len(datasets)
    body = (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            f"Variant & Train (M) & {header} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("table_ablation", body,
                caption="Classifier ablation on DistilBERT. Train (M) is the number of trainable "
                        "parameters in millions. The best variant in each column is in bold.",
                label="tab:ablation")


def table_best_on_bert(results: dict) -> None:
    """The selected variant on both backbones, transposed so every metric fits.

    The assignment asks for all metrics here, so this table carries the four
    performance scores and every efficiency measurement we record, rather than a
    selection that happens to fit across the page.
    """
    key = best_head(results)
    if key is None:
        return
    datasets = [d for d in ABLATION_DATASETS
                if get(results, "distilbert", d, key) and get(results, "bert", d, key)]
    if not datasets:
        return

    cols = [(d, m) for d in datasets for m in ("distilbert", "bert")]
    # last field: is a larger number better for this metric
    rows = [
        ("Accuracy (\\%)", lambda r: r["metrics"]["accuracy"] * 100, "{:.2f}", True),
        ("Precision, macro (\\%)", lambda r: r["metrics"]["precision_macro"] * 100, "{:.2f}", True),
        ("Recall, macro (\\%)", lambda r: r["metrics"]["recall_macro"] * 100, "{:.2f}", True),
        ("F1, macro (\\%)", lambda r: r["metrics"]["f1_macro"] * 100, "{:.2f}", True),
        ("Test cross entropy", lambda r: r["test_loss"], "{:.3f}", False),
        ("Total parameters (M)", lambda r: r["params"]["total_params"] / 1e6, "{:.1f}", False),
        ("Trainable parameters (M)", lambda r: r["params"]["trainable_params"] / 1e6, "{:.1f}", False),
        ("GPU latency, batch 1 (ms)", lambda r: r["latency"]["latency_ms_p50"], "{:.2f}", False),
        ("CPU latency, batch 1 (ms)",
         lambda r: r["latency"].get("cpu_latency_ms_p50", float("nan")), "{:.1f}", False),
        ("Throughput, batch 32 (samples/s)",
         lambda r: r["latency"]["throughput_samples_per_s"], "{:.0f}", True),
        ("Peak GPU memory, training (MiB)",
         lambda r: r["training"]["train_peak_gpu_mb"], "{:.0f}", False),
        ("Peak GPU memory, inference (MiB)",
         lambda r: r["memory"]["inference_peak_mb"], "{:.0f}", False),
        ("Training wall clock (min)", lambda r: r["training"]["train_seconds"] / 60, "{:.1f}", False),
    ]

    header = " & ".join(MODEL_LABEL[m] for _, m in cols)
    group = " & ".join(f"\\multicolumn{{2}}{{c}}{{{DATASET_LABEL[d]}}}" for d in datasets)
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(datasets)))
    body_rows = []
    for name, fn, fmt, higher in rows:
        cells = []
        # each dataset is its own contest between the two backbones
        for i in range(0, len(cols), 2):
            pair = [fn(get(results, m, d, key)) for d, m in cols[i:i + 2]]
            cells += mark_best(pair, fmt, higher)
        body_rows.append(f"{name} & {' & '.join(cells)} \\\\")

    body = (f"\\begin{{tabular}}{{l{'rr' * len(datasets)}}}\n\\toprule\n"
            f" & {group} \\\\\n{cmid}\n & {header} \\\\\n\\midrule\n"
            + "\n".join(body_rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("table_best_on_bert", body,
                caption=f"The variant selected by the ablation ({ABLATION_SHORT.get(key, key)}) on each "
                        "backbone, with every metric this study records, measured on the same GPU. "
                        "Within each dataset the better of the two values is in bold, taking lower "
                        "as better for loss, size, latency, memory and time.",
                label="tab:best", size="\\scriptsize")
    write_table("best_variant_name", ABLATION_SHORT.get(key, key))


def macros_used_by_report(path: str = "report/main.tex") -> set[str]:
    """Every \\name{} the report body references, so none can be left undefined."""
    if not os.path.exists(path):
        return set()
    text = pathlib.Path(path).read_text()
    body = text.split("\\begin{document}", 1)[-1]
    return set(re.findall(r"\\([a-zA-Z]+)\{\}", body))


def write_numbers(results: dict) -> None:
    """Emit LaTeX macros so the report never hardcodes a measured number."""
    datasets = available_main(results)
    if not datasets:
        return
    macros = {}

    def m(name, value, fmt="{:.2f}"):
        # LaTeX control sequences accept letters only, so digits are stripped.
        key = "".join(ch for ch in name if ch.isalpha())
        macros[key] = fmt.format(value) if isinstance(value, float) else str(value)

    pd_ = get(results, "distilbert", datasets[0])["params"]["total_params"] / 1e6
    pb_ = get(results, "bert", datasets[0])["params"]["total_params"] / 1e6
    m("distilParams", pd_, "{:.1f}")
    m("bertParams", pb_, "{:.1f}")
    m("paramReduction", 100 * (1 - pd_ / pb_), "{:.0f}")

    def avg(model, fn):
        return float(np.mean([fn(get(results, model, d)) for d in datasets]))

    lat_d = avg("distilbert", lambda r: r["latency"]["latency_ms_p50"])
    lat_b = avg("bert", lambda r: r["latency"]["latency_ms_p50"])
    m("gpuLatencyDistil", lat_d, "{:.2f}")
    m("gpuLatencyBert", lat_b, "{:.2f}")
    m("gpuSpeedup", lat_b / lat_d, "{:.2f}")

    cpu_d = avg("distilbert", lambda r: r["latency"].get("cpu_latency_ms_p50", float("nan")))
    cpu_b = avg("bert", lambda r: r["latency"].get("cpu_latency_ms_p50", float("nan")))
    if np.isfinite(cpu_d) and np.isfinite(cpu_b):
        m("cpuLatencyDistil", cpu_d, "{:.1f}")
        m("cpuLatencyBert", cpu_b, "{:.1f}")
        m("cpuSpeedup", cpu_b / cpu_d, "{:.2f}")

    thr_d = avg("distilbert", lambda r: r["latency"]["throughput_samples_per_s"])
    thr_b = avg("bert", lambda r: r["latency"]["throughput_samples_per_s"])
    m("throughputDistil", thr_d, "{:.0f}")
    m("throughputBert", thr_b, "{:.0f}")
    m("throughputSpeedup", thr_d / thr_b, "{:.2f}")

    mem_d = avg("distilbert", lambda r: r["training"]["train_peak_gpu_mb"])
    mem_b = avg("bert", lambda r: r["training"]["train_peak_gpu_mb"])
    m("trainMemDistil", mem_d / 1024, "{:.2f}")
    m("trainMemBert", mem_b / 1024, "{:.2f}")
    m("memReduction", 100 * (1 - mem_d / mem_b), "{:.0f}")
    m("inferMemDistil", avg("distilbert", lambda r: r["memory"]["inference_peak_mb"]), "{:.0f}")
    m("inferMemBert", avg("bert", lambda r: r["memory"]["inference_peak_mb"]), "{:.0f}")

    acc_d = avg("distilbert", lambda r: r["metrics"]["accuracy"] * 100)
    acc_b = avg("bert", lambda r: r["metrics"]["accuracy"] * 100)
    f1_d = avg("distilbert", lambda r: r["metrics"]["f1_macro"] * 100)
    f1_b = avg("bert", lambda r: r["metrics"]["f1_macro"] * 100)
    m("meanAccDistil", acc_d)
    m("meanAccBert", acc_b)
    m("meanAccGap", acc_b - acc_d)
    m("meanFoneDistil", f1_d)
    m("meanFoneBert", f1_b)
    m("meanFoneGap", f1_b - f1_d)
    m("accRetention", 100 * acc_d / acc_b, "{:.1f}")

    gaps = {d: (get(results, "bert", d)["metrics"]["accuracy"]
                - get(results, "distilbert", d)["metrics"]["accuracy"]) * 100 for d in datasets}
    lo_ds = min(gaps, key=gaps.get)
    hi_ds = max(gaps, key=gaps.get)
    m("maxAccGap", gaps[hi_ds])
    m("minAccGap", gaps[lo_ds])
    m("minGapDataset", DATASET_LABEL[lo_ds])
    m("maxGapDataset", DATASET_LABEL[hi_ds])
    m("nDatasets", len(datasets))
    m("nRuns", len(results))
    # How many datasets DistilBERT actually wins or ties, so the prose cannot overclaim.
    m("bertWins", sum(1 for g in gaps.values() if g > 0.1))
    m("distilTiesOrWins", sum(1 for g in gaps.values() if g <= 0.1))

    tt_d = sum(get(results, "distilbert", d)["training"]["train_seconds"] for d in datasets) / 60
    tt_b = sum(get(results, "bert", d)["training"]["train_seconds"] for d in datasets) / 60
    m("trainMinutesDistil", tt_d, "{:.0f}")
    m("trainMinutesBert", tt_b, "{:.0f}")

    key = best_head(results)
    if key:
        m("bestVariant", ABLATION_SHORT.get(key, key))
        abl_ds = [d for d in ABLATION_DATASETS
                  if all(get(results, "distilbert", d, c.key) for c in ABLATIONS)]
        if abl_ds:
            f1s = {c.key: float(np.mean([get(results, "distilbert", d, c.key)["metrics"]["f1_macro"]
                                         * 100 for d in abl_ds])) for c in ABLATIONS}
            m("bestVariantFone", f1s[key])
            m("frozenVariantFone", f1s["B_frozen_all"])
            m("frozenGap", f1s["A_baseline"] - f1s["B_frozen_all"])
            m("halfFrozenGap", f1s["A_baseline"] - f1s["C_frozen_half"])
            shapes = ("D_narrow", "E_wide", "F_deep", "G_linear", "A_baseline")
            m("headSpread", max(f1s[k] for k in shapes) - min(f1s[k] for k in shapes))
            # What freezing the lower half actually buys, which is the practical finding.
            base = get(results, "distilbert", abl_ds[0], "A_baseline")
            half = get(results, "distilbert", abl_ds[0], "C_frozen_half")
            if base and half:
                m("halfFrozenTrainable", half["params"]["trainable_params"] / 1e6, "{:.1f}")
                m("halfFrozenTrainableCut",
                  100 * (1 - half["params"]["trainable_params"] / base["params"]["trainable_params"]),
                  "{:.0f}")
                cut = np.mean([get(results, "distilbert", d, "A_baseline")["training"]["train_seconds"]
                               / get(results, "distilbert", d, "C_frozen_half")["training"]["train_seconds"]
                               for d in abl_ds])
                m("halfFrozenSpeedup", float(cut), "{:.1f}")
        for d in abl_ds:
            rb = get(results, "bert", d, key)
            rd = get(results, "distilbert", d, key)
            if rb and rd:
                tag = "".join(w.capitalize() for w in DATASET_LABEL[d].split())
                m("best" + tag + "Distil", rd["metrics"]["f1_macro"] * 100)
                m("best" + tag + "Bert", rb["metrics"]["f1_macro"] * 100)

    ep_rows = [get(results, m, d, "A_baseline__ep4")
               for d in EPOCH_DATASETS for m in ("distilbert", "bert")]
    ep_rows = [r for r in ep_rows if r]
    if ep_rows:
        bests = [r["training"]["best_epoch_by_val_f1"] for r in ep_rows]
        m("epochStudyRuns", len(ep_rows))
        m("epochBestMin", min(bests))
        m("epochBestMax", max(bests))
        gains = []
        for r in ep_rows:
            per = {x["epoch"]: x["val_f1_macro"] * 100 for x in r["training"]["per_epoch"]}
            if 2 in per:
                gains.append(max(per.values()) - per[2])
        if gains:
            m("epochGainOverTwo", max(gains))

    web = os.path.join(WEB_DIR, "words.json")
    if os.path.exists(web):
        with open(web) as fh:
            w = json.load(fh)
        m("wordOverlap", w["meta"]["mean_overlap"], "{:.1f}")
        m("wordOverlapPct", w["meta"]["overlap_pct"], "{:.0f}")
        m("wordCka", w["meta"]["cka"], "{:.3f}")
        m("nWords", w["meta"]["n_words"])
        if "purity" in w["meta"]:
            m("purityBert", w["meta"]["purity"]["bert"] * 100, "{:.0f}")
            m("purityDistil", w["meta"]["purity"]["distilbert"] * 100, "{:.0f}")
        if w["meta"].get("layer_profile"):
            prof = w["meta"]["layer_profile"]
            m("ckaEmbeddings", prof[0]["cka"], "{:.3f}")
            m("ckaTop", prof[-1]["cka"], "{:.3f}")
            mid = prof[len(prof) // 2]
            m("ckaMid", mid["cka"], "{:.3f}")
            m("ckaMidDistilLayer", mid["distil_layer"])
            m("ckaMidBertLayer", mid["bert_layer"])
    sen = os.path.join(WEB_DIR, "sentences.json")
    if os.path.exists(sen):
        with open(sen) as fh:
            s_ = json.load(fh)
        # one entry per dataset now; the report quotes the AG News number
        per_ds = s_.get("datasets", {})
        if "ag_news" in per_ds:
            m("sentCka", per_ds["ag_news"]["meta"]["cka"], "{:.3f}")
            best_ds = max(per_ds, key=lambda k: per_ds[k]["meta"]["cka"])
            m("sentCkaMax", per_ds[best_ds]["meta"]["cka"], "{:.3f}")
            m("sentCkaMin", min(v["meta"]["cka"] for v in per_ds.values()), "{:.3f}")
        elif "meta" in s_:
            m("sentCka", s_["meta"]["cka"], "{:.3f}")

    # providecommand plus renewcommand so this file can be read after the report
    # has already declared fallbacks for the same names.
    # Any quantity the report uses that has no result file yet gets a visible
    # placeholder, so the report still compiles and the gap is obvious in the PDF.
    for name in macros_used_by_report():
        macros.setdefault(name, "\\textbf{[pending]}")

    body = "\n".join(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}"
                     for k, v in sorted(macros.items())) + "\n"
    write_table("numbers", body)


def summary_markdown(results: dict) -> str:
    """The results summary that gets spliced into README.md."""
    datasets = available_main(results)
    if not datasets:
        return "Run the experiments first, then run src/figures.py.\n"

    def fig(name, caption):
        return ["", f"![{caption}](artifacts/figures/{name}.png)", "",
                f"*{caption}*", ""]

    out = ["## Results summary", "",
           "Generated by src/figures.py from artifacts/results. Do not edit by hand.", "",
           "### Performance, identical classifier on both backbones", "",
           "| Dataset | Model | Accuracy | Precision | Recall | F1 macro |",
           "|---|---|---|---|---|---|"]
    perf_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    for d in datasets:
        marked = {k: mark_best([get(results, m_, d)["metrics"][k] * 100
                                for m_ in ("distilbert", "bert")], "{:.2f}", True, bold="md")
                  for k in perf_keys}
        for i, m_ in enumerate(("distilbert", "bert")):
            cells = " | ".join(marked[k][i] for k in perf_keys)
            out.append(f"| {DATASET_LABEL[d]} | {MODEL_LABEL[m_]} | {cells} |")

    def avg(model, fn):
        return float(np.mean([fn(get(results, model, d)) for d in datasets]))

    # The last field says whether a larger number is better, so the ratio column
    # always reads as "how many times better DistilBERT is".
    rows = [("Parameters (M)", lambda r: r["params"]["total_params"] / 1e6, "{:.1f}", False),
            ("GPU latency, batch 1 (ms)", lambda r: r["latency"]["latency_ms_p50"], "{:.2f}", False),
            ("CPU latency, batch 1 (ms)", lambda r: r["latency"].get("cpu_latency_ms_p50", float("nan")), "{:.1f}", False),
            ("Throughput, batch 32 (samples/s)", lambda r: r["latency"]["throughput_samples_per_s"], "{:.0f}", True),
            ("Peak GPU memory, training (MiB)", lambda r: r["training"]["train_peak_gpu_mb"], "{:.0f}", False),
            ("Peak GPU memory, inference (MiB)", lambda r: r["memory"]["inference_peak_mb"], "{:.0f}", False),
            ("Training wall clock, all datasets (min)",
             lambda r: r["training"]["train_seconds"] / 60 * len(datasets), "{:.0f}", False)]
    out += fig("fig2_performance_metrics",
               "Accuracy, precision, recall and F1 for both backbones on every dataset")
    out += fig("fig1_params_vs_accuracy_bubble",
               "Parameters against accuracy. Bubble area is single example GPU latency")
    out += fig("fig3_loss_curves",
               "Iterations against training loss (solid) and validation loss (dashed)")
    out += ["", "### Efficiency, averaged over the datasets", "",
            "| Measurement | BERT | DistilBERT | DistilBERT advantage |", "|---|---|---|---|"]
    for name, fn, fmt, higher_better in rows:
        b, dd = avg("bert", fn), avg("distilbert", fn)
        ratio = (dd / b) if higher_better else (b / dd if dd else float("nan"))
        cb, cd = mark_best([b, dd], fmt, higher_better, bold="md")
        out.append(f"| {name} | {cb} | {cd} | {ratio:.2f}x |")

    out += fig("fig4_efficiency",
               "Latency, throughput, peak training memory and wall clock time")

    key = best_head(results)
    if key:
        out += ["", f"### Ablation on the classifier, best variant: {ABLATION_SHORT.get(key, key)}", "",
                "| Variant | Trainable (M) | " +
                " | ".join(f"{DATASET_LABEL[d]} F1" for d in ABLATION_DATASETS) + " |",
                "|---" * (2 + len(ABLATION_DATASETS)) + "|"]
        abl_cols = [mark_best([get(results, "distilbert", d, c.key)["metrics"]["f1_macro"] * 100
                               for c in ABLATIONS], "{:.2f}", True, bold="md")
                    for d in ABLATION_DATASETS]
        for i, cfg in enumerate(ABLATIONS):
            base = get(results, "distilbert", ABLATION_DATASETS[0], cfg.key)
            tp = base["params"]["trainable_params"] / 1e6
            cells = " | ".join(col[i] for col in abl_cols)
            out.append(f"| {ABLATION_SHORT[cfg.key]} | {tp:.1f} | {cells} |")

        out += fig("fig5_ablation",
                   "Classifier ablation on DistilBERT. Only the freezing axis separates the variants")
        # Transposed so every metric fits, matching the table in the report.
        cols = [(d, m_) for d in ABLATION_DATASETS for m_ in ("distilbert", "bert")
                if get(results, m_, d, key)]
        metric_rows = [
            ("Accuracy (%)", lambda r: r["metrics"]["accuracy"] * 100, "{:.2f}", True),
            ("Precision, macro (%)", lambda r: r["metrics"]["precision_macro"] * 100, "{:.2f}", True),
            ("Recall, macro (%)", lambda r: r["metrics"]["recall_macro"] * 100, "{:.2f}", True),
            ("F1, macro (%)", lambda r: r["metrics"]["f1_macro"] * 100, "{:.2f}", True),
            ("Test cross entropy", lambda r: r["test_loss"], "{:.3f}", False),
            ("Total parameters (M)", lambda r: r["params"]["total_params"] / 1e6, "{:.1f}", False),
            ("Trainable parameters (M)", lambda r: r["params"]["trainable_params"] / 1e6, "{:.1f}", False),
            ("GPU latency, batch 1 (ms)", lambda r: r["latency"]["latency_ms_p50"], "{:.2f}", False),
            ("CPU latency, batch 1 (ms)",
             lambda r: r["latency"].get("cpu_latency_ms_p50", float("nan")), "{:.1f}", False),
            ("Throughput, batch 32 (samples/s)",
             lambda r: r["latency"]["throughput_samples_per_s"], "{:.0f}", True),
            ("Peak GPU memory, training (MiB)",
             lambda r: r["training"]["train_peak_gpu_mb"], "{:.0f}", False),
            ("Peak GPU memory, inference (MiB)",
             lambda r: r["memory"]["inference_peak_mb"], "{:.0f}", False),
            ("Training wall clock (min)", lambda r: r["training"]["train_seconds"] / 60, "{:.1f}", False),
        ]
        head = " | ".join(f"{DATASET_LABEL[d]} {MODEL_LABEL[m_]}" for d, m_ in cols)
        out += ["", "### The best variant with each backbone, every metric", "",
                f"| Metric | {head} |", "|---" * (1 + len(cols)) + "|"]
        for name, fn, fmt, higher in metric_rows:
            cells = []
            for i in range(0, len(cols), 2):
                pair = [fn(get(results, m_, d, key)) for d, m_ in cols[i:i + 2]]
                cells += mark_best(pair, fmt, higher, bold="md")
            out.append(f"| {name} | {' | '.join(cells)} |")

        out += fig("fig6_best_variant_on_bert",
                   "The selected classifier variant with each backbone")
        out += fig("fig7_confusion_ag_news",
                   "Row normalised confusion matrices on AG News. Both models fail in the same place")

    ep = [(d, m_, get(results, m_, d, "A_baseline__ep4"))
          for d in EPOCH_DATASETS for m_ in ("distilbert", "bert")]
    ep = [(d, m_, r) for d, m_, r in ep if r]
    if ep:
        n_ep = len(ep[0][2]["training"]["per_epoch"])
        out += ["", "### Epoch study, validation F1 macro after each epoch", "",
                "| Dataset | Model | " + " | ".join(str(i) for i in range(1, n_ep + 1)) +
                " | Best |", "|---" * (3 + n_ep) + "|"]
        for d, m_, r in ep:
            cells = " | ".join(mark_best([x["val_f1_macro"] * 100 for x in r["training"]["per_epoch"]],
                                         "{:.2f}", True, bold="md"))
            out.append(f"| {DATASET_LABEL[d]} | {MODEL_LABEL[m_]} | {cells} | "
                       f"{r['training']['best_epoch_by_val_f1']} |")

        out += fig("fig9_epoch_study",
                   "Validation F1 macro at every epoch boundary, dashed lines are the test split")

    web = os.path.join(WEB_DIR, "words.json")
    if os.path.exists(web):
        with open(web) as fh:
            w = json.load(fh)["meta"]
        out += ["", "### Representation alignment between the two pretrained encoders", "",
                f"Neighbour agreement over {w['n_words']} words: **{w['overlap_pct']} percent** of the "
                f"ten nearest neighbours are shared. Linear CKA of the layer averaged space: "
                f"**{w['cka']}**."]
        if "purity" in w:
            out.append(f"Topic purity of the ten nearest neighbours: BERT {w['purity']['bert']*100:.1f} "
                       f"percent, DistilBERT {w['purity']['distilbert']*100:.1f} percent.")
        if w.get("layer_profile"):
            out += ["", "| DistilBERT block | BERT block | Linear CKA | Shared neighbours out of 10 |",
                    "|---|---|---|---|"]
            for row in w["layer_profile"]:
                depth = ("embedding output" if row["distil_layer"] == 0
                         else f"block {row['distil_layer']}")
                teacher = ("embedding output" if row["bert_layer"] == 0
                           else f"block {row['bert_layer']}")
                out.append(f"| {depth} | {teacher} | {row['cka']:.3f} | {row['overlap']:.2f} |")
            out += fig("fig8_representation_similarity",
                       "Neighbour agreement between the two encoders, and how alignment falls with depth")
    return "\n".join(out) + "\n"


def update_readme(results: dict, path: str = "README.md") -> None:
    """Write the generated summary, and splice it into a README that asks for it.

    The canonical copy always lands in artifacts/tables/results_summary.md. A README
    is only rewritten if it carries the "## Results summary" heading, so a hand
    edited one is never clobbered; if it does not, we say so loudly instead of
    silently doing nothing.
    """
    block = summary_markdown(results)
    os.makedirs(TABLES_DIR, exist_ok=True)
    canonical = os.path.join(TABLES_DIR, "results_summary.md")
    pathlib.Path(canonical).write_text(block)
    print(f"  wrote {canonical}", flush=True)

    marker = "## Results summary"
    for target in (path, "README_backup.md"):
        if not os.path.exists(target):
            continue
        text = pathlib.Path(target).read_text()
        if marker not in text:
            if target == path:
                print(f"  NOTE: {target} has no '{marker}' heading, so it was left untouched. "
                      f"The generated tables are in {canonical}.", flush=True)
            continue
        start = text.index(marker)
        rest = text[start + len(marker):]
        nxt = rest.find("\n## ")
        end = len(text) if nxt == -1 else start + len(marker) + nxt + 1
        pathlib.Path(target).write_text(text[:start] + block + "\n" + text[end:])
        print(f"  {target} results summary updated", flush=True)


# Tables the report reads. Any of these without data becomes an explicit
# placeholder, so a stale or fabricated table can never survive in the repository.
EXPECTED_TABLES = {
    "table_datasets": "tab:data",
    "table_main_results": "tab:main",
    "table_efficiency": "tab:eff",
    "table_ablation": "tab:ablation",
    "table_best_on_bert": "tab:best",
    "table_epochs": "tab:epochs",
}


def fill_pending_tables() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    for name, label in EXPECTED_TABLES.items():
        path = os.path.join(TABLES_DIR, f"{name}.tex")
        if os.path.exists(path):
            continue
        with open(path, "w") as fh:
            fh.write("\\begin{table}[t]\\centering\\footnotesize\n"
                     "\\caption{Pending. The runs behind this table have not finished; "
                     "src/figures.py writes it as soon as they do.}\n"
                     f"\\label{{{label}}}\n"
                     "\\begin{tabular}{l}\\toprule pending \\\\ \\bottomrule\\end{tabular}\n"
                     "\\end{table}\n")
        print(f"  {name}.tex: no data yet, wrote a pending placeholder", flush=True)


def main() -> None:
    setup_style()
    results = load_results()
    print(f"loaded {len(results)} runs", flush=True)
    fig_params_accuracy_bubble(results)
    fig_performance(results)
    fig_loss_curves(results)
    fig_efficiency(results)
    fig_ablation(results)
    fig_best_on_bert(results)
    fig_confusion(results)
    fig_epochs(results)
    fig_representation()
    table_datasets(results)
    table_main(results)
    table_efficiency(results)
    table_ablation(results)
    table_best_on_bert(results)
    table_epochs(results)
    write_numbers(results)
    fill_pending_tables()
    update_readme(results)
    print("figures and tables complete")


if __name__ == "__main__":
    main()
