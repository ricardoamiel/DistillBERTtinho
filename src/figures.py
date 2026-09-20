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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (ABLATIONS, ABLATION_BY_KEY, ABLATION_DATASETS, DATASETS,
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
DATASET_LABEL = {
    "ag_news": "AG News",
    "sst2": "SST 2",
    "yelp_polarity": "Yelp Polarity",
    "yelp_full": "Yelp Full",
}
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


def tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)


def save(fig, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"), format=ext)
    plt.close(fig)
    print(f"  wrote {name}.pdf and {name}.svg", flush=True)


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
    """Required bubble chart: parameters against accuracy, size is latency."""
    datasets = available_main(results)
    if not datasets:
        return
    fig, ax = plt.subplots(figsize=(5.4, 2.05))

    lat = [get(results, m, d)["latency"]["latency_ms_p50"]
           for d in datasets for m in ("distilbert", "bert")]
    lo, hi = min(lat), max(lat)

    def area(ms):
        return 120 + 900 * (ms - lo) / max(hi - lo, 1e-9)

    for d in datasets:
        pts = []
        for m in ("distilbert", "bert"):
            r = get(results, m, d)
            pts.append((r["params"]["total_params"] / 1e6,
                        r["metrics"]["accuracy"] * 100,
                        r["latency"]["latency_ms_p50"]))
        ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                color=GRID, lw=1.0, zorder=1)
        for (x, y, ms), m in zip(pts, ("distilbert", "bert")):
            ax.scatter(x, y, s=area(ms), color=MODEL_COLOR[m], alpha=0.75,
                       edgecolor="white", linewidth=1.2, zorder=3)
        mid_y = (pts[0][1] + pts[1][1]) / 2
        ax.annotate(DATASET_LABEL[d], (pts[1][0], mid_y),
                    xytext=(13, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
        ax.annotate(f"{pts[0][1]:.1f}", (pts[0][0], pts[0][1]), xytext=(-24, -2.5),
                    textcoords="offset points", fontsize=6.5, color=INK_SOFT)
        ax.annotate(f"{pts[1][1]:.1f}", (pts[1][0], pts[1][1]), xytext=(8, -9),
                    textcoords="offset points", fontsize=6.5, color=INK_SOFT)

    ax.set_xlabel("Total parameters (millions)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(55, 150)
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=7,
                          color=MODEL_COLOR[m], label=MODEL_LABEL[m])
               for m in ("distilbert", "bert")]
    handles.append(plt.Line2D([], [], marker="o", ls="", markersize=4,
                              color=INK_SOFT, alpha=0.5,
                              label=f"bubble area: GPU latency {lo:.1f} to {hi:.1f} ms"))
    ax.legend(handles=handles, loc="lower right")
    tidy(ax)
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
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4, 1.9 * nrow), squeeze=False)

    for ax, d in zip(axes.flat, datasets):
        x = np.arange(len(keys))
        for i, m in enumerate(("distilbert", "bert")):
            r = get(results, m, d)
            vals = [r["metrics"][k] * 100 for k in keys]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36,
                          color=MODEL_COLOR[m], label=MODEL_LABEL[m])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                        ha="center", fontsize=6, color=INK_SOFT)
        lo = min(get(results, m, d)["metrics"][k] * 100
                 for m in ("distilbert", "bert") for k in keys)
        ax.set_ylim(max(0, lo - 8), 103)
        ax.set_xticks(x, names)
        ax.set_title(DATASET_LABEL[d])
        ax.set_ylabel("%")
        tidy(ax)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="lower right", ncols=2)
    fig.tight_layout()
    save(fig, "fig2_performance_metrics")


def fig_loss_curves(results: dict) -> None:
    """Iterations against training and validation loss."""
    datasets = available_main(results)
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4, 1.55), squeeze=False)
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
    fig.legend(handles=handles, ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.13), fontsize=7)
    fig.tight_layout()
    save(fig, "fig3_loss_curves")


def fig_efficiency(results: dict) -> None:
    """Latency, throughput, GPU memory and training time."""
    datasets = available_main(results)
    if not datasets:
        return
    panels = [
        ("GPU latency\nbatch 1 (ms)", lambda r: r["latency"]["latency_ms_p50"]),
        ("Throughput\nbatch 32 (samples/s)", lambda r: r["latency"]["throughput_samples_per_s"]),
        ("Peak GPU memory\ntraining (MiB)", lambda r: r["training"]["train_peak_gpu_mb"]),
        ("Wall clock\ntraining (min)", lambda r: r["training"]["train_seconds"] / 60),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(5.4, 1.85))
    x = np.arange(len(datasets))
    for ax, (title, fn) in zip(axes.flat, panels):
        for i, m in enumerate(("distilbert", "bert")):
            vals = [fn(get(results, m, d)) for d in datasets]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36,
                          color=MODEL_COLOR[m], label=MODEL_LABEL[m])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6, color=INK_SOFT)
        ax.set_xticks(x, [DATASET_LABEL[d] for d in datasets], rotation=38, ha="right", fontsize=6)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.margins(y=0.2)
        tidy(ax)
    handles = [plt.Line2D([], [], marker="s", ls="", color=MODEL_COLOR[m], label=MODEL_LABEL[m])
               for m in ("distilbert", "bert")]
    fig.legend(handles=handles, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.16), fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_efficiency")


def fig_ablation(results: dict) -> None:
    """Classifier ablation on DistilBERT."""
    datasets = [d for d in ABLATION_DATASETS
                if all(get(results, "distilbert", d, c.key) for c in ABLATIONS)]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets) + 1,
                             figsize=(5.4, 1.95),
                             gridspec_kw={"width_ratios": [1.15] * len(datasets) + [1.0]})
    axes = np.atleast_1d(axes)
    order = [c.key for c in ABLATIONS]

    for ax, d in zip(axes, datasets):
        vals = [get(results, "distilbert", d, k)["metrics"]["f1_macro"] * 100 for k in order]
        colors = [C_AXIS[ABLATION_AXIS[k]] for k in order]
        y = np.arange(len(order))
        ax.barh(y, vals, color=colors, height=0.68)
        for yi, v in zip(y, vals):
            ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=6.5, color=INK_SOFT)
        ax.set_yticks(y, [ABLATION_SHORT[k] for k in order])
        ax.invert_yaxis()
        ax.set_xlim(min(vals) - 6, 100)
        ax.set_xlabel("F1 macro (%)")
        ax.set_title(DATASET_LABEL[d])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[-1]
    for k in order:
        r = get(results, "distilbert", datasets[0], k)
        ax.scatter(r["params"]["trainable_params"] / 1e6,
                   np.mean([get(results, "distilbert", d, k)["metrics"]["f1_macro"] * 100
                            for d in datasets]),
                   s=60, color=C_AXIS[ABLATION_AXIS[k]], edgecolor="white", linewidth=1.0)
        ax.annotate(k.split("_")[0], (r["params"]["trainable_params"] / 1e6,
                    np.mean([get(results, "distilbert", d, k)["metrics"]["f1_macro"] * 100
                             for d in datasets])),
                    xytext=(5, -2), textcoords="offset points", fontsize=6.5, color=INK_SOFT)
    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_ylabel("Mean F1 macro (%)")
    ax.set_title("Cost against quality")
    tidy(ax)

    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=n.capitalize())
               for n, c in C_AXIS.items()]
    fig.legend(handles=handles, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
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
    fig, axes = plt.subplots(1, len(datasets) + 1, figsize=(5.4, 2.05))
    axes = np.atleast_1d(axes)

    for ax, d in zip(axes, datasets):
        x = np.arange(len(metrics))
        for i, m in enumerate(("distilbert", "bert")):
            r = get(results, m, d, key)
            vals = [r["metrics"][k] * 100 for k, _ in metrics]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36,
                          color=MODEL_COLOR[m], label=MODEL_LABEL[m])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                        ha="center", fontsize=6, color=INK_SOFT)
        lo = min(get(results, m, d, key)["metrics"][k] * 100
                 for m in ("distilbert", "bert") for k, _ in metrics)
        ax.set_ylim(max(0, lo - 6), 103)
        ax.set_xticks(x, [n for _, n in metrics], rotation=20, ha="right")
        ax.set_title(DATASET_LABEL[d])
        tidy(ax)

    ax = axes[-1]
    d = datasets[0]
    bars_data = [
        ("Latency (ms)", lambda r: r["latency"]["latency_ms_p50"]),
        ("GPU mem (GiB)", lambda r: r["training"]["train_peak_gpu_mb"] / 1024),
        ("Train (min)", lambda r: r["training"]["train_seconds"] / 60),
    ]
    x = np.arange(len(bars_data))
    for i, m in enumerate(("distilbert", "bert")):
        r = get(results, m, d, key)
        vals = [fn(r) for _, fn in bars_data]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=MODEL_COLOR[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6, color=INK_SOFT)
    ax.set_xticks(x, [n for n, _ in bars_data], rotation=20, ha="right")
    ax.set_title("Cost on " + DATASET_LABEL[d])
    ax.margins(y=0.2)
    tidy(ax)
    axes.flat[0].legend(loc="lower left", ncols=2)
    fig.suptitle(f"Best classifier variant: {ABLATION_SHORT.get(key, key)}", fontsize=9)
    fig.tight_layout()
    save(fig, "fig6_best_variant_on_bert")


def fig_confusion(results: dict) -> None:
    """Confusion matrices on AG News."""
    if not get(results, "distilbert", "ag_news"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))
    names = DATASETS["ag_news"].label_names
    for ax, m in zip(axes, ("distilbert", "bert")):
        cm = np.array(get(results, m, "ag_news")["metrics"]["confusion_matrix"], dtype=float)
        cm = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 55 else INK)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_title(MODEL_LABEL[m])
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Row percentage")
    save(fig, "fig7_confusion_ag_news")


def fig_epochs(results: dict) -> None:
    """How many epochs the task actually needs."""
    datasets = [d for d in EPOCH_DATASETS
                if all(get(results, m, d, "A_baseline__ep4") for m in ("distilbert", "bert"))]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4, 1.9), squeeze=False)
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
    fig.legend(handles=handles, ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.16), fontsize=7)
    fig.tight_layout()
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
            cells = " & ".join(f"{x['val_f1_macro'] * 100:.2f}" for x in per)
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

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.95))

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
    fig.tight_layout()
    save(fig, "fig8_representation_similarity")


# --------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------

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
    rows = []
    for d in datasets:
        for m in ("distilbert", "bert"):
            r = get(results, m, d)["metrics"]
            rows.append(f"{DATASET_LABEL[d]} & {MODEL_LABEL[m]} & "
                        f"{r['accuracy']*100:.2f} & {r['precision_macro']*100:.2f} & "
                        f"{r['recall_macro']*100:.2f} & {r['f1_macro']*100:.2f} \\\\")
        rows.append("\\addlinespace")
    body = ("\\begin{tabular}{llrrrr}\n\\toprule\n"
            "Dataset & Model & Accuracy & Precision & Recall & F1 \\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("table_main_results", body,
                caption="Performance of both backbones with the identical baseline classifier. "
                        "Precision, recall and F1 are macro averaged.", label="tab:main")


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
    rows = []
    for cfg in ABLATIONS:
        r0 = get(results, "distilbert", datasets[0], cfg.key)
        cells = []
        for d in datasets:
            mm = get(results, "distilbert", d, cfg.key)["metrics"]
            cells += [f"{mm['accuracy']*100:.2f}", f"{mm['f1_macro']*100:.2f}"]
        rows.append(f"{ABLATION_SHORT[cfg.key]} & {r0['params']['trainable_params']/1e6:.1f} & "
                    + " & ".join(cells) + " \\\\")
    cols = "lr" + "rr" * len(datasets)
    body = (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            f"Variant & Train (M) & {header} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("table_ablation", body,
                caption="Classifier ablation on DistilBERT. Train (M) is the number of trainable "
                        "parameters in millions.", label="tab:ablation")


def table_best_on_bert(results: dict) -> None:
    key = best_head(results)
    if key is None:
        return
    datasets = [d for d in ABLATION_DATASETS
                if get(results, "distilbert", d, key) and get(results, "bert", d, key)]
    if not datasets:
        return
    rows = []
    for d in datasets:
        for m in ("distilbert", "bert"):
            r = get(results, m, d, key)
            mm = r["metrics"]
            rows.append(f"{DATASET_LABEL[d]} & {MODEL_LABEL[m]} & "
                        f"{mm['accuracy']*100:.2f} & {mm['precision_macro']*100:.2f} & "
                        f"{mm['recall_macro']*100:.2f} & {mm['f1_macro']*100:.2f} & "
                        f"{r['params']['total_params']/1e6:.1f} & "
                        f"{r['latency']['latency_ms_p50']:.2f} & "
                        f"{r['training']['train_peak_gpu_mb']/1024:.2f} \\\\")
        rows.append("\\addlinespace")
    body = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
            "Dataset & Model & Accuracy & Precision & Recall & F1 & Params (M) & GPU ms & Train GiB \\\\\n"
            "\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("table_best_on_bert", body,
                caption="The selected classifier variant with each backbone, all metrics.",
                label="tab:best")
    write_table("best_variant_name", ABLATION_SHORT.get(key, key))


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
            spread = max(f1s[k] for k in ("D_narrow", "E_wide", "F_deep", "G_linear", "A_baseline")) - \
                     min(f1s[k] for k in ("D_narrow", "E_wide", "F_deep", "G_linear", "A_baseline"))
            m("headSpread", spread)
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
        m("sentCka", s_["meta"]["cka"], "{:.3f}")

    # providecommand plus renewcommand so this file can be read after the report
    # has already declared fallbacks for the same names.
    body = "\n".join(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}"
                     for k, v in macros.items()) + "\n"
    write_table("numbers", body)


def summary_markdown(results: dict) -> str:
    """The results summary that gets spliced into README.md."""
    datasets = available_main(results)
    if not datasets:
        return "Run the experiments first, then run src/figures.py.\n"

    out = ["## Results summary", "",
           "Generated by src/figures.py from artifacts/results. Do not edit by hand.", "",
           "### Performance, identical classifier on both backbones", "",
           "| Dataset | Model | Accuracy | Precision | Recall | F1 macro |",
           "|---|---|---|---|---|---|"]
    for d in datasets:
        for m_ in ("distilbert", "bert"):
            mm = get(results, m_, d)["metrics"]
            out.append(f"| {DATASET_LABEL[d]} | {MODEL_LABEL[m_]} | {mm['accuracy']*100:.2f} | "
                       f"{mm['precision_macro']*100:.2f} | {mm['recall_macro']*100:.2f} | "
                       f"{mm['f1_macro']*100:.2f} |")

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
    out += ["", "### Efficiency, averaged over the datasets", "",
            "| Measurement | BERT | DistilBERT | DistilBERT advantage |", "|---|---|---|---|"]
    for name, fn, fmt, higher_better in rows:
        b, dd = avg("bert", fn), avg("distilbert", fn)
        ratio = (dd / b) if higher_better else (b / dd if dd else float("nan"))
        out.append(f"| {name} | {fmt.format(b)} | {fmt.format(dd)} | {ratio:.2f}x |")

    key = best_head(results)
    if key:
        out += ["", f"### Ablation on the classifier, best variant: {ABLATION_SHORT.get(key, key)}", "",
                "| Variant | Trainable (M) | " +
                " | ".join(f"{DATASET_LABEL[d]} F1" for d in ABLATION_DATASETS) + " |",
                "|---" * (2 + len(ABLATION_DATASETS)) + "|"]
        for cfg in ABLATIONS:
            vals = [get(results, "distilbert", d, cfg.key) for d in ABLATION_DATASETS]
            if not all(vals):
                continue
            tp = vals[0]["params"]["trainable_params"] / 1e6
            cells = " | ".join(f"{v['metrics']['f1_macro']*100:.2f}" for v in vals)
            out.append(f"| {ABLATION_SHORT[cfg.key]} | {tp:.1f} | {cells} |")

        out += ["", "### The best variant with each backbone", "",
                "| Dataset | Model | Accuracy | Precision | Recall | F1 macro | Params (M) | GPU ms | Train GiB |",
                "|---|---|---|---|---|---|---|---|---|"]
        for d in ABLATION_DATASETS:
            for m_ in ("distilbert", "bert"):
                r = get(results, m_, d, key)
                if not r:
                    continue
                mm = r["metrics"]
                out.append(f"| {DATASET_LABEL[d]} | {MODEL_LABEL[m_]} | {mm['accuracy']*100:.2f} | "
                           f"{mm['precision_macro']*100:.2f} | {mm['recall_macro']*100:.2f} | "
                           f"{mm['f1_macro']*100:.2f} | {r['params']['total_params']/1e6:.1f} | "
                           f"{r['latency']['latency_ms_p50']:.2f} | "
                           f"{r['training']['train_peak_gpu_mb']/1024:.2f} |")

    ep = [(d, m_, get(results, m_, d, "A_baseline__ep4"))
          for d in EPOCH_DATASETS for m_ in ("distilbert", "bert")]
    ep = [(d, m_, r) for d, m_, r in ep if r]
    if ep:
        n_ep = len(ep[0][2]["training"]["per_epoch"])
        out += ["", "### Epoch study, validation F1 macro after each epoch", "",
                "| Dataset | Model | " + " | ".join(str(i) for i in range(1, n_ep + 1)) +
                " | Best |", "|---" * (3 + n_ep) + "|"]
        for d, m_, r in ep:
            cells = " | ".join(f"{x['val_f1_macro']*100:.2f}" for x in r["training"]["per_epoch"])
            out.append(f"| {DATASET_LABEL[d]} | {MODEL_LABEL[m_]} | {cells} | "
                       f"{r['training']['best_epoch_by_val_f1']} |")

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
                out.append(f"| {row['distil_layer']} | {row['bert_layer']} | {row['cka']:.3f} | "
                           f"{row['overlap']:.2f} |")
    return "\n".join(out) + "\n"


def update_readme(results: dict, path: str = "README.md") -> None:
    """Splice the generated summary into README.md between its own headings."""
    if not os.path.exists(path):
        return
    text = pathlib.Path(path).read_text()
    block = summary_markdown(results)
    marker = "## Results summary"
    if marker in text:
        start = text.index(marker)
        rest = text[start + len(marker):]
        nxt = rest.find("\n## ")
        end = len(text) if nxt == -1 else start + len(marker) + nxt + 1
        text = text[:start] + block + "\n" + text[end:]
    else:
        anchor = "## What is compared"
        text = text.replace(anchor, block + "\n" + anchor, 1)
    pathlib.Path(path).write_text(text)
    print("  README results summary updated", flush=True)


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
