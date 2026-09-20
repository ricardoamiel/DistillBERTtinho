"""Export the data consumed by the interactive web demo.

Three files land in web/data:

    words.json      496 words embedded by pretrained BERT and DistilBERT,
                    with a 2D layout and the ten nearest neighbours of each
                    word inside each model.
    sentences.json  AG News test sentences embedded by both fine tuned models
                    and coloured by their true topic.
    summary.json    headline accuracy, size and speed numbers.

The visual claim is made in 2D, but every number reported in the page is
computed in the original 768 dimensional space.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

from config import MODELS, RESULTS_DIR, WEB_DATA_DIR
from wordlist import flat_words

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10
SEED = 42


# --------------------------------------------------------------------------
# embedding helpers
# --------------------------------------------------------------------------

@torch.no_grad()
def embed_words(checkpoint: str, words: list[str], batch_size: int = 64) -> np.ndarray:
    """Sub token mean of every word at every layer.

    Returns an array of shape [n_layers + 1, n_words, hidden]. Index 0 is the
    embedding output, index l is the output of transformer block l.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, output_hidden_states=True).to(DEVICE).eval()
    out = None
    for start in range(0, len(words), batch_size):
        chunk = words[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        states = model(**enc).hidden_states
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0                                   # drop CLS
        mask[torch.arange(mask.size(0)), enc["attention_mask"].sum(1) - 1] = 0   # drop SEP
        weights = mask.unsqueeze(-1).float()
        pooled = torch.stack([(h * weights).sum(1) / weights.sum(1).clamp(min=1)
                              for h in states]).float().cpu().numpy()
        out = pooled if out is None else np.concatenate([out, pooled], axis=1)
    del model
    torch.cuda.empty_cache()
    return out


def word_space(layers: np.ndarray) -> np.ndarray:
    """One vector per word: the mean over transformer layers, then mean centred.

    Two well known properties of BERT motivate both steps. The last layer is
    specialised for the pretraining objective and is a poor place to read word
    similarity, so we average over the layers instead. The resulting vectors are
    strongly anisotropic, all pointing into a narrow cone, so the mean is removed
    before any cosine is taken.
    """
    pooled = layers[1:].mean(axis=0)
    return pooled - pooled.mean(axis=0)


def group_purity(matrix: np.ndarray, groups: list[str], k: int = TOP_K) -> float:
    """Share of a word's k nearest neighbours that carry its own topic label."""
    idx, _ = neighbours(matrix, k)
    return float(np.mean([[groups[j] for j in row].count(groups[i]) / k
                          for i, row in enumerate(idx)]))


def layer_profile(bert: np.ndarray, distil: np.ndarray) -> list[dict]:
    """Alignment at matched depths, from the embeddings up to the last block.

    DistilBERT has half the blocks of BERT, so block d of the student is compared
    with block 2d of the teacher, which is the layer it was initialised from.
    """
    rows = []
    n_distil = distil.shape[0] - 1
    for d in range(n_distil + 1):
        b = min(2 * d, bert.shape[0] - 1)
        rows.append({
            "distil_layer": d,
            "bert_layer": b,
            "cka": round(cka(bert[b], distil[d]), 4),
            "overlap": round(float(knn_agreement(bert[b], distil[d], TOP_K).mean()), 3),
        })
    return rows


@torch.no_grad()
def embed_sentences(encoder_dir: str, texts: list[str], max_length: int = 128,
                    batch_size: int = 64) -> np.ndarray:
    """CLS vector of a fine tuned encoder."""
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    model = AutoModel.from_pretrained(encoder_dir).to(DEVICE).eval()
    vectors = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        pooled = model(**enc).last_hidden_state[:, 0]
        vectors.append(pooled.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(vectors)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-9)


def neighbours(matrix: np.ndarray, k: int = TOP_K):
    """Indices and cosine scores of the k nearest neighbours of every row."""
    sim = unit(matrix) @ unit(matrix).T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argsort(-sim, axis=1)[:, :k]
    scores = np.take_along_axis(sim, idx, axis=1)
    return idx, scores


def project_2d(matrix: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    """PCA down to 50 dimensions, then t-SNE down to 2."""
    reduced = PCA(n_components=min(50, matrix.shape[1], matrix.shape[0] - 1),
                  random_state=SEED).fit_transform(matrix)
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=SEED, max_iter=1000).fit_transform(reduced)


def align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate, scale and shift source onto target (orthogonal Procrustes).

    The two projections are computed independently, so this only removes the
    arbitrary orientation of t-SNE and makes the two panels comparable by eye.
    """
    a = source - source.mean(0)
    b = target - target.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    rotation = u @ vt
    scale = s.sum() / (a ** 2).sum()
    return (a @ rotation) * scale + target.mean(0)


def normalise(points: np.ndarray) -> np.ndarray:
    """Map a 2D cloud into the unit square so the front end can scale it."""
    lo, hi = points.min(0), points.max(0)
    return (points - lo) / np.maximum(hi - lo, 1e-9)


def knn_agreement(a: np.ndarray, b: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Per row overlap between the k nearest neighbours of two spaces."""
    idx_a, _ = neighbours(a, k)
    idx_b, _ = neighbours(b, k)
    return np.array([len(set(x) & set(y)) for x, y in zip(idx_a, idx_b)])


def cka(a: np.ndarray, b: np.ndarray) -> float:
    """Linear centred kernel alignment between two representation matrices."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    num = np.linalg.norm(a.T @ b, "fro") ** 2
    den = np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")
    return float(num / max(den, 1e-12))


# --------------------------------------------------------------------------
# exports
# --------------------------------------------------------------------------

def export_words(out_dir: str) -> dict:
    pairs = flat_words()
    words = [w for w, _ in pairs]
    groups = [g for _, g in pairs]

    print("embedding words with both pretrained backbones", flush=True)
    layers = {key: embed_words(ckpt, words) for key, ckpt in MODELS.items()}
    profile = layer_profile(layers["bert"], layers["distilbert"])
    emb = {key: word_space(v) for key, v in layers.items()}
    purity = {key: round(group_purity(v, groups), 4) for key, v in emb.items()}

    xy_bert = normalise(project_2d(emb["bert"]))
    xy_distil = normalise(align(project_2d(emb["distilbert"]), xy_bert))

    nn_idx, nn_score = {}, {}
    for key in MODELS:
        nn_idx[key], nn_score[key] = neighbours(emb[key], TOP_K)

    overlap = knn_agreement(emb["bert"], emb["distilbert"], TOP_K)

    items = []
    for i, (word, group) in enumerate(pairs):
        items.append({
            "i": i,
            "word": word,
            "group": group,
            "bert": [round(float(xy_bert[i, 0]), 4), round(float(xy_bert[i, 1]), 4)],
            "distilbert": [round(float(xy_distil[i, 0]), 4), round(float(xy_distil[i, 1]), 4)],
            "nn": {
                key: [{"i": int(j), "s": round(float(s), 4)}
                      for j, s in zip(nn_idx[key][i], nn_score[key][i])]
                for key in MODELS
            },
            "overlap": int(overlap[i]),
        })

    payload = {
        "words": items,
        "groups": sorted(set(groups)),
        "meta": {
            "top_k": TOP_K,
            "n_words": len(items),
            "mean_overlap": round(float(overlap.mean()), 3),
            "overlap_pct": round(float(overlap.mean()) / TOP_K * 100, 1),
            "cka": round(cka(emb["bert"], emb["distilbert"]), 4),
            "purity": purity,
            "layer_profile": profile,
            "note": "Word vectors are the mean over transformer layers, mean centred to "
                    "remove anisotropy. Neighbours, agreement and CKA are computed in 768 "
                    "dimensions. The 2D layout is PCA then t-SNE, aligned by Procrustes.",
        },
    }
    with open(os.path.join(out_dir, "words.json"), "w") as fh:
        json.dump(payload, fh)
    print(f"  mean neighbour overlap {payload['meta']['mean_overlap']}/{TOP_K}, "
          f"CKA {payload['meta']['cka']}", flush=True)
    print(f"  topic purity: BERT {purity['bert']}, DistilBERT {purity['distilbert']}", flush=True)
    print("  alignment by depth (distil layer, bert layer, CKA, overlap):", flush=True)
    for row in profile:
        print(f"    D{row['distil_layer']} B{row['bert_layer']}: "
              f"CKA {row['cka']:.3f}  overlap {row['overlap']:.2f}", flush=True)
    return payload["meta"]


def export_sentences(out_dir: str, n: int = 1200, max_chars: int = 220) -> dict:
    from datasets import load_dataset

    dirs = {key: os.path.join("artifacts", "encoders", f"{key}__ag_news__A_baseline")
            for key in MODELS}
    missing = [k for k, d in dirs.items() if not os.path.isdir(d)]
    if missing:
        print(f"  skipping sentences, missing encoders for {missing}", flush=True)
        return {}

    labels_names = ["World", "Sports", "Business", "Sci/Tech"]
    ds = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=SEED).select(range(n))
    texts = [t for t in ds["text"]]
    labels = list(ds["label"])

    print("embedding AG News sentences with both fine tuned encoders", flush=True)
    emb = {key: embed_sentences(dirs[key], texts) for key in MODELS}

    xy_bert = normalise(project_2d(emb["bert"]))
    xy_distil = normalise(align(project_2d(emb["distilbert"]), xy_bert))
    overlap = knn_agreement(emb["bert"], emb["distilbert"], TOP_K)

    items = []
    for i, text in enumerate(texts):
        clean = " ".join(text.split())
        items.append({
            "i": i,
            "text": clean[:max_chars] + ("..." if len(clean) > max_chars else ""),
            "label": int(labels[i]),
            "bert": [round(float(xy_bert[i, 0]), 4), round(float(xy_bert[i, 1]), 4)],
            "distilbert": [round(float(xy_distil[i, 0]), 4), round(float(xy_distil[i, 1]), 4)],
        })

    meta = {
        "n": len(items),
        "mean_overlap": round(float(overlap.mean()), 3),
        "overlap_pct": round(float(overlap.mean()) / TOP_K * 100, 1),
        "cka": round(cka(emb["bert"], emb["distilbert"]), 4),
    }
    with open(os.path.join(out_dir, "sentences.json"), "w") as fh:
        json.dump({"sentences": items, "labels": labels_names, "meta": meta}, fh)
    print(f"  sentence CKA {meta['cka']}", flush=True)
    return meta


def export_summary(out_dir: str, results_dir: str = RESULTS_DIR) -> None:
    """Collect the headline numbers the page shows next to the scatter plots."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*__A_baseline.json"))):
        with open(path) as fh:
            r = json.load(fh)
        rows.append({
            "model": r["model"],
            "dataset": r["dataset"],
            "accuracy": round(r["metrics"]["accuracy"] * 100, 2),
            "f1_macro": round(r["metrics"]["f1_macro"] * 100, 2),
            "params_m": round(r["params"]["total_params"] / 1e6, 1),
            "latency_ms": round(r["latency"]["latency_ms_p50"], 2),
            "throughput": round(r["latency"]["throughput_samples_per_s"], 1),
            "train_gpu_mb": round(r["training"]["train_peak_gpu_mb"], 1),
            "train_seconds": round(r["training"]["train_seconds"], 1),
        })
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"runs": rows}, fh, indent=2)
    print(f"  summary rows: {len(rows)}", flush=True)


def validate(out_dir: str) -> None:
    """Check the invariants the web page depends on, so silent corruption fails loudly."""
    with open(os.path.join(out_dir, "words.json")) as fh:
        data = json.load(fh)
    words, meta = data["words"], data["meta"]
    k = meta["top_k"]

    assert len(words) == meta["n_words"], "word count does not match the metadata"
    for w in words:
        for key in MODELS:
            assert len(w["nn"][key]) == k, f"{w['word']} has no {key} neighbour list"
        shared = len({n["i"] for n in w["nn"]["bert"]} & {n["i"] for n in w["nn"]["distilbert"]})
        assert shared == w["overlap"], f"{w['word']} overlap disagrees with its neighbour lists"
    mean = sum(w["overlap"] for w in words) / len(words)
    assert abs(mean - meta["mean_overlap"]) < 1e-3, "mean overlap disagrees with the per word values"  # metadata is rounded to 3 decimals
    assert 0.0 <= meta["cka"] <= 1.0, "CKA outside its valid range"
    print(f"  validated: {len(words)} words, every neighbour list complete and consistent", flush=True)


def main() -> None:
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    export_words(WEB_DATA_DIR)
    validate(WEB_DATA_DIR)
    export_sentences(WEB_DATA_DIR)
    export_summary(WEB_DATA_DIR)
    print("web data written to", WEB_DATA_DIR)


if __name__ == "__main__":
    main()
