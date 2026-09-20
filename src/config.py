"""Central configuration for the DistilBERT vs BERT study.

Everything that an experiment needs to be reproducible lives here: which
checkpoints are compared, how each dataset is read and sized, and which
classifier variants make up the ablation grid.
"""

from dataclasses import dataclass, field

SEED = 42

# Backbone checkpoints under comparison.
MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
}


@dataclass
class DatasetSpec:
    """How to load and size one text classification dataset."""

    path: str                 # Hugging Face dataset id
    name: str | None          # configuration name, if any
    text_field: str           # column holding the raw text
    num_labels: int
    max_length: int           # tokenizer truncation length
    test_split: str           # split used for the final reported metrics
    train_size: int | None = None   # subsample size, None keeps everything
    test_size: int | None = None
    label_names: tuple[str, ...] = ()


DATASETS = {
    "ag_news": DatasetSpec(
        path="fancyzhx/ag_news", name=None, text_field="text", num_labels=4,
        max_length=128, test_split="test",
        label_names=("World", "Sports", "Business", "Sci/Tech"),
    ),
    "sst2": DatasetSpec(
        path="nyu-mll/glue", name="sst2", text_field="sentence", num_labels=2,
        max_length=64, test_split="validation",
        label_names=("negative", "positive"),
    ),
    "yelp_polarity": DatasetSpec(
        path="fancyzhx/yelp_polarity", name=None, text_field="text", num_labels=2,
        max_length=256, test_split="test",
        train_size=100_000, test_size=10_000,
        label_names=("negative", "positive"),
    ),
    "yelp_full": DatasetSpec(
        path="Yelp/yelp_review_full", name=None, text_field="text", num_labels=5,
        max_length=256, test_split="test",
        train_size=100_000, test_size=10_000,
        label_names=("1 star", "2 stars", "3 stars", "4 stars", "5 stars"),
    ),
}

# How each dataset is written wherever a human reads it.
DATASET_LABELS = {
    "ag_news": "AG News",
    "sst2": "SST 2",
    "yelp_polarity": "Yelp Polarity",
    "yelp_full": "Yelp Full",
}

# Datasets used for the headline BERT vs DistilBERT comparison.
MAIN_DATASETS = ["ag_news", "sst2", "yelp_polarity", "yelp_full"]

# Datasets used to select the best classifier head in the ablation study.
ABLATION_DATASETS = ["sst2", "ag_news"]

# Datasets used for the epoch study. Devlin et al. recommend 2, 3 or 4 epochs for
# fine tuning, so we train to the top of that range and score every epoch.
EPOCH_DATASETS = ["sst2", "ag_news"]
EPOCH_STUDY_EPOCHS = 4


@dataclass
class TrainConfig:
    """Optimisation settings shared by every run."""

    epochs: int = 2
    batch_size: int = 32
    eval_batch_size: int = 64
    lr_backbone: float = 2e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    eval_points: int = 12        # how many (train loss, validation loss) samples to record
    val_subset: int = 2000       # examples used for the periodic validation loss
    val_fraction: float = 0.05   # slice of train held out for the loss curves
    amp: bool = True
    seed: int = SEED
    eval_each_epoch: bool = False   # epoch study: score validation and test at every epoch boundary


@dataclass
class HeadConfig:
    """One classifier variant of the ablation study.

    The three axes named in the assignment map onto three fields:
    freezing the transformer (frozen_layers), neurons per classifier layer
    (hidden_sizes values) and number of classifier layers (hidden_sizes length).
    """

    key: str
    label: str
    hidden_sizes: tuple[int, ...] = (768,)
    frozen_layers: int | str = 0    # 0 none, integer n freezes the first n blocks, "all" freezes the backbone
    dropout: float = 0.1
    note: str = ""


ABLATIONS = [
    HeadConfig("A_baseline", "Baseline head 1x768, full fine tuning",
               hidden_sizes=(768,), frozen_layers=0,
               note="Reproduces the default DistilBERT classification head."),
    HeadConfig("B_frozen_all", "Frozen transformer, head 1x768",
               hidden_sizes=(768,), frozen_layers="all",
               note="Linear probe: only the classifier receives gradients."),
    HeadConfig("C_frozen_half", "First 3 blocks frozen, head 1x768",
               hidden_sizes=(768,), frozen_layers=3,
               note="Keeps the lower half of the transformer fixed."),
    HeadConfig("D_narrow", "Narrow head 1x128, full fine tuning",
               hidden_sizes=(128,), frozen_layers=0,
               note="Fewer neurons per classifier layer."),
    HeadConfig("E_wide", "Wide head 1x2048, full fine tuning",
               hidden_sizes=(2048,), frozen_layers=0,
               note="More neurons per classifier layer."),
    HeadConfig("F_deep", "Deep head 3x768, full fine tuning",
               hidden_sizes=(768, 768, 768), frozen_layers=0,
               note="More classifier layers."),
    HeadConfig("G_linear", "No hidden layer, full fine tuning",
               hidden_sizes=(), frozen_layers=0,
               note="Pooled CLS vector goes straight to the logits."),
]

ABLATION_BY_KEY = {cfg.key: cfg for cfg in ABLATIONS}

# Directory layout, relative to the repository root.
RESULTS_DIR = "artifacts/results"
FIGURES_DIR = "artifacts/figures"
LOGS_DIR = "artifacts/logs"
WEB_DATA_DIR = "web/data"
