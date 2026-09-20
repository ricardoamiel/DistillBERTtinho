"""Backbone plus configurable classification head.

One class covers every experiment. The backbone is either BERT base or
DistilBERT base, and the head is an MLP whose depth and width are what the
ablation study varies. Both backbones are read the same way, through the
final hidden state of the CLS token, so the comparison stays fair.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from config import MODELS


def encoder_blocks(backbone) -> nn.ModuleList:
    """Return the list of transformer blocks for BERT or DistilBERT."""
    if hasattr(backbone, "encoder"):          # BERT
        return backbone.encoder.layer
    return backbone.transformer.layer         # DistilBERT


def embedding_module(backbone) -> nn.Module:
    return backbone.embeddings


class TextClassifier(nn.Module):
    """Transformer backbone with an MLP classifier on the CLS representation."""

    def __init__(
        self,
        backbone_key: str,
        num_labels: int,
        hidden_sizes: tuple[int, ...] = (768,),
        dropout: float = 0.1,
        frozen_layers: int | str = 0,
    ):
        super().__init__()
        checkpoint = MODELS[backbone_key]
        self.backbone_key = backbone_key
        self.checkpoint = checkpoint
        self.config = AutoConfig.from_pretrained(checkpoint)
        self.backbone = AutoModel.from_pretrained(checkpoint)

        width = self.config.hidden_size
        layers: list[nn.Module] = []
        for size in hidden_sizes:
            layers += [nn.Dropout(dropout), nn.Linear(width, size), nn.ReLU()]
            width = size
        layers += [nn.Dropout(dropout), nn.Linear(width, num_labels)]
        self.head = nn.Sequential(*layers)

        self.frozen_layers = frozen_layers
        self._apply_freezing(frozen_layers)

    def _apply_freezing(self, frozen_layers: int | str) -> None:
        """Freeze the embeddings and the first n transformer blocks."""
        if frozen_layers in (0, None):
            return
        for param in embedding_module(self.backbone).parameters():
            param.requires_grad = False
        blocks = encoder_blocks(self.backbone)
        n = len(blocks) if frozen_layers == "all" else int(frozen_layers)
        for block in blocks[:n]:
            for param in block.parameters():
                param.requires_grad = False
        if frozen_layers == "all":
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden[:, 0])

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        """CLS representation, used by the embedding export for the web demo."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def parameter_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters())
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
            "head_params": head,
            "backbone_params": total - head,
            "num_transformer_blocks": len(encoder_blocks(self.backbone)),
        }

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        """Lower learning rate for pretrained weights, higher for the new head."""
        no_decay = ("bias", "LayerNorm.weight", "layer_norm")

        def split(named):
            decay, plain = [], []
            for name, param in named:
                if not param.requires_grad:
                    continue
                (plain if any(k in name for k in no_decay) else decay).append(param)
            return decay, plain

        b_decay, b_plain = split(self.backbone.named_parameters())
        h_decay, h_plain = split(self.head.named_parameters())
        groups = [
            {"params": b_decay, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": b_plain, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": h_decay, "lr": lr_head, "weight_decay": weight_decay},
            {"params": h_plain, "lr": lr_head, "weight_decay": 0.0},
        ]
        return [g for g in groups if g["params"]]
