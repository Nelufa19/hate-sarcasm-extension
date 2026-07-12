"""
Model architecture — copied verbatim from the training notebook:
Phase 6 (Architectures)/distilbert-base (Lightweight model)/lightweight-model.ipynb

Do not "clean up" this class. The forward() method intentionally indexes into
hate_head[0]/[3] and sarcasm_head[0]/[3] instead of calling the Sequential
blocks directly — that is exactly how the checkpoints were trained, and
load_state_dict() matches weights by layer name/shape, not by behavior.
A behavioral rewrite (e.g. calling self.hate_head(cls_output) directly)
would load without error but silently produce different numbers, because
it would run through Dropout twice instead of once.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class LightweightDualHeadModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.hate_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.sarcasm_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size + 128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

        hate_hidden = F.relu(self.hate_head[0](cls_output))
        hate_hidden = self.dropout(hate_hidden)
        hate_logit = self.hate_head[3](hate_hidden)

        sarc_hidden = F.relu(self.sarcasm_head[0](cls_output))
        sarc_hidden = self.dropout(sarc_hidden)
        sarc_logit = self.sarcasm_head[3](sarc_hidden)

        fused = torch.cat([cls_output, hate_hidden, sarc_hidden], dim=1)
        logits_3class = self.fusion_mlp(fused)

        return logits_3class, hate_logit.squeeze(-1), sarc_logit.squeeze(-1)
