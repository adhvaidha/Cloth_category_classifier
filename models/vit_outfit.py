from typing import Optional

import torch
import torch.nn as nn


class OutfitCompatibilityModel(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_layers: int = 4, num_heads: int = 8, ff_multiplier: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_multiplier * embedding_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.compatibility_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D) sequence of item embeddings
        h = self.encoder(tokens)  # (B, N, D)
        pooled = h.mean(dim=1)    # (B, D)
        score = self.compatibility_head(pooled)  # (B, 1)
        return score.squeeze(-1)  # (B,)

    def score_compatibility(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens)
