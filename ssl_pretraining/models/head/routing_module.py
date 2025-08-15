

import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptMapper(nn.Module):
    def __init__(self, vocab_size=30522, embedding_dim=1024, hidden_dim=2048, output_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            emb = emb * mask
            pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = emb.mean(dim=1)

        logits = self.encoder(pooled)
        probs = torch.sigmoid(logits)
        return emb, probs