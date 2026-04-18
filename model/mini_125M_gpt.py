import torch
import torch.nn as nn
import math


# ======================
# CONFIG
# ======================

class GPTConfig:
    vocab_size = 50000
    block_size = 128
    n_layer = 12
    n_head = 12
    n_embd = 768


# ======================
# ATTENTION
# ======================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            batch_first=True
        )

    def forward(self, x):

        T = x.size(1)

        mask = torch.triu(
            torch.ones(T, T),
            diagonal=1
        ).bool().to(x.device)

        y, _ = self.attn(
            x, x, x,
            attn_mask=mask
        )

        return y


# ======================
# TRANSFORMER BLOCK
# ======================

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


# ======================
# GPT MODEL
# ======================

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                config.block_size,
                config.n_embd
            )
        )

        self.blocks = nn.Sequential(
            *[Block(config)
              for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

    def forward(self, idx):

        B, T = idx.shape

        tok = self.token_emb(idx)
        x = tok + self.pos_emb[:, :T]

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)

        return logits

if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)

    print("Model created")

    torch.save(model.state_dict(), "gpt_125m.pt")

    print("Model saved successfully")