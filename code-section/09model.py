# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Dependencies are imported explicitly; run directly or with run_sections.py.

from importlib import import_module
import torch.nn as nn
import torch

# Numbered filenames require import_module rather than Python from-import syntax.
_section_01 = import_module('01setup')
FAMILIES = _section_01.FAMILIES

# %% [markdown]
# ## 9. Model
#
# Three parts, each a variation point:
#
# 1. **RepEncoder**: an LSTM over one rep. The drill embedding is concatenated to *every timestep*, so a shared LSTM can still behave differently per drill. With `per_family_encoders=True` each drill family gets its own LSTM (bias-variance tradeoff: more specialization, less data per model).
# 2. **Pooling** over the reps of a player, respecting the mask: `mean`, `max`, `weighted` (one learned score per rep, softmax) or `attention` (a single small self-attention block, then mean over valid reps).
# 3. **Head**: an MLP that outputs one value per route family (multi-output), with optional attributes concatenated to the player vector.
#
# Gotchas:
# - A player with **zero** valid reps would give NaNs in attention (all keys masked). Section 7 already drops players below `min_reps_per_player`.
# - Keep the model tiny. About 100 players cannot support a large network.

# %%
class RepEncoder(nn.Module):
    """[AI-SCAFFOLD] LSTM over one rep; drill embedding appended at every timestep."""
    def __init__(self, n_channels, n_drills, drill_dim, hidden, dropout):
        super().__init__()
        self.drill_emb = nn.Embedding(n_drills, drill_dim)
        self.lstm = nn.LSTM(n_channels + drill_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, drill_id):                      # x: (N, T, C), drill_id: (N,)
        d = self.drill_emb(drill_id).unsqueeze(1).expand(-1, x.size(1), -1)
        _, (h, _) = self.lstm(torch.cat([x, d], dim=-1))
        return self.drop(h[-1])                          # (N, hidden)

class Pool(nn.Module):
    """[AI-SCAFFOLD] Mask-aware pooling over reps. h: (B, R, D), mask: (B, R) True = real rep."""
    def __init__(self, kind, dim, heads=2, dropout=0.1):
        super().__init__()
        self.kind = kind
        self.last_weights = None                         # saved for interpretation (section 13)
        if kind == "weighted":
            self.score = nn.Linear(dim, 1)
        if kind == "attention":
            self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(dim)
    def forward(self, h, mask):
        m = mask.unsqueeze(-1).float()
        if self.kind == "mean":
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        if self.kind == "max":
            return h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        if self.kind == "weighted":
            s = self.score(h).squeeze(-1).masked_fill(~mask, -1e9)
            w = torch.softmax(s, dim=1)
            self.last_weights = w.detach()
            return (w.unsqueeze(-1) * h).sum(1)
        if self.kind == "attention":
            out, w = self.mha(h, h, h, key_padding_mask=~mask, need_weights=True)
            self.last_weights = w.detach()               # (B, R, R); average over queries to rank reps
            h2 = self.norm(h + out)                      # residual + norm
            return (h2 * m).sum(1) / m.sum(1).clamp(min=1)
        raise ValueError(self.kind)

class PlayerModel(nn.Module):
    def __init__(self, n_channels, n_drills, n_attrs, n_out, cfg):
        super().__init__()
        self.cfg = cfg
        n_enc = len(FAMILIES) if cfg["per_family_encoders"] else 1
        self.encoders = nn.ModuleDict({str(i): RepEncoder(n_channels, n_drills, cfg["drill_dim"],
                                                          cfg["hidden"], cfg["dropout"])
                                       for i in range(n_enc)})
        self.pool = Pool(cfg["pool"], cfg["hidden"])
        head_in = cfg["hidden"] + (n_attrs if cfg["use_attrs"] else 0)
        self.head = nn.Sequential(nn.Linear(head_in, 32), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                                  nn.Linear(32, n_out))
    def encode_reps(self, x, drill, fam, mask):
        B, R, T, C = x.shape
        fx, fd = x.reshape(B * R, T, C), drill.reshape(-1)
        ff = fam.reshape(-1) if self.cfg["per_family_encoders"] else torch.zeros_like(fd)
        valid = mask.reshape(-1)
        out = x.new_zeros(B * R, self.cfg["hidden"])
        for key, enc in self.encoders.items():
            sel = (ff == int(key)) & valid
            if sel.any():
                out[sel] = enc(fx[sel], fd[sel])
        return out.reshape(B, R, -1)
    def forward(self, b):
        h = self.encode_reps(b["x"], b["drill"], b["fam"], b["mask"])
        v = self.pool(h, b["mask"])                      # player vector
        if self.cfg["use_attrs"]:
            v = torch.cat([v, b["attrs"]], dim=-1)
        return self.head(v)                              # (B, n_out)
