# [AI-SCAFFOLD] Pipeline fixes implemented with AI assistance; include in course disclosure.
# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Dependencies are imported explicitly; run directly or with run_sections.py.

from importlib import import_module
import torch.nn as nn
import torch

# Numbered filenames require import_module rather than Python from-import syntax.
_section_01 = import_module('01setup')
DEVICE = _section_01.DEVICE

# %% [markdown]
# ## 10. Training
#
# - **Loss**: masked, weighted MSE. Entries with `ym = 0` (family not run / too few routes) contribute nothing; `w` upweights players with more routes (their averages are less noisy).
# - **Early stopping** uses a validation set carved out of the *training* players. Never early-stop on the test fold.
# - **Regularization**: dropout, AdamW weight decay, small hidden size, gradient clipping. Consider data augmentation (small random time shifts or noise on `x`) as a variation.
# - Targets are in separation units; standardizing `Y` per family using train statistics usually helps optimization (TODO).

# %%
def masked_mse(pred, y, ym, w):
    num = (((pred - y) ** 2) * ym * w).sum()
    den = (ym * w).sum().clamp(min=1e-8)
    return num / den

def to_dev(b):
    return {k: v.to(DEVICE) for k, v in b.items()}

@torch.no_grad()
def predict(model, dl):
    model.eval()
    return torch.cat([model(to_dev(b)).cpu() for b in dl]).numpy()

@torch.no_grad()
def eval_loss(model, dl):
    model.eval()
    numerator, denominator = 0.0, 0.0
    for b in dl:
        b = to_dev(b)
        weights = b["ym"] * b["w"]
        numerator += (((model(b) - b["y"]) ** 2) * weights).sum().item()
        denominator += weights.sum().item()
    if denominator == 0:
        raise ValueError("Validation set has no valid target weights")
    return numerator / denominator

def fit(model, train_dl, val_dl, cfg):
    """[AI-SCAFFOLD] AdamW + early stopping on the inner validation loss; restores the best weights."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_state, bad = float("inf"), None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for b in train_dl:
            b = to_dev(b)
            opt.zero_grad()
            loss = masked_mse(model(b), b["y"], b["ym"], b["w"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = eval_loss(model, val_dl)
        if not torch.isfinite(torch.tensor(v)):
            raise ValueError("Nonfinite validation loss")
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
