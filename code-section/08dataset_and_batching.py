# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Run sections in order with run_sections.py (shared notebook state).

# %% [markdown]
# ## 8. Dataset and batching (variable number of reps per player)
#
# A training example is a **player**: a set of reps (each `(T, C)`), their drill and family ids, an attribute vector, and the targets with their mask and weights.
#
# Players have different numbers of reps, so the collate function pads to the largest count in the batch and returns a boolean `mask` marking real reps. Every later module must respect that mask (padding must never influence pooling or attention).
#
# Standardization of channels and attributes uses statistics computed from the **training fold only** and is passed into the dataset.

# %%
def fit_scalers(train_player_ids, reps_by_player, attrs):
    """[AI-SCAFFOLD] Channel and attribute mean/std from TRAIN players only."""
    allx = np.concatenate([r["x"] for p in train_player_ids for r in reps_by_player[p]], axis=0)  # (N*T, C)
    ch_mean, ch_std = allx.mean(0), allx.std(0) + 1e-6
    return ch_mean, ch_std, attrs.mean(0), attrs.std(0) + 1e-6

class PlayerDataset(Dataset):
    def __init__(self, idx, player_ids, reps_by_player, attrs, Y, YM, W, scalers):
        self.idx, self.player_ids = idx, player_ids
        self.reps_by_player, self.attrs = reps_by_player, attrs
        self.Y, self.YM, self.W = Y, YM, W
        self.ch_mean, self.ch_std, self.a_mean, self.a_std = scalers
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        reps = self.reps_by_player[self.player_ids[j]]
        x = np.stack([(r["x"] - self.ch_mean) / self.ch_std for r in reps])      # (R, T, C)
        return dict(
            x=torch.tensor(x, dtype=torch.float32),
            drill=torch.tensor([r["drill_id"] for r in reps]),
            fam=torch.tensor([r["family_id"] for r in reps]),
            attrs=torch.tensor((self.attrs[j] - self.a_mean) / self.a_std, dtype=torch.float32),
            y=torch.tensor(self.Y[j]), ym=torch.tensor(self.YM[j]), w=torch.tensor(self.W[j]))

def collate(batch):
    """Pad the rep dimension to the max count in the batch and build the validity mask."""
    B = len(batch)
    R = max(b["x"].shape[0] for b in batch)
    T, C = batch[0]["x"].shape[1:]
    x = torch.zeros(B, R, T, C)
    drill = torch.zeros(B, R, dtype=torch.long)
    fam = torch.zeros(B, R, dtype=torch.long)
    mask = torch.zeros(B, R, dtype=torch.bool)
    for i, b in enumerate(batch):
        r = b["x"].shape[0]
        x[i, :r], drill[i, :r], fam[i, :r], mask[i, :r] = b["x"], b["drill"], b["fam"], True
    out = dict(x=x, drill=drill, fam=fam, mask=mask)
    for k in ["attrs", "y", "ym", "w"]:
        out[k] = torch.stack([b[k] for b in batch])
    return out
