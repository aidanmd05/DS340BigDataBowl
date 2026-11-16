# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Run sections in order with run_sections.py (shared notebook state).

# %% [markdown]
# ## 1. Setup and configuration
#
# Keep every tunable value here so the ablations in section 12 are just different configs. Everything that other sections treat as a decision (target definition, route families, drill scope, channel sets) is defined **once** in this section. Do not redefine `ROUTE_FAMILY`, `ROUTE_FAMS` or `SEP_COL` in later cells.
#
# The course runs on Colab, so install anything missing in a separate cell; torch, sklearn, pandas and scipy are normally preinstalled.

# %%
import random, math, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

SEED = 0
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder holding the WR-only CSVs (and, for the one-time build, the full competition CSVs).
# Your earlier notebook read/wrote them in the working directory, hence ".".
# On Kaggle: Path("/kaggle/input/nfl-big-data-bowl-2027"); on Colab: a mounted Drive folder.
DATA_DIR = Path("data")

# %%
# ---------------- Target definition (decided from the audit) ----------------
SEP_COL = "separation_at_pass_forward"

MIN_ROUTES = 20        # min routes WITH a separation value per family to get a target (a COUNT, not a distance).
                       # Split-half reliability plateaus around 20; run 15 and 25 as sensitivity checks.
MIN_ROUTE_REPS = 3     # min route-drill reps per WR (run 6 as a sensitivity check)

# Single source of truth for route families.
# FLAT and SCREEN are left out on purpose: they happen at or behind the line, so separation
# at pass forward means something different there.
ROUTE_FAMILY = {
    "GO": "vertical", "POST": "vertical", "CORNER": "vertical", "WHEEL": "vertical",
    "OUT": "break", "IN": "break", "CROSS": "break", "SLANT": "break", "ANGLE": "break",
    "HITCH": "short",
}
ROUTE_FAMS = sorted(set(ROUTE_FAMILY.values()))

# ---------------- Combine drill scope ----------------
# drill_type -> encoder family. 3-cone / shuttle TRACKING is too sparse (about 0.3 reps per WR):
# their official times go into feature set A instead.
FAMILY_OF_DRILL_TYPE = {"FORTY_YARD_DASH": "sprint", "SKILL_DRILLS_WR": "route"}
FAMILIES = sorted(set(FAMILY_OF_DRILL_TYPE.values()))
FAMILY_ID = {f: i for i, f in enumerate(FAMILIES)}

# Left out of v1: ball-skill drills (not real routes) and a drill run in only one draft class (CIRCUS = 2025 only).
EXCLUDE_BASE = {"CIRCUS", "GAUNTLET_DRILL", "OVER_SHOULDER_ADJUST"}

# ---------------- Channel sets (this is the ablation in section 12) ----------------
# All of these columns are created by add_kinematics() in section 4.
CHANNEL_SETS = {
    "derived":   ["speed", "acc_tan", "acc_cent"],            # default: signed, physically meaningful
    "derived+a": ["speed", "acc_tan", "acc_cent", "a"],       # plus the sensor's total acceleration
    "turn":      ["speed", "acc_tan", "turn"],                # turn rate instead of sideways acceleration
    "raw":       ["s", "a", "dir_sin", "dir_cos"],            # "as given" (dir is NOT mirrored for left-side drills)
}

CFG = dict(
    T = 100,                              # fixed length every rep is resampled to
    channels = CHANNEL_SETS["derived"],
    min_routes_per_family = MIN_ROUTES,   # used by build_targets() in section 6
    min_reps_per_player = MIN_ROUTE_REPS, # used by make_player_table() in section 7
    hidden = 32,                  # LSTM hidden size (keep small: about 100 players!)
    drill_dim = 4,                # size of the drill embedding
    pool = "mean",                # "mean" | "max" | "weighted" | "attention"
    per_family_encoders = False,  # False = one shared LSTM; True = one LSTM per drill family
    use_attrs = True,             # feed official measurables into the head
    dropout = 0.3,
    lr = 1e-3, weight_decay = 1e-2,
    epochs = 200, patience = 20, batch_size = 16,
    n_folds = 5, n_seeds = 3,
)
print(DEVICE, "| channels:", CFG["channels"])
