# ## 1. Setup and configuration
#
# Keep tunable values here. Everything that other sections treat as a decision (target definition, route families, drill scope, channel sets) is defined once in this section. 

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

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SEP_COL = "separation_at_pass_forward"

MIN_ROUTES = 20        # min routes WITH a separation value per family to get a target (a COUNT, not a distance).
MIN_ROUTE_REPS = 3     # min route-drill reps per WR

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
# drill_type -> encoder family. 3-cone / shuttle TRACKING is too sparse
FAMILY_OF_DRILL_TYPE = {"FORTY_YARD_DASH": "sprint", "SKILL_DRILLS_WR": "route"}
FAMILIES = sorted(set(FAMILY_OF_DRILL_TYPE.values()))
FAMILY_ID = {f: i for i, f in enumerate(FAMILIES)}

# Left out of v1: ball-skill drills (not real routes)
EXCLUDE_BASE = {"GAUNTLET_DRILL", "OVER_SHOULDER_ADJUST"}

# ---------------- Channel sets (these is the variable options in section 12) ----------------
# Derived channels are created in section 4; `s` and `a` come from the raw tracking data.
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
if __name__ == "__main__":
    print(DEVICE, "| channels:", CFG["channels"])
