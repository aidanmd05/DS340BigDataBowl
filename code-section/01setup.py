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

# TODO: point this at the folder holding the competition CSVs.
# On Kaggle: Path("/kaggle/input/nfl-big-data-bowl-2027")
# On Colab: upload to Drive (or download with the Kaggle API) and mount it.
DATA_DIR = Path("data")

CFG = dict(
    T = 100,                  # fixed length every rep is resampled to
    channels = ["speed", "acc", "turn"],  # TODO: decide final channel list (section 5)
    min_routes_per_family = 15,   # a player needs this many routes in a family to get a target
    min_reps_per_player = 3,      # drop players with fewer usable reps
    hidden = 32,                  # LSTM hidden size (keep small: ~100 players!)
    drill_dim = 4,                # size of the drill-type embedding
    pool = "mean",                # "mean" | "max" | "weighted" | "attention"
    per_family_encoders = False,  # False = one shared LSTM; True = one LSTM per drill family
    use_attrs = True,             # feed official measurables into the head
    dropout = 0.3,
    lr = 1e-3, weight_decay = 1e-2,
    epochs = 200, patience = 20, batch_size = 16,
    n_folds = 5, n_seeds = 3,
)
print(DEVICE)