# [AI-SCAFFOLD] Pipeline fixes implemented with AI assistance; include in course disclosure.
# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Dependencies are imported explicitly; run directly or with run_sections.py.

from importlib import import_module
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Numbered filenames require import_module rather than Python from-import syntax.
_section_01 = import_module('01setup')
CFG = _section_01.CFG
ROUTE_FAMS = _section_01.ROUTE_FAMS
SEED = _section_01.SEED
_section_02 = import_module('02load_data')
results = _section_02.results
_section_05 = import_module('05rep_tensors')
reps = _section_05.reps
_section_06 = import_module('06targets')
targets = _section_06.targets

# %% [markdown]
# ## 7. Baselines A and B with the evaluation harness (your milestone)
#
# Everything here is scored with the **same folds and metrics** later used for the neural model, so numbers are comparable.
#
# **Splits.** Players are the unit. Default: shuffled K-fold over players (each player appears in exactly one test fold). Secondary check: **leave one draft class out** (the realistic "predict the next class" test), if the audit shows enough WRs per class.
#
# **Metrics.** Per route family: Spearman correlation between predicted and actual (ranking is what a scout cares about) and R^2. Report the mean across families and the per-family values, with a bootstrap over players for confidence intervals (n is small, so intervals matter).
#
# **Baselines.**
# - **A**: official numbers (40, 3-cone, shuttle, vertical, broad jump, height, weight; draft pick as a variation). Ridge per route family.
# - **B**: hand-built kinematics per player: top speed, time to 90% of top speed, peak acceleration, hardest deceleration, peak turn rate, speed retained after the break. Ridge (or a small random forest from class) per route family.
#
# If A or B matches the neural model, that is a result worth reporting.

# %%
def make_player_table(reps, targets, results=None, cfg=None):
    """[AI-SCAFFOLD] One entry per WR that has targets and >= min_reps_per_player reps."""
    cfg = CFG if cfg is None else cfg
    route_id = _section_01.FAMILY_ID["route"]
    by_player = {}
    for r in reps:
        by_player.setdefault(r["nfl_id"], []).append(r)
    ids = [pid for pid in targets
           if sum(r["family_id"] == route_id for r in by_player.get(pid, [])) >= cfg["min_reps_per_player"]]
    return ids, by_player

player_ids, reps_by_player = make_player_table(reps, targets, results)
Y  = np.stack([targets[p][0] for p in player_ids])
YM = np.stack([targets[p][1] for p in player_ids])
W  = np.stack([targets[p][2] for p in player_ids])
draft_year = results.set_index("nfl_id").loc[player_ids, "draft_year"].to_numpy()   # VERIFY

def get_splits(n, mode="kfold", seed=SEED):
    """Yield (train_idx, test_idx). mode: 'kfold' or 'draft_class'."""
    if mode == "kfold":
        yield from KFold(CFG["n_folds"], shuffle=True, random_state=seed).split(np.arange(n))
    else:
        for yr in np.unique(draft_year):
            test = np.where(draft_year == yr)[0]
            train = np.where(draft_year != yr)[0]
            yield train, test

def score(pred, Y, YM, W=None):
    """Strict OOF metrics; undefined correlations/R² are explicitly NaN."""
    if pred.shape != Y.shape or YM.shape != Y.shape:
        raise ValueError("Prediction, target and mask shapes must match")
    if not np.isfinite(pred[YM > 0]).all() or not np.isfinite(Y[YM > 0]).all():
        raise ValueError("Missing or nonfinite predictions/targets for eligible players")
    rows = []
    for k, fam in enumerate(ROUTE_FAMS):
        m = YM[:, k] > 0
        y, p = Y[m, k], pred[m, k]
        rho = (float(spearmanr(p, y).statistic) if len(y) >= 2
               and np.ptp(y) > 0 and np.ptp(p) > 0 else np.nan)
        ss_tot = ((y - y.mean()) ** 2).sum() if len(y) else 0
        rows.append(dict(family=fam, n=int(m.sum()), spearman=rho,
                         r2=float(1 - ((y - p) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan))
    return pd.DataFrame(rows)

def ridge_oof(X, Y, YM, mode="kfold", alpha=10.0):
    """Compatibility helper; fold-local imputation and scaling, strict target checks.

    For route-adjusted targets use evaluation.py, which rebuilds targets within folds.
    """
    from evaluation import ridge_predict
    oof = np.full_like(Y, np.nan)
    for tr, te in get_splits(len(X), mode):
        oof[te], _ = ridge_predict(X, tr, te, Y, YM, alpha)
    return oof

# Feature construction and model comparisons live in evaluation.py so every model
# uses identical folds, adjusted targets, and training-only preprocessing.
# Run: python code-section/run_sections.py --evaluate baselines
