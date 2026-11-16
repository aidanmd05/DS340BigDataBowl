# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Run sections in order with run_sections.py (shared notebook state).

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
def make_player_table(reps, targets, results):
    """[AI-SCAFFOLD] One entry per WR that has targets and >= min_reps_per_player reps."""
    by_player = {}
    for r in reps:
        by_player.setdefault(r["nfl_id"], []).append(r)
    ids = [pid for pid in targets
           if len(by_player.get(pid, [])) >= CFG["min_reps_per_player"]]
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
    """Spearman and R^2 per route family, using only players with a target for that family."""
    rows = []
    for k, fam in enumerate(ROUTE_FAMS):
        m = YM[:, k] > 0
        if m.sum() < 5:
            continue
        rho = spearmanr(pred[m, k], Y[m, k]).correlation
        ss_res = ((Y[m, k] - pred[m, k]) ** 2).sum()
        ss_tot = ((Y[m, k] - Y[m, k].mean()) ** 2).sum()
        rows.append(dict(family=fam, n=int(m.sum()), spearman=rho, r2=1 - ss_res / ss_tot))
    return pd.DataFrame(rows)

def ridge_oof(X, Y, YM, mode="kfold", alpha=10.0):
    """Out-of-fold predictions from per-family ridge. X: (n_players, n_features)."""
    oof = np.full_like(Y, np.nan)
    for tr, te in get_splits(len(X), mode):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8        # scale with TRAIN stats only
        for k in range(Y.shape[1]):
            trk_ = tr[YM[tr, k] > 0]
            if len(trk_) < 10:
                continue
            m = Ridge(alpha=alpha).fit((X[trk_] - mu) / sd, Y[trk_, k])
            oof[te, k] = m.predict((X[te] - mu) / sd)
    return oof

# --- Feature set A ---------------------------------------------------------------------------
# TODO: choose official columns from combine_results (VERIFY names) and build X_A, shape (n_players, n_features).
# Handle opt-outs / missing values (e.g. impute with TRAIN medians inside each fold, plus a missing flag).
# X_A = results.set_index("nfl_id").loc[player_ids, [...]].to_numpy()

# --- Feature set B ---------------------------------------------------------------------------
# TODO: per rep, compute the kinematic summaries, then take the per-player mean (or best rep).
# X_B = ...

# --- Run and record ---------------------------------------------------------------------------
# resA = score(ridge_oof(X_A, Y, YM), Y, YM); print(resA)
# resB = score(ridge_oof(X_B, Y, YM), Y, YM); print(resB)
# TODO: also run A+draft pick, and the "draft pick only" baseline.
