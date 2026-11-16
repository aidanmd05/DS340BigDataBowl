# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Run sections in order with run_sections.py (shared notebook state).

# %% [markdown]
# ## 11. Grouped cross-validation for the neural model
#
# For every outer fold:
# 1. Split the training players into **inner train / inner validation** (about 85/15) for early stopping.
# 2. Fit scalers on inner-train players only.
# 3. Train `n_seeds` models with different seeds and **average their predictions** (small-data networks are high variance; report the spread across seeds too).
# 4. Store out-of-fold predictions for the test players.
#
# Then score the OOF matrix with the same `score()` used for the baselines. Leakage checklist: scalers, drill/route mappings that depend on data, imputation, and early stopping must never see test-fold players.

# %%
def run_cv(cfg, attrs, mode="kfold"):
    """[AI-SCAFFOLD] Returns an OOF prediction matrix (n_players, n_route_families)."""
    n = len(player_ids)
    oof = np.full_like(Y, np.nan)
    n_channels = reps[0]["x"].shape[1]
    for fold, (tr, te) in enumerate(get_splits(n, mode)):
        rng = np.random.RandomState(SEED + fold)
        perm = rng.permutation(tr)
        n_val = max(1, int(0.15 * len(perm)))
        inner_val, inner_tr = perm[:n_val], perm[n_val:]

        scalers = fit_scalers([player_ids[i] for i in inner_tr], reps_by_player, attrs[inner_tr])
        mk = lambda idx: DataLoader(
            PlayerDataset(idx, player_ids, reps_by_player, attrs, Y, YM, W, scalers),
            batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)
        dl_tr = DataLoader(PlayerDataset(inner_tr, player_ids, reps_by_player, attrs, Y, YM, W, scalers),
                           batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
        dl_val, dl_te = mk(inner_val), mk(te)

        preds = []
        for s in range(cfg["n_seeds"]):
            set_seed(SEED + 100 * fold + s)
            model = PlayerModel(n_channels, len(DRILL_IDS), attrs.shape[1], Y.shape[1], cfg).to(DEVICE)
            fit(model, dl_tr, dl_val, cfg)
            preds.append(predict(model, dl_te))
        oof[te] = np.mean(preds, axis=0)
    return oof

# TODO: build `attrs` (n_players, n_attrs) = the same official numbers used in feature set A.
# attrs = X_A
# oof_C = run_cv(CFG, attrs)
# resC = score(oof_C, Y, YM); print(resC)
# TODO: put resA, resB, resC side by side (one table, with bootstrap CIs over players).
