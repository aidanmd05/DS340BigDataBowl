# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Dependencies are imported explicitly; run directly or with run_sections.py.

from importlib import import_module
import pandas as pd

# Numbered filenames require import_module rather than Python from-import syntax.
_section_01 = import_module('01setup')
CFG = _section_01.CFG
_section_07 = import_module('07baselines')
Y = _section_07.Y
YM = _section_07.YM
score = _section_07.score
_section_11 = import_module('11cross_validation')
run_cv = _section_11.run_cv

# %% [markdown]
# ## 12. Variations and ablations (this is your experiment section)
#
# Run each as a config change, using the same folds and seeds. Keep a results DataFrame (one row per experiment) and save it to disk after every run so a crashed Colab session does not cost you results.
#
# | Variation | What it answers |
# |---|---|
# | `pool`: mean vs max vs weighted vs attention | Does attention over reps help, or is mean pooling enough? |
# | `per_family_encoders`: False vs True | Shared LSTM vs one per drill family (bias-variance tradeoff) |
# | drill embedding on/off | Does the model need to know which drill it is reading? |
# | leave out one drill family at a time | Which families carry the signal? |
# | `use_attrs`: with / without official numbers (and with draft pick) | Does the learned part add information beyond A? |
# | `hidden`, `dropout`, `weight_decay` | Where does overfitting start? |
# | rep selection: all reps vs best rep per drill | Does the model benefit from using all reps? |
# | matched vs mismatched drill-route pairs | Football-facing hypothesis: does the WR route drill predict break-route separation better than the 40? |
# | split mode: kfold vs draft_class | Does performance survive a realistic "next class" test? |
# | LSTM vs a simple non-sequential model on summary features | Does the sequence model earn its complexity? |
#
# Always include the baselines A and B in the same table. A null result (the neural model does not beat A or B) is a legitimate finding: report it honestly and discuss why (sample size, noise in targets, selection).

# %%
EXPERIMENTS = {
    "pool_mean":       dict(pool="mean"),
    "pool_max":        dict(pool="max"),
    "pool_weighted":   dict(pool="weighted"),
    "pool_attention":  dict(pool="attention"),
    "per_family_enc":  dict(per_family_encoders=True),
    "no_attrs":        dict(use_attrs=False),
    # TODO: add drill-ablation entries (requires a way to filter reps by family before building the table)
}

all_results = []
def run_experiments(attrs, mode="kfold"):
    for name, over in EXPERIMENTS.items():
        cfg = {**CFG, **over}
        oof = run_cv(cfg, attrs, mode)
        r = score(oof, Y, YM)
        r.insert(0, "experiment", name)
        all_results.append(r)
        pd.concat(all_results).to_csv("results_so_far.csv", index=False)   # checkpoint after every run
    return pd.concat(all_results)

# results_df = run_experiments(attrs)
# summary = results_df.groupby("experiment")[["spearman", "r2"]].mean()
