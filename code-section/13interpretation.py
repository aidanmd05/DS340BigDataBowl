# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Dependencies are imported explicitly; run directly or with run_sections.py.

# %% [markdown]
# ## 13. Interpretation (what makes this useful to a coach, not just a number)
#
# - **Pooling weights.** For `weighted` / `attention` models, pull `model.pool.last_weights` on test players and aggregate by drill: which drills does the model lean on for each route family? Treat this as a hint, not proof. Confirm with the drill-ablation experiment.
# - **Permutation importance.** Shuffle one drill family's reps across players in the test fold and measure the drop in Spearman.
# - **Residual analysis.** Which WRs does the model under- and over-predict? Is it related to draft slot, route volume, or opt-outs?
# - **Compare against A.** Show a scatter of A-only predictions vs C predictions for the same players. Where do they disagree, and who is right?
# - **Figures to keep (they feed both your paper and the Big Data Bowl writeup):** (1) per-family Spearman for A / B / C with CIs, (2) pooling variants, (3) drill-ablation drops, (4) example rep traces with the attention weights, (5) one "scouting card" style summary.

# %%
# TODO: after training a final model on all training players, run the test loader and collect pool weights.
# Example sketch:
#   model.eval(); _ = model(to_dev(batch)); w = model.pool.last_weights   # (B, R) or (B, R, R)
#   Map each rep back to its drill_id, group by drill name, and average the weight per (route family, drill).
