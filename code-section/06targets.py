# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Run sections in order with run_sections.py (shared notebook state).

# %% [markdown]
# ## 6. Targets from `player_play.csv`
#
# Goal: one number per (player, route family): average **separation at pass forward** on routes of that family.
#
# Design decisions to make and write down:
# - **Route families.** Group `route_ran` values into a few families so each has enough routes per player. The page lists GO, OUT, SLANT, CROSS, POST, CORNER, HITCH, SCREEN, FLAT, WHEEL; there may be others, so print the unmapped ones.
# - **Noise control.** A player's average over 10 routes is very noisy. Use a minimum route count (`min_routes_per_family`), weight the loss by route count, and optionally **shrink** each player's mean toward the family mean (empirical Bayes).
# - **Context adjustment (optional but valuable).** Separation depends on coverage (man/zone) and cushion at the snap. Fit a simple regression of separation on route type + coverage + cushion and use the *residual*; otherwise a WR who mostly faces zone looks "better".
# - **Mask.** If a player has no qualifying routes in a family, that target is missing: store `ym = 0` so the loss ignores it.
# - Only use **regular-season** rows if the table includes preseason ids (VERIFY how to filter).

# %%
# ROUTE_FAMILY, ROUTE_FAMS and SEP_COL are defined once in section 1.

def build_targets(play_df, wr_ids):
    """[AI-SCAFFOLD] Returns dict nfl_id -> (y, ym, w), each of length K = len(ROUTE_FAMS)."""
    p = play_df[play_df["nfl_id"].isin(wr_ids)].dropna(subset=["route_ran", SEP_COL]).copy()
    p["fam"] = p["route_ran"].map(ROUTE_FAMILY)
    unmapped = p.loc[p["fam"].isna(), "route_ran"].unique()
    if len(unmapped):
        print("Routes left out of every family (expected: FLAT, SCREEN):", list(unmapped))
    p = p.dropna(subset=["fam"])
    # TODO (optional): replace SEP_COL by a residual after adjusting for coverage and cushion.
    out = {}
    for pid, g in p.groupby("nfl_id"):
        y = np.zeros(len(ROUTE_FAMS), dtype=np.float32)
        ym = np.zeros(len(ROUTE_FAMS), dtype=np.float32)
        w = np.zeros(len(ROUTE_FAMS), dtype=np.float32)
        for k, fam in enumerate(ROUTE_FAMS):
            gg = g[g["fam"] == fam]
            if len(gg) >= CFG["min_routes_per_family"]:
                y[k], ym[k], w[k] = gg[SEP_COL].mean(), 1.0, math.sqrt(len(gg))   # TODO: shrinkage
        if ym.sum() > 0:
            out[pid] = (y, ym, w)
    return out

# wr_ids comes from section 2
targets = build_targets(play, wr_ids)
print(len(targets), "WRs with at least one qualifying route family")
