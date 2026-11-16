# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Dependencies are imported explicitly; run directly or with run_sections.py.

from importlib import import_module
from pipeline_display import display
import numpy as np
import pandas as pd

# Numbered filenames require import_module rather than Python from-import syntax.
_section_01 = import_module('01setup')
EXCLUDE_BASE = _section_01.EXCLUDE_BASE
FAMILY_OF_DRILL_TYPE = _section_01.FAMILY_OF_DRILL_TYPE
MIN_ROUTES = _section_01.MIN_ROUTES
MIN_ROUTE_REPS = _section_01.MIN_ROUTE_REPS
ROUTE_FAMILY = _section_01.ROUTE_FAMILY
ROUTE_FAMS = _section_01.ROUTE_FAMS
SEP_COL = _section_01.SEP_COL
_section_02 = import_module('02load_data')
wr_ids = _section_02.wr_ids
wr_only_play = _section_02.wr_only_play
wr_only_results = _section_02.wr_only_results
wr_only_trk = _section_02.wr_only_trk

# %% [markdown]
# ## 3. Data audit (findings so far, and what each one decided)
#
# | Question | What the audit showed | Decision |
# |---|---|---|
# | Route-drill coverage | about 14 skill-drill reps per WR (range 8-18), roughly one rep per drill | One shared LSTM over the route drills, with a drill embedding |
# | Name drift | `SPEED_OUT_LEFT` and `SPEED_OUT_ROUTE_LEFT` are one drill; `POST_CORNER` ran on both sides | Merge to a base name, mirror left-side reps (section 4) |
# | Class confounds | `CIRCUS` exists only for 2025 | Exclude from v1 |
# | 3-cone / shuttle | about 0.3 tracking reps per WR; official times for only about a third of WRs | No tracking encoder; official times in feature set A with missing flags |
# | Ball frames | about a third of attempts, absent for several drills | Not used in v1 |
# | Separation coverage | recorded on about 88% of WR routes, steady across route types | Usable as an "ability to get open" target, not a targeted-only stat |
# | Target noise | split-half reliability plateaus near 20 routes per family | `MIN_ROUTES = 20` |
#
# Cells 3.1 to 3.6 reproduce these checks with the final definitions, so re-running them after any config change updates the counts. Decision gates (adjust with the TAs):
# - Fewer than about 60 usable WRs in a family: merge it into another family or drop it from the multi-output head.
# - Fewer than about 20 WRs in a draft class: use pooled leave-one-class-out scoring (and K-fold as the main evaluation), not a single-class holdout.

# %%
# ---- 3.0 One row per Combine drill attempt ----
KEYS = ["event_id", "nfl_id"]          # identifies one drill attempt

def parse_drill(name):
    """'POST_CORNER_ROUTE_LEFT' -> ('POST_CORNER', 'LEFT'); 'GAUNTLET_DRILL' -> ('GAUNTLET_DRILL', None)."""
    n = name.replace("_ROUTE", "")             # merges SPEED_OUT_ROUTE_LEFT / SPEED_OUT_LEFT
    side = "LEFT" if n.endswith("_LEFT") else "RIGHT" if n.endswith("_RIGHT") else None
    base = n.rsplit("_", 1)[0] if side else n
    return base, side

trk_player = wr_only_trk[wr_only_trk["entity_type"] == "PLAYER"].copy()
trk_ball   = wr_only_trk[wr_only_trk["entity_type"] == "BALL"].copy()

wr_attempts = (trk_player.groupby(KEYS)
               .agg(drill_type=("drill_type", "first"), drill_name=("drill_name", "first"),
                    draft_year=("draft_year", "first"), attempt=("attempt", "first"),
                    n_frames=("time", "count"))
               .reset_index())

_parsed = {n: parse_drill(n) for n in wr_attempts["drill_name"].unique()}
wr_attempts["base"] = wr_attempts["drill_name"].map(lambda n: _parsed[n][0])
wr_attempts["side"] = wr_attempts["drill_name"].map(lambda n: _parsed[n][1])
wr_attempts["in_scope"] = (wr_attempts["drill_type"].isin(FAMILY_OF_DRILL_TYPE)
                           & ~wr_attempts["base"].isin(EXCLUDE_BASE))

# Does the attempt have ball frames? (ball rows carry the same nfl_id, so join on KEYS)
_ball = trk_ball[KEYS].drop_duplicates().assign(has_ball=True)
wr_attempts = wr_attempts.merge(_ball, on=KEYS, how="left")
wr_attempts["has_ball"] = wr_attempts["has_ball"].eq(True)

print("attempts:", len(wr_attempts), "| in scope:", int(wr_attempts["in_scope"].sum()))

# %%
# ---- 3.1 Players and reps per base drill x draft year (name variants merged) ----
print("WRs per drill")
display(wr_attempts.pivot_table(index=["drill_type", "base"], columns="draft_year",
                                values="nfl_id", aggfunc="nunique", fill_value=0))
print("Reps per drill")
display(wr_attempts.pivot_table(index=["drill_type", "base"], columns="draft_year",
                                values="event_id", aggfunc="nunique", fill_value=0))
# Check: any in-scope drill with a 0 in some draft year is confounded with draft class.

# %%
# ---- 3.2 Reps per WR, and ball-frame coverage ----
reps_per_wr = wr_attempts.groupby(["nfl_id", "drill_type"]).size().unstack(fill_value=0)
display(reps_per_wr.describe())

scope_reps = wr_attempts[wr_attempts["in_scope"]].groupby("nfl_id").size()
print("in-scope reps per WR:"); print(scope_reps.describe().round(1).to_string())
print("WRs with fewer than MIN_ROUTE_REPS in-scope reps:", int((scope_reps < MIN_ROUTE_REPS).sum()))

print("\nShare of attempts with ball frames, by base drill (v1 does not use them):")
print(wr_attempts.groupby("base")["has_ball"].mean().round(2).to_string())

# %%
# ---- 3.3 Separation targets per WR per route family ----
routes = wr_only_play[wr_only_play["route_ran"].notna()].copy()
routes["has_sep"] = routes[SEP_COL].notna()
routes["fam"] = routes["route_ran"].map(ROUTE_FAMILY)

print("WR route rows:", len(routes), "| non-null separation rate:", round(routes["has_sep"].mean(), 3))
print("route_ran values left out of every family:", sorted(routes.loc[routes["fam"].isna(), "route_ran"].unique()))

# Separation with route-type differences removed (a GO is not a POST), so family means compare players, not route mixes.
have = routes[routes["has_sep"] & routes["fam"].notna()].copy()
have["sep_adj"] = have[SEP_COL] - have.groupby("route_ran")[SEP_COL].transform("mean")

per = (have.groupby(["nfl_id", "fam"]).size().unstack(fill_value=0)
       .reindex(index=sorted(wr_ids), columns=ROUTE_FAMS, fill_value=0))

print("\nWRs with at least N separation values, by family:")
print(pd.DataFrame({f"N>={t}": (per >= t).sum() for t in [10, 15, 20, 25]}).to_string())
print("\nWRs by number of families meeting MIN_ROUTES:")
print((per >= MIN_ROUTES).sum(axis=1).value_counts().sort_index().to_string())

# %%
# ---- 3.4 Target reliability by threshold (split-half, Spearman-Brown corrected) ----
# Reliability about 0.6 means about 60% of between-player differences in the average are stable.
# A rough ceiling on any model's correlation with the target is sqrt(reliability).
def split_half(df, fam, thr=15, n_rep=200, seed=0):
    rng = np.random.default_rng(seed)
    g = df[df["fam"] == fam].groupby("nfl_id")["sep_adj"].apply(lambda s: s.to_numpy())
    g = g[g.apply(len) >= thr]
    if len(g) < 10:
        return dict(family=fam, n_players=len(g), half_r=np.nan, reliability=np.nan)
    rs = []
    for _ in range(n_rep):
        a, b = [], []
        for vals in g:
            p = rng.permutation(len(vals)); h = len(vals) // 2
            a.append(vals[p[:h]].mean()); b.append(vals[p[h:2 * h]].mean())
        rs.append(np.corrcoef(a, b)[0, 1])
    r = float(np.mean(rs))
    return dict(family=fam, n_players=len(g), half_r=round(r, 3), reliability=round(2 * r / (1 + r), 3))

rel = pd.DataFrame([{"thr": t, **split_half(have, f, t)} for t in [5, 10, 15, 20, 25, 30, 40] for f in ROUTE_FAMS])
print("Reliability:");       display(rel.pivot(index="thr", columns="family", values="reliability"))
print("Players retained:");  display(rel.pivot(index="thr", columns="family", values="n_players"))

# %%
# ---- 3.5 Intersection: WRs with usable Combine reps AND usable targets ----
def usable_table(att, per, min_routes=None, min_reps=None):
    """att: one row per attempt with nfl_id, drill_type, base. Returns (tab, usable)."""
    min_routes = MIN_ROUTES if min_routes is None else min_routes
    min_reps = MIN_ROUTE_REPS if min_reps is None else min_reps
    route_reps = (att[(att["drill_type"] == "SKILL_DRILLS_WR") & ~att["base"].isin(EXCLUDE_BASE)]
                  .groupby("nfl_id").size().rename("route_reps"))
    forty_reps = att[att["drill_type"] == "FORTY_YARD_DASH"].groupby("nfl_id").size().rename("forty_reps")
    fams = list(per.columns)
    tab = (pd.DataFrame(index=sorted(wr_ids))
           .join(wr_only_results.set_index("nfl_id")[["draft_year"]])
           .join(route_reps).join(forty_reps).join(per))
    fill_cols = ["route_reps", "forty_reps"] + fams
    tab[fill_cols] = tab[fill_cols].fillna(0)
    tab["has_reps"] = tab["route_reps"] >= min_reps
    tab["n_fams"] = (tab[fams] >= min_routes).sum(axis=1)
    usable = tab[tab["has_reps"] & (tab["n_fams"] >= 1)]
    print(f"MIN_ROUTES={min_routes}, MIN_ROUTE_REPS={min_reps}")
    print("WRs considered:           ", len(tab))
    print("with enough route reps:   ", int(tab["has_reps"].sum()))
    print("with >= 1 family target:  ", int((tab["n_fams"] >= 1).sum()))
    print("usable (both):            ", len(usable))
    print("usable per family:        ", {f: int(((tab[f] >= min_routes) & tab["has_reps"]).sum()) for f in fams})
    print("usable by draft class:    ", usable["draft_year"].value_counts().sort_index().to_dict())
    return tab, usable

tab, usable = usable_table(wr_attempts, per)
print("\nDropped WRs and why:")
display(tab[~tab.index.isin(usable.index)][["draft_year", "route_reps", "forty_reps"] + ROUTE_FAMS])
# Sensitivity: usable_table(wr_attempts, per, min_routes=15) and (..., min_routes=25, min_reps=6)

# %%
# ---- 3.6 Official Combine numbers: coverage (this is feature set A) ----
official_cols = ["combine_height", "combine_weight", "hand_size", "arm_length", "wing_span",
                 "ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
official_cols = [c for c in official_cols if c in wr_only_results.columns]
print(f"Non-null counts over {len(wr_only_results)} WR rows:")
print(wr_only_results[official_cols].notna().sum().to_string())
# ngs_*_score columns: check how they are defined before using them. ngs_athleticism_score may be derived
# from the same Combine tracking you are modeling, which would leak into the "official numbers" baseline.
