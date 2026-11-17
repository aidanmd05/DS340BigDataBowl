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

# 3. Data audit (findings so far, and what each one decided)

# ---- 3.0 One row per Combine drill attempt ----
KEYS = ["event_id", "nfl_id"]          # identifies one drill attempt

def parse_drill(name):
    """Return the canonical drill base and side for a raw drill name."""
    if name == "CIRCUS_ROUTE_LEFT":
        return "POST_CORNER", "LEFT"
    n = name.replace("_ROUTE", "")             # merges SPEED_OUT_ROUTE_LEFT / SPEED_OUT_LEFT
    side = "LEFT" if n.endswith("_LEFT") else "RIGHT" if n.endswith("_RIGHT") else None
    base = n.rsplit("_", 1)[0] if side else n
    return base, side

trk_player = wr_only_trk[wr_only_trk["entity_type"] == "PLAYER"].copy()

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

# ---- 3.1 Players and reps per base drill x draft year (name variants merged) ----
if __name__ == "__main__":
    print("WRs per drill")
    display(wr_attempts.pivot_table(index=["drill_type", "base"], columns="draft_year",
                                    values="nfl_id", aggfunc="nunique", fill_value=0))
    print("Reps per drill")
    display(wr_attempts.pivot_table(index=["drill_type", "base"], columns="draft_year",
                                    values="event_id", aggfunc="nunique", fill_value=0))


# ---- 3.2 Route Reps per WR ----
route_reps_per_wr = (wr_attempts[(wr_attempts["drill_type"] == "SKILL_DRILLS_WR") & wr_attempts["in_scope"]]
                     .groupby("nfl_id").size()
                     .reindex(sorted(wr_ids), fill_value=0))     # WRs with no route reps show as 0
if __name__ == "__main__":
    print("route-drill reps per WR:")
    print(route_reps_per_wr.describe().round(1).to_string())
    print("WRs with fewer than MIN_ROUTE_REPS route reps:", int((route_reps_per_wr < MIN_ROUTE_REPS).sum()))
    print("WRs with fewer than 6 route reps:           ", int((route_reps_per_wr < 6).sum()))


# ---- 3.3 Separation targets per WR per route family ----
routes = wr_only_play[wr_only_play["route_ran"].notna()].copy()
routes["has_sep"] = routes[SEP_COL].notna()
routes["fam"] = routes["route_ran"].map(ROUTE_FAMILY)

if __name__ == "__main__":
    print("WR route rows:", len(routes), "| non-null separation rate:", round(routes["has_sep"].mean(), 3))
    print("route_ran values left out of every family:", sorted(routes.loc[routes["fam"].isna(), "route_ran"].unique()))

# Separation with route-type differences removed (a GO is not a POST), so family means compare players, not route mixes.
have = routes[routes["has_sep"] & routes["fam"].notna()].copy()
have["sep_adj"] = have[SEP_COL] - have.groupby("route_ran")[SEP_COL].transform("mean")

per = (have.groupby(["nfl_id", "fam"]).size().unstack(fill_value=0)
       .reindex(index=sorted(wr_ids), columns=ROUTE_FAMS, fill_value=0))

if __name__ == "__main__":
    print("\nWRs with at least N separation values, by family:")
    print(pd.DataFrame({f"N>={t}": (per >= t).sum() for t in [10, 15, 20, 25]}).to_string())
    print("\nWRs by number of families meeting MIN_ROUTES:")
    print((per >= MIN_ROUTES).sum(axis=1).value_counts().sort_index().to_string())


# ---- 3.4 Target inclusion rule for v1 ----
# To keep the first version simple and comparable across players, require at least
# MIN_ROUTES routes with a separation value in each route family.
qualified = (per >= MIN_ROUTES).sum()
if __name__ == "__main__":
    print(f"V1 rule: require at least {MIN_ROUTES} routes with separation per family.")
    print("WRs meeting the rule by family:")
    print(qualified.to_string())


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
    if __name__ == "__main__":
        print(f"MIN_ROUTES={min_routes}, MIN_ROUTE_REPS={min_reps}")
        print("WRs considered:           ", len(tab))
        print("with enough route reps:   ", int(tab["has_reps"].sum()))
        print("with >= 1 family target:  ", int((tab["n_fams"] >= 1).sum()))
        print("usable (both):            ", len(usable))
        print("usable per family:        ", {f: int(((tab[f] >= min_routes) & tab["has_reps"]).sum()) for f in fams})
        print("usable by draft class:    ", usable["draft_year"].value_counts().sort_index().to_dict())
    return tab, usable

tab, usable = usable_table(wr_attempts, per)
if __name__ == "__main__":
    print("\nDropped WRs and why:")
    display(tab[~tab.index.isin(usable.index)][["draft_year", "route_reps", "forty_reps"] + ROUTE_FAMS])


# ---- 3.6 Official Combine numbers: coverage ----
official_cols = ["combine_height", "combine_weight", "hand_size", "arm_length", "wing_span",
                 "ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
official_cols = [c for c in official_cols if c in wr_only_results.columns]
if __name__ == "__main__":
    print(f"Non-null counts over {len(wr_only_results)} WR rows:")
    print(wr_only_results[official_cols].notna().sum().to_string())
# Leaves out NGS (Next Generation Stats) values since these values undergo a ML process themselves 
# and may be correlated with combine values already fed in (avoid data leakage)