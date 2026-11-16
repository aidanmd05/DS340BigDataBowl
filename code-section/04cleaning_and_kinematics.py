# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Dependencies are imported explicitly; run directly or with run_sections.py.

from importlib import import_module
from pipeline_display import display
import numpy as np

# Numbered filenames require import_module rather than Python from-import syntax.
_section_01 = import_module('01setup')
EXCLUDE_BASE = _section_01.EXCLUDE_BASE
FAMILY_OF_DRILL_TYPE = _section_01.FAMILY_OF_DRILL_TYPE
_section_03 = import_module('03data_audit')
KEYS = _section_03.KEYS
parse_drill = _section_03.parse_drill
per = _section_03.per
trk_player = _section_03.trk_player
usable = _section_03.usable
usable_table = _section_03.usable_table

# %% [markdown]
# ## 4. Cleaning and kinematics
#
# What the checks showed, and what we do about it:
# - **`a` is unsigned and is the total acceleration magnitude.** It correlates about 0.98 with `sqrt(acc_tan**2 + acc_cent**2)` (along-path plus sideways), and only about 0.55 with the along-path part alone. So it carries no information the two components do not. We build the **signed** components: `acc_tan` (accelerating vs braking) and `acc_cent` (cutting left vs right).
# - **`dir` is an angle that wraps at 360.** Never feed raw degrees. We use the wrap-safe change in heading (`turn`, deg/s), and `sin`/`cos` for the "raw" ablation.
# - **`dis` is not used** (it reportedly under-counts).
# - **Absolute `x`, `y` are not used** (they encode where a drill was run, not how the player moved).
# - **Heading is unreliable at low speed**, so `turn` and `acc_cent` are zeroed below 1 yd/s.
# - **Attempt QC:** drop attempts that are too short, have time gaps, impossible speeds or NaNs. Look at the distributions first, then set the thresholds.
# - **Name variants and sides:** merge to a base drill name and mirror left-side reps so a left cut and a right cut look alike.

# %%
# ---- 4.1 Kinematic channels ----
def add_kinematics(df):
    """Adds speed, acc_tan (signed, along the path), turn (deg/s), acc_cent (signed, sideways), dir_sin, dir_cos."""
    d = df.sort_values(KEYS + ["time"]).copy()
    g = d.groupby(KEYS)
    d["dt_raw"] = g["time"].diff().dt.total_seconds()                    # NaN on each attempt's first frame
    d["dt"] = d["dt_raw"].fillna(0.1).clip(lower=1e-3)
    d["speed"] = d["s"]                                                  # find_segment() in section 5 expects "speed"
    d["s_sm"] = g["s"].transform(lambda x: x.rolling(3, center=True, min_periods=1).mean())
    g = d.groupby(KEYS)                                                  # rebuild so it sees the new column
    d["acc_tan"] = g["s_sm"].diff() / d["dt"]
    d["turn"] = ((g["dir"].diff() + 180) % 360 - 180) / d["dt"]          # deg/s, wrap-safe
    # First frame of each attempt has no previous frame (clips often start in motion): copy the next frame's value.
    g = d.groupby(KEYS)
    d[["acc_tan", "turn"]] = g[["acc_tan", "turn"]].bfill().fillna(0.0)
    d.loc[d["s"] < 1.0, "turn"] = 0.0                                    # heading is unreliable at low speed
    d["acc_cent"] = d["s"] * np.radians(d["turn"])                       # signed sideways acceleration
    rad = np.radians(d["dir"])
    d["dir_sin"], d["dir_cos"] = np.sin(rad), np.cos(rad)
    return d

trk_k = add_kinematics(trk_player)
print(trk_k[["speed", "acc_tan", "turn", "acc_cent"]].describe().round(2).to_string())

# %%
# ---- 4.2 Check (optional): does `a` equal total acceleration? ----
print("a has negatives:", bool((trk_k["a"] < 0).any()), "(False means unsigned)")
m = (trk_k["s"] >= 1.0) & trk_k["dt_raw"].notna()     # skip each attempt's first frame (no previous frame)
acc_total = np.hypot(trk_k["acc_tan"], trk_k["acc_cent"])
print("corr(|acc_tan|, a):", round(np.corrcoef(trk_k.loc[m, "acc_tan"].abs(), trk_k.loc[m, "a"])[0, 1], 3))
print("corr(acc_total, a):", round(np.corrcoef(acc_total[m], trk_k.loc[m, "a"])[0, 1], 3))
# Expect about 0.55 vs about 0.98 (what you observed). If the second number drops a lot, the new
# add_kinematics is not reproducing the sensor: check dt and the turn computation.

# %%
# ---- 4.3 Attempt QC ----
# Step 1: look at the distributions per drill type. Step 2: set the thresholds so they only catch clear glitches.
QC = dict(min_frames=15,       # at 10 Hz, 15 frames = 1.5 s: shorter than any real drill  (TODO: tune from the table)
          max_gap_s=0.5,       # larger time gap inside an attempt = dropped frames
          max_speed=12.0)      # yd/s; the fastest 40-yard dashes are about 10-11 yd/s  (TODO: tune)

def attempt_qc(d, qc=QC):
    a = (d.groupby(KEYS)
         .agg(drill_type=("drill_type", "first"), n_frames=("time", "count"), peak_speed=("s", "max"),
              max_gap=("dt_raw", "max"), n_nan=("s", lambda x: int(x.isna().sum())))
         .reset_index())
    a["bad_short"] = a["n_frames"] < qc["min_frames"]
    a["bad_gap"]   = a["max_gap"] > qc["max_gap_s"]
    a["bad_speed"] = a["peak_speed"] > qc["max_speed"]
    a["bad_nan"]   = a["n_nan"] > 0
    a["bad"] = a[["bad_short", "bad_gap", "bad_speed", "bad_nan"]].any(axis=1)
    return a

qc_tbl = attempt_qc(trk_k)
print("Distributions by drill type:")
display(qc_tbl.groupby("drill_type")[["n_frames", "peak_speed", "max_gap"]]
        .quantile([0.0, 0.01, 0.5, 0.99, 1.0]).round(2))
print("Flags:", qc_tbl[["bad_short", "bad_gap", "bad_speed", "bad_nan", "bad"]].sum().to_dict())
print("Attempts flagged by drill type:", qc_tbl.groupby("drill_type")["bad"].sum().to_dict())

bad_keys = qc_tbl.loc[qc_tbl["bad"], KEYS].assign(_bad=True)
d_clean = trk_k.merge(bad_keys, on=KEYS, how="left")
d_clean = d_clean[d_clean["_bad"].isna()].drop(columns="_bad").copy()
print("attempts kept:", d_clean.groupby(KEYS).ngroups, "of", len(qc_tbl))

# %%
# ---- 4.4 Canonical drill names, left/right mirroring, drill scope ----
_names = {n: parse_drill(n) for n in d_clean["drill_name"].unique()}
d_clean["drill_raw"]  = d_clean["drill_name"]
d_clean["drill_base"] = d_clean["drill_name"].map(lambda n: _names[n][0])
d_clean["side"]       = d_clean["drill_name"].map(lambda n: _names[n][1])

# Sign convention check: for each attempt, the direction of its hardest cut (sign of acc_cent at peak |acc_cent|).
# Only drills run on BOTH sides can verify this (POST_CORNER). Before mirroring, LEFT and RIGHT reps should have
# OPPOSITE mean signs (about -1 vs +1). After mirroring they should match.
_idx = d_clean.assign(_abs=d_clean["acc_cent"].abs()).groupby(KEYS)["_abs"].idxmax()
peak = d_clean.loc[_idx, KEYS + ["drill_base", "side", "acc_cent"]].copy()
peak["sign_before"] = np.sign(peak["acc_cent"])
peak["sign_after"] = np.where(peak["side"] == "LEFT", -peak["sign_before"], peak["sign_before"])
chk = (peak[peak["side"].notna()].groupby(["drill_base", "side"])[["sign_before", "sign_after"]]
       .mean().round(2))
display(chk)

MIRROR_LEFT = True      # set False if the check above shows LEFT and RIGHT already have matching signs
if MIRROR_LEFT:
    left = d_clean["side"] == "LEFT"
    d_clean.loc[left, ["acc_cent", "turn"]] *= -1
    # note: dir_sin / dir_cos (the "raw" channel set) are not mirrored

# Scope: drill types and drills used in v1. drill_name becomes the canonical base name,
# because section 5 builds DRILL_IDS from wr_trk["drill_name"].
in_scope = d_clean["drill_type"].isin(FAMILY_OF_DRILL_TYPE) & ~d_clean["drill_base"].isin(EXCLUDE_BASE)
wr_trk = d_clean[in_scope].copy()
wr_trk["drill_name"] = wr_trk["drill_base"]

print("\nwr_trk:", len(wr_trk), "frames |", wr_trk.groupby(KEYS).ngroups, "attempts |",
      wr_trk["nfl_id"].nunique(), "WRs")
print(wr_trk.groupby("drill_name")[KEYS[0]].nunique().to_string())

# %%
# ---- 4.5 Final check: usable WRs after cleaning (compare with 3.5) ----
att_clean = (wr_trk.groupby(KEYS)
             .agg(drill_type=("drill_type", "first"), base=("drill_base", "first"))
             .reset_index())
tab_clean, usable_clean = usable_table(att_clean, per)
print("\nWRs lost to cleaning:", sorted(set(usable.index) - set(usable_clean.index)))
