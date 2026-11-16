# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Run sections in order with run_sections.py (shared notebook state).

# %% [markdown]
# ## 5. Turn each rep into a fixed-size tensor
#
# Each rep becomes an array of shape `(T, C)`:
# 1. **Trim to the movement**: start at the first frame where speed exceeds a small threshold (the clip may start in motion, so also handle the case where it already is).
# 2. **Pick alignment events per family**, then resample the segment(s) between events to fixed length `T`:
#    - `sprint`: motion start -> end of the run. (Entrants report the first ~10 yd of the 40 are unreliable in tracking; consider ignoring official-time comparisons.)
#    - `cod` (3-cone, shuttle): motion start -> end. Optionally align on each direction change.
#    - `route`: motion start -> break point (largest heading change) -> catch/end. If ball frames exist, use the ball to find the catch.
# 3. **Channels**: speed, signed acceleration, turn rate (add more if useful, e.g. position relative to the start).
# 4. **Duration** is lost by resampling, so append it as one extra *constant* channel.
# 5. **Standardize channels using training-fold statistics only** (done in section 9, not here; fitting scalers on all reps leaks test information).
#
# Output: a list of rep dicts that every later section consumes.

# %%
def resample(arr, T):
    """[AI-SCAFFOLD] Linearly resample a (n, C) array to (T, C)."""
    n = len(arr)
    if n == 1:
        return np.repeat(arr, T, axis=0)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, T)
    return np.stack([np.interp(dst, src, arr[:, c]) for c in range(arr.shape[1])], axis=1)

def find_segment(rep_df, family):
    """TODO: return (start_idx, end_idx) of the part of the rep to keep, using family-specific events.
    Minimal version: first frame with speed > 0.5 through the last frame with speed > 0.5."""
    moving = np.where(rep_df["speed"].to_numpy() > 0.5)[0]
    if len(moving) < 5:
        return None
    return moving[0], moving[-1] + 1

# Build a stable integer id for every canonical drill name (used for the drill embedding).
DRILL_IDS = {d: i for i, d in enumerate(sorted(wr_trk["drill_name"].unique()))}

def build_reps(df, T=CFG["T"], channels=CFG["channels"]):
    """[AI-SCAFFOLD] One dict per usable rep: x (T, C+1), drill_id, family_id, nfl_id, draft_year."""
    reps = []
    for (ev, pid), rep in df.groupby(KEYS):
        dt_type = rep["drill_type"].iloc[0]
        family = FAMILY_OF_DRILL_TYPE.get(dt_type)
        if family is None:
            continue                                  # drill types we are not using
        seg = find_segment(rep, family)
        if seg is None:
            continue
        a, b = seg
        arr = rep[channels].to_numpy()[a:b]
        duration = (b - a) * 0.1
        x = resample(arr, T)
        x = np.concatenate([x, np.full((T, 1), duration)], axis=1)   # constant duration channel
        reps.append(dict(x=x.astype(np.float32), nfl_id=pid, draft_year=rep["draft_year"].iloc[0],
                         drill_id=DRILL_IDS[rep["drill_name"].iloc[0]], family_id=FAMILY_ID[family]))
    return reps

reps = build_reps(wr_trk)
print(len(reps), "reps for", len({r["nfl_id"] for r in reps}), "WRs")
# TODO: plot ~6 reps per family (speed vs resampled time) and sanity check the alignment visually.
