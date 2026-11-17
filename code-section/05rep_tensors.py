# [AI-SCAFFOLD] Pipeline fixes implemented with AI assistance; include in course disclosure.
# Extracted from legacy-code/bdb_wr_route_scaffold_refactored.ipynb.
# Dependencies are imported explicitly; run directly or with run_sections.py.

from importlib import import_module
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Numbered filenames require import_module rather than Python from-import syntax.
_section_01 = import_module('01setup')
CFG = _section_01.CFG
FAMILY_ID = _section_01.FAMILY_ID
FAMILY_OF_DRILL_TYPE = _section_01.FAMILY_OF_DRILL_TYPE
SEED = _section_01.SEED
_section_03 = import_module('03data_audit')
KEYS = _section_03.KEYS
_section_04 = import_module('04cleaning_and_kinematics')
wr_trk = _section_04.wr_trk

# %% [markdown]
# ## 5. Turn each rep into a fixed-size tensor
#
# Each rep becomes an array of shape `(T, C)`:
# 1. **Trim to the movement**: start at the first frame where speed exceeds a small threshold (the clip may start in motion, so also handle the case where it already is).
# 2. Sprint: stop at post-peak rollout (below half peak speed); route: retain the moving interval.
#    Break/catch alignment is deferred until a detector is checked against labeled examples.
# 3. Interpolate selected channels against actual timestamps to fixed length T.
# 4. Append actual elapsed duration as an extra constant channel.
# 5. Standardize with training-fold statistics in the evaluation/dataset code.
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
    """Trim movement, excluding sprint rollout after peak speed falls below half.

    Route clips retain the complete moving interval: break alignment remains opt-in
    until a heading-based detector has been checked against example traces.
    """
    speed = rep_df["speed"].to_numpy()
    moving = np.flatnonzero(speed > 0.5)
    if len(moving) < 5:
        return None
    a, b = int(moving[0]), int(moving[-1]) + 1
    if family == "sprint":
        peak = a + int(np.argmax(speed[a:b]))
        rollout = np.flatnonzero(speed[peak:b] < 0.5 * speed[peak])
        if len(rollout):
            b = peak + int(rollout[0])
    return (a, b) if b - a >= 5 else None

# Build a stable integer id for every canonical drill name (used for the drill embedding).
DRILL_IDS = {d: i for i, d in enumerate(sorted(wr_trk["drill_name"].unique()))}

def build_reps(df, T=None, channels=None):
    """[AI-SCAFFOLD] One dict per usable rep: x (T, C+1), drill_id, family_id, nfl_id, draft_year."""
    T = CFG["T"] if T is None else T
    channels = CFG["channels"] if channels is None else channels
    if T < 2:
        raise ValueError("T must be at least 2")
    reps = []
    for (ev, pid), rep in df.groupby(KEYS):
        rep = rep.sort_values("time")
        if rep["time"].isna().any() or rep["time"].duplicated().any():
            raise ValueError(f"Invalid timestamps in attempt {ev}, player {pid}")
        dt_type = rep["drill_type"].iloc[0]
        family = FAMILY_OF_DRILL_TYPE.get(dt_type)
        if family is None:
            continue                                  # drill types we are not using
        seg = find_segment(rep, family)
        if seg is None:
            continue
        a, b = seg
        arr = rep[channels].to_numpy()[a:b]
        times = (rep["time"].iloc[a:b] - rep["time"].iloc[a]).dt.total_seconds().to_numpy()
        duration = float(times[-1])
        if duration <= 0 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid segment in attempt {ev}, player {pid}")
        x = np.stack([np.interp(np.linspace(0, duration, T), times, arr[:, c])
                      for c in range(arr.shape[1])], axis=1)
        x = np.concatenate([x, np.full((T, 1), duration)], axis=1)   # constant duration channel
        reps.append(dict(x=x.astype(np.float32), event_id=ev, nfl_id=pid, draft_year=rep["draft_year"].iloc[0],
                         drill_id=DRILL_IDS[rep["drill_name"].iloc[0]], family_id=FAMILY_ID[family]))
    return reps

reps = build_reps(wr_trk)
print(len(reps), "reps for", len({r["nfl_id"] for r in reps}), "WRs")


def plot_reps_per_family(rep_list, channels=None, n_per_family=6, seed=SEED):
    """Plot sampled speed traces by family; return the figure for saving or display.

    Progress is normalized within each trimmed rep, not elapsed seconds.
    These traces are resampled but have not been standardized for training.
    """
    if n_per_family < 1:
        raise ValueError("n_per_family must be at least 1")
    channels = CFG["channels"] if channels is None else channels
    speed_channel = "speed" if "speed" in channels else "s"
    if speed_channel not in channels:
        raise ValueError("The channel set must include speed or s to plot speed.")
    speed_idx = channels.index(speed_channel)
    families = [(name, fid) for name, fid in sorted(FAMILY_ID.items())
                if any(r["family_id"] == fid for r in rep_list)]
    if not families:
        raise ValueError("No reps available to plot.")

    rng = np.random.default_rng(seed)
    drill_names = {value: name for name, value in DRILL_IDS.items()}
    fig, axes = plt.subplots(len(families), 1, figsize=(11, 4 * len(families)),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (family, fid) in zip(axes[:, 0], families):
        candidates = [r for r in rep_list if r["family_id"] == fid]
        chosen = rng.choice(len(candidates), size=min(n_per_family, len(candidates)),
                            replace=False)
        for idx in chosen:
            rep = candidates[idx]
            progress = np.linspace(0, 100, len(rep["x"]))
            drill = drill_names.get(rep["drill_id"], str(rep["drill_id"]))
            duration = rep["x"][0, -1]
            ax.plot(progress, rep["x"][:, speed_idx], alpha=0.8,
                    label=f'{rep["nfl_id"]} | {drill} | {duration:.1f}s')
        ax.set_title(f"{family.title()}: {len(chosen)} examples / {len(candidates)} total reps")
        ax.set_ylabel("Speed (yd/s)")
        ax.grid(alpha=0.25)
        ax.legend(title="Player ID | Drill | Duration", fontsize=8)
    axes[-1, 0].set_xlabel("Progress through trimmed rep (%)")
    fig.suptitle("Resampled Combine reps by drill family")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_reps_per_family(reps)
    output = Path(__file__).resolve().parents[1] / "outputs" / "rep_traces_by_family.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print("Saved", output)
    plt.show()
