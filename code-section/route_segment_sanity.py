"""Measure how much route tracking the v1 segment rule removes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "wr_only_trk.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "route_segment_sanity.png"
KEYS = ["event_id", "nfl_id"]
MIN_FRAMES = 15
MAX_GAP_S = 0.5
MAX_SPEED = 12.0
SPEED_THRESHOLD = 0.5
EXCLUDE_BASE = {"GAUNTLET_DRILL", "OVER_SHOULDER_ADJUST"}


def parse_drill(name):
    """Return the canonical drill base and side used by the pipeline."""
    if name == "CIRCUS_ROUTE_LEFT":
        return "POST_CORNER", "LEFT"
    normalized = name.replace("_ROUTE", "")
    side = "LEFT" if normalized.endswith("_LEFT") else (
        "RIGHT" if normalized.endswith("_RIGHT") else None
    )
    base = normalized.rsplit("_", 1)[0] if side else normalized
    return base, side


def load_route_tracking():
    """Load player route attempts and apply the section 4 QC rules."""
    tracking = pd.read_csv(DATA_PATH, parse_dates=["time"])
    tracking = tracking[tracking["entity_type"] == "PLAYER"].copy()
    tracking = tracking[tracking["drill_type"] == "SKILL_DRILLS_WR"].copy()
    tracking["base"] = tracking["drill_name"].map(lambda name: parse_drill(name)[0])
    tracking = tracking[~tracking["base"].isin(EXCLUDE_BASE)].copy()
    tracking = tracking.sort_values(KEYS + ["time"])
    grouped = tracking.groupby(KEYS)
    tracking["dt_raw"] = grouped["time"].diff().dt.total_seconds()

    qc = (tracking.groupby(KEYS)
          .agg(drill_name=("drill_name", "first"),
               n_frames=("time", "count"),
               peak_speed=("s", "max"),
               max_gap=("dt_raw", "max"),
               n_nan=("s", lambda values: int(values.isna().sum())))
          .reset_index())
    qc["bad"] = (
        (qc["n_frames"] < MIN_FRAMES)
        | (qc["max_gap"] > MAX_GAP_S)
        | (qc["peak_speed"] > MAX_SPEED)
        | (qc["n_nan"] > 0)
    )
    good_keys = qc.loc[~qc["bad"], KEYS]
    return tracking.merge(good_keys, on=KEYS, how="inner"), qc


def measure_segments(tracking):
    """Measure the current first/last speed-threshold window per attempt."""
    rows = []
    for (event_id, nfl_id), attempt in tracking.groupby(KEYS):
        speed = attempt["s"].to_numpy()
        moving = np.flatnonzero(speed > SPEED_THRESHOLD)
        if len(moving) < 5:
            continue
        start, end = int(moving[0]), int(moving[-1] + 1)
        n_frames = len(attempt)
        drill_name = attempt["drill_name"].iloc[0]
        base, side = parse_drill(drill_name)
        rows.append({
            "event_id": event_id,
            "nfl_id": nfl_id,
            "drill_name": drill_name,
            "base": base,
            "side": side,
            "n_frames": n_frames,
            "kept_frames": end - start,
            "leading_removed": start,
            "trailing_removed": n_frames - end,
            "removed_frames": start + (n_frames - end),
            "removed_fraction": (start + (n_frames - end)) / n_frames,
        })
    return pd.DataFrame(rows)


def print_summary(measurements, qc):
    """Print overall and drill-level trimming summaries."""
    print(f"QC attempts: {len(qc)} total, {int((~qc['bad']).sum())} retained")
    print(f"Eligible route attempts: {len(measurements)}")
    print("\nOverall removed fraction:")
    print(measurements["removed_fraction"].describe(percentiles=[.5, .75, .9, .95]).round(3))

    summary = (measurements.groupby("base")
               .agg(attempts=("event_id", "size"),
                    median_removed=("removed_fraction", "median"),
                    p90_removed=("removed_fraction", lambda values: values.quantile(.9)),
                    max_removed=("removed_fraction", "max"),
                    median_leading=("leading_removed", "median"),
                    median_trailing=("trailing_removed", "median"))
               .sort_values("median_removed", ascending=False))
    print("\nRemoved fraction by canonical drill:")
    print(summary.round(3).to_string())

    print("\nMost heavily trimmed attempts:")
    columns = ["event_id", "nfl_id", "drill_name", "n_frames", "kept_frames",
               "leading_removed", "trailing_removed", "removed_fraction"]
    print(measurements.nlargest(10, "removed_fraction")[columns].round(3).to_string(index=False))


def plot_summary(measurements, output_path=OUTPUT_PATH):
    """Save a histogram and drill-level boxplot of removed frame fractions."""
    order = (measurements.groupby("base")["removed_fraction"]
             .median().sort_values(ascending=False).index)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(measurements["removed_fraction"], bins=25, edgecolor="white")
    axes[0].axvline(0.2, color="red", linestyle="--", label="20% removed")
    axes[0].set_title("Frames removed per route attempt")
    axes[0].set_xlabel("fraction removed from start/end")
    axes[0].set_ylabel("attempts")
    axes[0].legend()

    data = [measurements.loc[measurements["base"] == base, "removed_fraction"]
            for base in order]
    axes[1].boxplot(data, tick_labels=order, vert=False, showfliers=False)
    axes[1].set_title("Removed fraction by drill")
    axes[1].set_xlabel("fraction removed from start/end")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    return fig


def main():
    tracking, qc = load_route_tracking()
    measurements = measure_segments(tracking)
    print_summary(measurements, qc)
    plot_summary(measurements)
    print(f"\nSaved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
