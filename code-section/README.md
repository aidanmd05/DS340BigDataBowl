# Code-section pipeline

Python 3.9+ is supported. Install the direct runtime dependencies in an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run from the repository root:

```sh
# Load, audit, clean, and build inputs without starting training.
.venv/bin/python code-section/run_sections.py --through 7

# Five-fold baselines: mean, A, B, draft-only, and A + draft pick.
.venv/bin/python code-section/run_sections.py --evaluate baselines --output outputs/baselines

# Main comparison adds sequence-only C and C + A, with three seeds by default.
.venv/bin/python code-section/run_sections.py --evaluate neural --output outputs/neural

# Fast end-to-end check of pooling and drill-family variants; not final results.
.venv/bin/python code-section/run_sections.py --evaluate ablations --folds 2 --epochs 2 --seeds 1 --bootstrap 20 --output outputs/smoke

# Secondary generalization check; pools predictions across held-out draft classes.
.venv/bin/python code-section/run_sections.py --evaluate baselines --mode draft_class --output outputs/draft_classes

# Focused correctness checks (requires local data).
.venv/bin/python -m unittest discover -s tests -v
```

Use a fresh output directory for each evaluation. `--evaluate ablations` matches
**every model's cohort** to players with both sprint and route reps, then compares
mean/max/weighted/attention pooling and route-only/sprint-only inputs. Main
comparisons retain all eligible players. Each experiment uses identical outer
folds, inner validation players, seeds, and fold-specific target values.

`evaluation.py --help` lists flags including `--min-routes` (15/20/25 sensitivity),
`--min-route-reps`, `--timesteps`, `--channels`, `--epochs`, and `--seeds`.
Each invocation rebuilds reps, targets, the cohort, and splits from its configuration;
changing an imported `CFG` alone does not update notebook-style module globals.
Sections 11 and 12 expose Python wrappers around the same runner.

## Data-only rookie contextual targets

Prepare outcomes for future Combine-sequence/LSTM/attention work, without importing
tracking, training a model, or running evaluation:

```sh
.venv/bin/python code-section/prepare_rookie_context.py
# Use --output with a fresh directory for another run.
.venv/bin/python -m unittest discover -s tests -p test_rookie_context.py -v
```

This standalone path reads **only** `wr_only_play.csv`, `wr_only_players.csv`, and
`games.csv`. The reference is all regular-season WR routes from every available
season in the WR extract, including second/third seasons and the scored players.
Only regular-season plays with `season == draft_year` are scored. Unlike the
predictive evaluation path below, these are pooled descriptive benchmarks, not
fold-specific model labels. Future model evaluation must refit the reference using
its allowed training population; these full-population scores must not be reused
as if they had been fitted inside a split.

Context is route type × defensive coverage × starting cushion. Cushion buckets
are `[0,2)`, `[2,4)`, `[4,6)`, `[6,8)`, `[8,10)`, and `[10,infinity)` yards; unknown
cushion and coverage have explicit categories. Small buckets borrow from route ×
man/zone × cushion, route × cushion, and route alone. At each level the child
weight is `n/(n+100) * min(distinct_receivers/15, 1)` by default, separately for
separation and target observations. The route-only root has weight one.

Separation percentiles (0–100) use the resulting mixture empirical distribution
with midpoint ranks for ties. Z-scores use its mean and population standard
deviation; zero-variance references yield a missing z-score. Target probability is
the corresponding mixture of observed target rates. Its denominator includes all
routes with a known target flag, **even if separation is missing**. This probability
is a **context baseline**, not a receiver-specific prediction. Each scored route
retains its observed target flag and its raw separation.

Default outputs in `outputs/rookie_context/`:

Start with [the output guide](../outputs/rookie_context/README.md) for every column,
unit, denominator, eligibility mask and pooling level. The preparation script writes
this guide into every new output package. The filtered reference-row copy is not
exported because it is reproducible from the original CSVs.

- `context_buckets.csv`: counts, raw separation quantiles and moments, shrinkage
  weights, effective reference moments and expected target rates at every level.
- `rookie_routes.csv`: rookie routes with contextual z-scores, percentiles, target
  probabilities, bucket support, and reference levels.
- `rookie_receiver_summary.csv`: per-receiver/per-route and per-family averages,
  observed and expected target rates, their difference, counts and eligibility.
- `rookie_targets.npz`: numeric `y`, boolean `mask`, and integer `counts`, each shaped
  `(106 receivers, route/family outputs, 5 metrics)` for the current roster. Named
  `nfl_ids`, `output_labels` and `metric_names` arrays define alignment. Missing or
  insufficient targets are zero with `mask=False`; never train on those zeros.
- `metadata.json`: population definitions, parameters, versions, source hashes and
  an explicit record that no model evaluation was performed.

The five array metrics are mean separation z-score, mean separation percentile,
observed target rate, expected contextual target probability, and observed minus
expected target rate. Prefer separation z-score and target-rate-above-expected for
receiver-performance labels; the expected probability describes opportunity.
Percentile averages are averages over routes, not receiver percentile rankings.

The default eligibility threshold is 20 valid observations per output/metric,
configurable with `--min-routes`. Routes with missing separation still contribute
to target-rate summaries. All route types are retained in per-route outputs;
FLAT/SCREEN remain excluded from the existing three family outputs. Combine
sequence inputs remain in section 5; join the prepared outcomes by `nfl_id`, not
by array row position. No evaluation/training commands run automatically.

## Data and evaluation contract

WR extracts live in `data/`. `games.csv` must be in `data/` or
`nfl-big-data-bowl-2027/`; its verified `season_type == "REG"` mapping filters the
outcomes before audit or training. Unknown game IDs and missing columns fail early.
Player eligibility requires at least three **route-drill** reps and at least one
route-family target with 20 finite separation observations by default. Sprint reps
do not count toward the route threshold.

Targets are mean separation residuals after subtracting route-type means. Those
means are fitted on **inner-training players only** and shared across models in
that fold. An unseen training route type uses the training mean for its family;
an entirely absent training family fails explicitly. The exploratory audit's
all-player route adjustment is never passed to evaluation. Raw section-6 targets
establish eligibility only.

Ridge models fit on outer-training players. Neural models fit on inner-training
players and early-stop on inner-validation players, with no outer-test access.
Imputation, missingness flags and scaling are fitted on each model's training
partition. Baselines use unweighted Ridge; neural loss uses square-root route-count
weights. Metrics are unweighted player-level Spearman and R². Bootstrap intervals
resample players with fixed OOF predictions, including paired differences versus A;
they do not estimate uncertainty from retraining or selecting experiments. Seed
scores are saved separately. Use inner validation for model selection; exploratory
OOF ablation comparisons are not a new independent final test.

QC rejects short/gapped attempts, impossible peak speeds, invalid timestamps and
nonfinite kinematic channels. Reps are sorted and interpolated in actual elapsed
time; the last channel is duration in seconds. Route reps retain their moving
interval, while sprint rollout ends when speed falls below half the peak. Automatic
route-break/catch alignment and break-retention features remain deferred until
validated against labeled examples. Coverage/cushion adjustment and shrinkage are
also not applied.

## Outputs

Each run writes resolved configuration, data hashes, software versions, cohort and
player folds; training preprocessing and route adjustments; per-model OOF CSVs;
neural weights, per-seed predictions and scores; metrics with paired player-bootstrap
95% intervals; residual tables and diagnostic/comparison plots; and sampled rep
traces. Weighted/attention runs also save held-out per-rep pooling weights.
Weights are shared across route outputs, and are descriptive rather than causal.
Fold checkpoints retain completed predictions if a later fold fails; automatic
resume is not implemented.

Numbered filenames and `# %%` cell markers remain available for notebook-style
exploration. `run_sections.py` without `--evaluate` imports them in order and does
not launch experiments. The source notebooks in `legacy-code/` are unchanged.
Existing personal audit and trace-plot additions are preserved. AI-assisted code is
marked `[AI-SCAFFOLD]`; update the course disclosure to include these changes.

## File order

1. `01setup.py`
2. `02load_data.py`
3. `03data_audit.py`
4. `04cleaning_and_kinematics.py`
5. `05rep_tensors.py`
6. `06targets.py`
7. `07baselines.py`
8. `08dataset_and_batching.py`
9. `09model.py`
10. `10training.py`
11. `11cross_validation.py`
12. `12variations_and_ablations.py`
13. `13interpretation.py`
14. `14wrap_up_checklist.py`

## Original notebook overview

# DS340 Final Project Scaffold: Combine Drill Sequences -> NFL WR Route Performance

**Question.** Does a learned representation of a WR's Combine drill movement (LSTM encoders + pooling) predict NFL route performance (e.g. separation at pass forward) *beyond* the official Combine numbers and draft position?

**Pipeline.**
```
reps in each drill family -> [LSTM encoder(s)] -> one vector per rep
        -> [pooling: mean | max | learned-weight | self-attention] -> player vector
        -> (+ optional attributes) -> [MLP head] -> one prediction per route family
```

**Feature sets compared (same downstream evaluation for all):**
- **A**: official Combine numbers + height/weight (+ draft pick as a variation)
- **B**: hand-built kinematics from the tracking data (top speed, braking, turn sharpness, ...)
- **C**: learned (LSTM / attention model)

**How the sections map to your milestones**
- Sections 1-7 (setup, audit, cleaning, tensors, targets, baselines A/B) = your **mid-project TA milestone** (aim for ~Nov 17).
- Sections 8-11 = the neural model and grouped cross-validation.
- Sections 12-13 = variations/ablations and interpretation (figures for the paper + lightning talk).

**Conventions.**
- Anything marked `VERIFY` is a column name or value I inferred from public descriptions of the data. Check it against your actual files.
- Anything marked `TODO` is yours to complete.
- **AI statement (course requirement):** this scaffold was generated with Claude. The syllabus requires you to disclose AI use and comment AI-written code. Keep the `# [AI-SCAFFOLD]` tags on lines you did not write yourselves, and delete the tag on code you rewrite. Update your written AI statement to match what you actually kept.

---
**Decisions locked in so far (from the data audit; revisit only with evidence)**
- **Population:** WRs only, using premade `wr_only_*.csv` files (106 WRs have Combine tracking).
- **Target:** average `separation_at_pass_forward` per route family (`vertical`, `break`, `short`), adjusted for route type. FLAT and SCREEN are excluded.
- **Min routes per family:** `MIN_ROUTES = 20` (split-half reliability plateaus there; 15 and 25 are sensitivity checks). Note: this is a **count of routes**, not a distance.
- **Combine drills in the encoder:** the WR route drills (after merging name variants and mirroring left-side reps) plus the 40-yard dash. Left out: 3-cone and shuttle tracking (too sparse), ball tracking (about a third of attempts), `GAUNTLET_DRILL`, `OVER_SHOULDER_ADJUST`, and `CIRCUS` (2025 class only).
- **Channels:** `speed`, `acc_tan` (signed, along the path), `acc_cent` (signed, sideways). The sensor's `a` is the total acceleration magnitude, so it adds no new information. Raw channels are kept as an ablation.
