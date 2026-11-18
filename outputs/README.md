# Current outputs

This folder contains data preparation outputs for the rookie receiver project.
Old baseline comparisons, draft-class evaluations, neural smoke runs, model weights,
fold files, and validation reports were removed because they used the older targets
and are not relevant to the current data-only stage.

| Output | What it shows / why it remains |
|---|---|
| [rookie_context/README.md](rookie_context/README.md) | Start here: an exact guide to the populations, metrics, every CSV column, and the numeric target arrays. |
| [rookie_context/rookie_receiver_summary.csv](rookie_context/rookie_receiver_summary.csv) | Main readable result: each rookie's average separation performance and observed versus contextual target rate, separately by route type and route family. Includes sample sizes and eligibility. |
| [rookie_context/rookie_routes.csv](rookie_context/rookie_routes.csv) | Individual rookie routes, with observed separation, contextual separation percentile/z-score, context-based target probability and actual target flag. |
| [rookie_context/context_buckets.csv](rookie_context/context_buckets.csv) | Reference distributions: route/coverage/cushion groups, support counts, raw separation quantiles and effective pooled separation/target expectations. |
| [rookie_context/rookie_targets.npz](rookie_context/rookie_targets.npz) | Receiver-aligned numeric outcomes, masks, counts and axis labels for future LSTM integration. Not a trained model or sequence-input file. |
| [rookie_context/metadata.json](rookie_context/metadata.json) | Reproducibility: preparation parameters, row counts, source hashes, versions, and population definitions. |
| [rep_traces_by_family.png](rep_traces_by_family.png) | Combine-input sanity check: six sampled speed traces per drill family, before training standardization. Route drills and sprints are shown separately. |

## Reading the Combine trace plot

- Each line is one Combine rep, labeled by receiver ID, canonical drill and trimmed duration.
- The horizontal axis is **0–100% progress through that rep**, not elapsed seconds.
  Each rep is resampled to 100 timesteps. Two aligned x-values do not imply the
  same elapsed time or a shared cut/catch event.
- The vertical axis is speed in **yards per second**. It is not a standardized score.
- The legend's duration is elapsed seconds across the retained movement segment.
- The examples show the input shape and rough motion profiles. They do not show
  rookie NFL performance, learned attention weights, model predictions or accuracy.
- Sampling is reproducible with seed 0. Examples come from usable Combine reps and
  are not restricted to receivers who meet every future target threshold.

## Important distinctions

The reference covers all available regular-season receiver seasons in the WR
extract; the scored routes are rookie seasons only. A target probability is the
**context's historical target rate**, not an individual rookie's predicted ability.
An average route percentile is not a receiver ranking percentile. No model has
been evaluated using these new contextual outcomes.

Source CSVs stay in `data/`. The old evaluation code remains available, but running
it explicitly would create a separate set of outputs using its configured targets.
Only the data-preparation command is relevant at this stage; see the detailed guide.
