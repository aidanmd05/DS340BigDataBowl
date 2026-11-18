## Start with the receiver summary

`rookie_receiver_summary.csv` is the easiest file to inspect. Each row describes
one receiver **during his rookie regular season**, for one route type or route
family. `rookie_routes.csv` contains the individual routes behind those averages.
`context_buckets.csv` describes their comparison groups. `rookie_targets.npz`
contains numeric copies of selected summary metrics for future model integration.
`metadata.json` records how the files were prepared.

These are descriptive outcomes, not LSTM predictions or evaluation results. The
reference contains all regular-season routes in `data/wr_only_play.csv`, pooling
all available seasons, including the rookie routes being scored. It contains only
the available WR draft cohorts, not the NFL's entire WR population. Source CSVs
remain in `data/`; a redundant filtered copy is not exported.

## 1. rookie_receiver_summary.csv

One row per `nfl_id` × `scope` × `route`. Receivers without recorded rookie routes
do not appear here. Values below the minimum count remain visible for inspection;
the eligibility flags determine whether they enter the numeric targets.

| Column | Exact meaning |
|---|---|
| `nfl_id` | Receiver identifier; join to `data/wr_only_players.csv` for the name and to Combine inputs by this ID. |
| `draft_year` | Draft class; the scored game season equals this year. |
| `scope` | `route` for a specific route type; `family` for a combination of related types. These rows overlap and must not be added together. |
| `route` | Route name or family name, interpreted using `scope`. |
| `n_routes` | Number of this receiver's rookie-season routes in this output, whether or not separation is available. |
| `n_games` | Distinct games contributing those routes, not necessarily games with usable separation. |
| `n_separation` | Routes with a usable separation z-score; requires observed separation and a nonzero reference standard deviation. |
| `n_target` | Routes with both a known observed target flag and an available contextual target probability. This can exceed `n_separation`. |
| `raw_separation_mean` | Mean observed separation at pass forward, in yards, ignoring missing values. No context adjustment. |
| `separation_z` | Mean of the individual routes' contextual z-scores. Positive means more separation than the reference expects. This is not a z-score of the receiver's raw average. |
| `separation_percentile` | Mean of individual contextual percentiles, on a 0–100 scale. It is not this receiver's rank among receivers. Missing route scores are omitted. |
| `target_rate` | Actual targets divided by `n_target`, on a 0–1 scale. |
| `expected_target_probability` | Mean context-based target probability over those same `n_target` routes. It describes the contexts the receiver faced, not a receiver-specific skill prediction. |
| `target_rate_above_expected` | `target_rate - expected_target_probability`. A value of `0.05` means five percentage points more often targeted than the context baseline. |
| `separation_eligible` | Whether `n_separation` meets the configured minimum, currently 20. |
| `target_eligible` | Whether `n_target` meets that minimum. It does not indicate separation eligibility. |

Averages give each valid route equal weight. Target metrics use the same routes
for observed and expected rates. Separation means omit missing separation; the
percentile mean can include a zero-variance context whose z-score is undefined.
Such a route is excluded from `n_separation`, which controls separation eligibility.

Families follow the existing pipeline:

| Family | Included route types |
|---|---|
| `vertical` | GO, POST, CORNER, WHEEL |
| `break` | OUT, IN, CROSS, SLANT, ANGLE |
| `short` | HITCH |

FLAT and SCREEN remain available as individual route outputs but have no family.
No claim of causal receiver ability is made: play design, quarterback behavior and
other context can still affect these outcomes.

## 2. rookie_routes.csv

One row per receiver-play in the receiver's rookie regular season. The combination
of `game_id`, `play_id`, and `nfl_id` identifies a row. This is the file to use when
you want to see a particular route's separation percentile and target probability.

| Column(s) | Exact meaning |
|---|---|
| `game_id`, `play_id`, `nfl_id` | Game, play, and receiver identifiers. |
| `route_ran` | Recorded route type. |
| `separation_at_pass_forward` | Observed separation at pass release, in yards; blank if unavailable. |
| `cushion` | Recorded starting cushion, in yards; a context variable, not the pass-release outcome. |
| `team_coverage_man_zone`, `team_coverage_type` | Original broad and detailed defensive coverage labels; may be blank. |
| `target` | Original flag indicating whether this receiver was targeted on this play. |
| `draft_year`, `season`, `season_type` | Metadata used to retain `season == draft_year` and `season_type == REG`. |
| `is_rookie_season` | Always true in this scored file. |
| `coverage_type`, `coverage_group` | Detailed coverage and broad man/zone values used in grouping; missing values become `UNKNOWN`. |
| `cushion_bucket` | Context interval: `[0,2)`, `[2,4)`, `[4,6)`, `[6,8)`, `[8,10)`, `[10,infinity)` yards. Unknown, negative or nonfinite values become `UNKNOWN`. For example, exactly 4 yards belongs to `4-6`. |
| `target_observed` | Numeric target flag: 1 for targeted, 0 for not targeted, blank if unknown. |
| `route_family` | Family mapping above; blank for excluded/unmapped types. |
| `context_sep_mean` | Expected separation in yards from the pooled context distribution. |
| `context_sep_std` | Population standard deviation of that same pooled distribution, in yards. It includes both within-bucket variation and differences between pooled means. |
| `separation_z` | `(separation_at_pass_forward - context_sep_mean) / context_sep_std`. Blank if separation is unavailable or the standard deviation is effectively zero. |
| `separation_percentile` | Percentile of observed separation within the pooled empirical reference distribution. Ties receive their midpoint rank. This is not calculated by assuming a normal distribution. Blank if separation is missing. |
| `expected_target_probability` | P(targeted given route, coverage, cushion), between 0 and 1, estimated from the entire reference population. Routes missing separation still contribute when their target flag is known. |
| `sep_bucket_n`, `sep_bucket_players` | Separation observation count and distinct receiver count in the most detailed usable separation bucket. These are direct bucket counts, not the total support of the pooled distribution. |
| `sep_bucket_weight` | Weight assigned to that bucket versus its broader pooled parent; 0 means entirely parent, 1 means entirely bucket. It is not an attention weight or confidence interval. |
| `sep_reference_level` | Most detailed bucket level used for separation, defined below. |
| `target_bucket_n`, `target_bucket_players` | Known-target route count and distinct receiver count in the most detailed usable target bucket. |
| `target_bucket_weight` | Weight of that target bucket versus its broader pooled parent. |
| `target_reference_level` | Most detailed level used for target probability. It may differ from the separation level. |

### Bucket levels and pooling

| Level | Variables defining the bucket |
|---|---|
| 0 | Route type only |
| 1 | Route type + cushion bucket |
| 2 | Route type + broad man/zone group + cushion bucket |
| 3 | Route type + detailed coverage + broad man/zone group + cushion bucket |
| -1 | No usable reference for that metric; score unavailable |

At each child level, the default weight is
`n / (n + 100) * min(distinct_receivers / 15, 1)`.
The resulting distribution/rate is `weight * child + (1 - weight) * pooled_parent`.
Separation and targeting use their own counts and weights. The root uses weight 1.
Thus a level-3 score usually still incorporates broader levels; level 3 does not
mean the estimate comes only from its most specific bucket.

## 3. context_buckets.csv

One row per reference bucket at every level. These rows describe **all available
regular-season receiver seasons**, not just rookie outcomes. Counts overlap across
levels: do not sum levels together.

| Column(s) | Exact meaning |
|---|---|
| `level` | 0–3, using the level definitions above. |
| `route_ran`, `cushion_bucket`, `coverage_group`, `coverage_type` | Bucket keys. A blank key means the variable is not used at this level; `UNKNOWN` means the variable is used but its recorded value was missing. |
| `n_routes`, `n_players` | All routes and distinct receivers in the exact bucket. |
| `n_separation`, `separation_players` | Finite separation observations and distinct receivers contributing them. Unlike the summary's score count, this is a raw finite-observation count. |
| `raw_sep_mean`, `raw_sep_std` | Exact bucket's unpooled mean and population standard deviation, in yards. |
| `raw_sep_p10`, `raw_sep_p25`, `raw_sep_p50`, `raw_sep_p75`, `raw_sep_p90` | Exact bucket's unpooled separation quantiles, in yards. `p50` is the median; these are benchmark values, not a receiver's percentile scores. |
| `n_target`, `target_players` | Routes with known target flags and distinct contributing receivers, independent of separation availability. |
| `raw_target_rate` | Exact bucket's targeted count divided by `n_target`, before pooling. |
| `sep_child_weight`, `target_child_weight` | This bucket's default support-based mixing weights. The root is 1. A bucket without usable observations contributes no child weight. |
| `context_sep_mean`, `context_sep_std` | Effective separation mean and standard deviation after mixing this level with all its broader parents. These effective values generate the route z-scores. |
| `expected_target_probability` | Effective target rate after the same hierarchical pooling, using target-specific support. This generates the route target probabilities. |

The raw quantiles do not describe the final pooled distribution. The scored route
percentiles come from the mixture empirical CDF, not interpolation between these
five exported quantiles. The raw source data and preparation code reconstruct that
CDF exactly.

## 4. rookie_targets.npz

This is an array package, not a trained model or Combine input sequence file.

| Array | Meaning |
|---|---|
| `nfl_ids` | Receiver IDs defining the first axis. Includes the full WR roster, even receivers without usable rookie outcomes. Join to Combine reps by ID. |
| `output_labels` | Second-axis labels, such as `route:GO` or `family:break`. The route and family outputs overlap. |
| `metric_names` | Third-axis labels, in order: `separation_z`, `separation_percentile`, `target_rate`, `expected_target_probability`, `target_rate_above_expected`. |
| `y` | Float32 summary values with axes `(receiver, route/family, metric)`. Values without enough observations or without a finite score are stored as zero. |
| `mask` | Boolean array with the same shape. Only entries marked true are valid model targets. Zero with a true mask is a real score; zero with a false mask is a placeholder. |
| `counts` | Integer supporting route counts in the same shape, including entries below the eligibility threshold. Separation metrics use the summary's `n_separation`; target metrics use its `n_target`. |

The 20-observation rule is applied separately to each output and metric. Receivers
with no scored rookie routes have all-false masks. The array contains unstandardized
metrics with different units; it is not yet wired into the LSTM dataset or loss.
Expected target probability is a context baseline; separation z-score and target
rate above expected are more direct candidate receiver-performance labels.

```python
import numpy as np

a = np.load('outputs/rookie_context/rookie_targets.npz', allow_pickle=False)
# Interpret entries through all three sets of labels, never by assumed row order.
print(a['nfl_ids'], a['output_labels'], a['metric_names'])
```

## 5. metadata.json

Records the reference/scoring populations, whether scored receivers are included
in the reference, the fact that no evaluation was performed, the target probability
and percentile definitions, cushion intervals, pooling hierarchy and support
parameters, minimum-route rule, actual row/receiver counts, source-file SHA-256
hashes, and NumPy/pandas versions. Hashes let you check whether the source data
changed; they are not football metrics. This file is needed to interpret and
reproduce the prepared outcomes.

## Recreate the outputs

From the repository root, use a fresh destination:

```sh
.venv/bin/python code-section/prepare_rookie_context.py --output outputs/rookie_context_new
```

This prepares data only. It does not run baselines, model training, validation or
cross-validation. Before future predictive evaluation, fit context references using
only the allowed training population; the current full-population reference also
contains the scored receivers' outcomes and later seasons.
