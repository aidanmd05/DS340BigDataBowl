# Joined Combine inputs and rookie outcomes

Kept **106 receivers** with **1389 Combine
reps**. Each receiver has at least 3 usable route-drill
reps; rookie outcome eligibility is not required. Sprint reps are retained when available
but do not count toward that minimum. Receivers were joined by `nfl_id`, not row order.
**43 receivers have no eligible performance
outcomes** and are retained with false performance masks.

- `joined_inputs_outcomes.npz`: aligned, unstandardized sequences and outcomes.
- `prediction_targets.csv`: the chosen two outcomes per route family. This is the
  main human-readable table for the first LSTM outputs; unavailable labels are blank.
- `player_join_report.csv`: every WR roster member, with name/year, rep counts,
  count of valid outcome entries, rookie record availability, inclusion flag and exclusion reasons. Valid
  entries count route/family × metric combinations, not independent observations.
- `participation_by_route.csv`: one row per retained receiver and route/family
  output, showing recorded route counts, the threshold indicator, and observation masks.
  `recorded_count_observed` is false for players with no recorded rookie routes;
  `full_season_threshold_known` is true only for already-observed threshold successes.
- `metadata.json`: source hashes, input channel and ID mappings, dimensions,
  eligibility rules and a record that no model was trained/evaluated.

## Selected LSTM prediction outputs

The first prediction task is **separation z-score and recorded rookie route count**
for each of `break`, `short`, and `vertical`: six predictions per receiver. The
selected tensor has shape `(106, 3, 2)`. All 106 current
receivers remain included. Route counts are raw counts, not percentages, threshold
flags, log counts or standardized values. They count every recorded route in the
family, including routes without separation. FLAT/SCREEN are outside these families.

Separation uses its existing 20-observation eligibility rule. Route counts do not
require 20 routes: a recorded count of 3 is a valid count label. With recorded rookie
routes in another family, zero routes in a family is a valid **recorded-dataset**
zero. If a receiver has no recorded rookie routes at all, all count labels are
masked unknown. Counts describe available records, not confirmed full-season totals.

`prediction_targets.csv` has `nfl_id`, `draft_year`, `route_family`, `separation_z`,
`separation_valid`, `n_separation`, `route_count`, and `route_count_valid`. The
separation count is supporting scorable observations, whereas route_count is the
chosen volume outcome. The validity flags govern which values are future labels.

## Array layout

| Array | Meaning |
|---|---|
| `nfl_ids`, `draft_year` | Receiver IDs and draft years in the common player-axis order. |
| `x` | Float32 Combine reps, shape `(1389, 100, 4)` = (all retained reps, timesteps, channels). Channels are speed, acc_tan, acc_cent, duration. No scaling has been fitted. |
| `player_rep_offsets` | Length players+1. Player i's reps are `x[offsets[i]:offsets[i+1]]`. Reps stay grouped, without global padding. |
| `rep_player_index` | For each rep, index i into `nfl_ids`; this links every input rep back to its outcome row. |
| `rep_event_ids` | Original Combine attempt IDs; an attempt is identified by event ID and receiver ID together. |
| `rep_drill_ids`, `rep_family_ids` | Per-rep canonical drill and movement-family IDs. Decode with metadata mappings. |
| `prediction_y` | Selected targets: (players, three families, two metrics), in the label order below. This is the array intended for the upcoming LSTM dataset integration. |
| `prediction_mask` | Per-player/family/metric validity flags. False entries in prediction_y are zero placeholders, not zero performance. |
| `prediction_output_labels`, `prediction_metric_names` | Family order (break, short, vertical) and metric order (separation_z, route_count). |
| `prediction_separation_counts` | Supporting separation observation count for each player/family; no loss weights have been chosen. |
| `y` | Full original prepared outcomes, shape `(106, 15, 5)` = (players, route/family outputs, metrics), preserved from the rookie target package when available; missing source rows receive zeros and false masks. |
| `mask` | Boolean validity mask with the same shape as y. False entries must not contribute to a future loss. |
| `counts` | Supporting observation counts, aligned with y and mask. |
| `participation_route_counts` | Recorded rookie route counts, shape (players, outputs), using all routes regardless of separation availability. Counts overlap between route and family outputs. |
| `participation_reaches_threshold` | Whether each recorded count reaches the configured threshold (currently 20). False does not establish full-season failure. |
| `participation_observed_mask` | Whether the player has any recorded rookie regular-season routes. If false, zero counts are placeholders for no records, not proof of zero participation. If true, counts describe this dataset only; missing individual route types count as zero recorded routes. |
| `participation_full_season_threshold_mask` | Only confirmed positive threshold labels are valid: observed counts already reach the threshold. Below-threshold labels remain masked because full-season coverage has not been established. |
| `participation_threshold` | Scalar minimum recorded route count used for the participation indicator; distinct from separation-score availability. |
| `output_labels`, `metric_names` | Labels for y's second and third axes; all prepared outputs and metrics are preserved. |

```python
import numpy as np

a = np.load('outputs/lstm_inputs/joined_inputs_outcomes.npz', allow_pickle=False)
i = 0
start, end = a['player_rep_offsets'][i:i+2]
player_id = a['nfl_ids'][i]
sequences = a['x'][start:end]
outcomes, valid = a['prediction_y'][i], a['prediction_mask'][i]
```

Receivers with no eligible performance outcomes remain in this package with their
performance masks false. The report distinguishes missing source outcome rows from
present rows without eligible metrics, and no recorded rookie routes from recorded
routes below performance eligibility. Full rookie-season observation coverage is
marked unknown; no absence is automatically labeled a full-season participation failure.

The underlying rookie outcomes are explained in
[the contextual output guide](../rookie_context/README.md). Their full-population
reference includes scored receivers. This join does not make them fold-safe for
future predictive evaluation. Output selection is complete; no train/test split, normalization,
loss weighting, model fitting, or evaluation has been performed. Existing training
commands do not yet consume this package automatically.
