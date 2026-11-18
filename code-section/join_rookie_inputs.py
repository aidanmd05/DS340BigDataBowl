"""[AI-SCAFFOLD] Join Combine sequences to rookie outcomes by receiver ID.

Data assembly and family-output selection only: no scaling, loss weighting, training or evaluation.
"""
import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_KEYS = ['nfl_ids', 'output_labels', 'metric_names', 'y', 'mask', 'counts']


def join_inputs_outcomes(reps, targets, players, family_ids, min_route_reps=3):
    """Return aligned arrays and a roster-wide inclusion report, independent of input order.

    Rep storage is flat and grouped by player; offsets preserve each player's variable
    number of reps without global padding. Outcome axes/values/masks are preserved.
    """
    if min_route_reps < 1:
        raise ValueError('min_route_reps must be positive')
    if not set(TARGET_KEYS).issubset(targets):
        raise ValueError('Target package is missing required arrays')
    ids = np.asarray(targets['nfl_ids'])
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(set(ids)) != len(ids):
        raise ValueError('Target nfl_ids must be a unique integer vector')
    for key in ['output_labels', 'metric_names']:
        labels = np.asarray(targets[key])
        if labels.ndim != 1 or len(set(labels)) != len(labels):
            raise ValueError(f'{key} must be a unique label vector')
    shape = (len(ids), len(targets['output_labels']), len(targets['metric_names']))
    if any(np.asarray(targets[k]).shape != shape for k in ['y','mask','counts']):
        raise ValueError('Target value/mask/count shapes do not match axis labels')
    if np.asarray(targets['mask']).dtype != np.bool_:
        raise ValueError('Target masks must be boolean')
    if not np.isfinite(targets['y']).all():
        raise ValueError('Target values must be finite, with missing entries masked')
    if not np.issubdtype(np.asarray(targets['counts']).dtype, np.integer) or (targets['counts'] < 0).any():
        raise ValueError('Target counts must be nonnegative integers')
    if ((targets['counts'] == 0) & targets['mask']).any():
        raise ValueError('Eligible targets must have supporting observations')
    if players.nfl_id.duplicated().any() or players[['nfl_id','draft_year']].isna().any().any():
        raise ValueError('Roster requires unique receiver IDs and known draft years')
    roster = players.set_index('nfl_id')
    if set(ids) - set(roster.index):
        raise ValueError('Target IDs are missing from the WR roster')
    target_index = {int(pid):i for i,pid in enumerate(ids)}
    by_player, seen, rep_shape = {}, set(), None
    for rep in reps:
        pid = rep['nfl_id']
        if pid not in roster.index:
            raise ValueError(f'Combine receiver {pid} is missing from the WR roster')
        if rep['draft_year'] != roster.loc[pid, 'draft_year']:
            raise ValueError(f'Combine/roster draft years disagree for {pid}')
        key = (rep['event_id'], pid)
        if key in seen:
            raise ValueError(f'Duplicate Combine attempt: {key}')
        seen.add(key)
        x = np.asarray(rep['x'])
        if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
            raise ValueError(f'Invalid rep tensor for {key}')
        rep_shape = x.shape if rep_shape is None else rep_shape
        if x.shape != rep_shape or rep['family_id'] not in family_ids.values():
            raise ValueError('Inconsistent tensor shapes or unknown drill family')
        by_player.setdefault(int(pid), []).append(rep)
    report_rows, kept = [], []
    for pid in sorted(roster.index):
        player_reps = by_player.get(pid, [])
        nr = sum(r['family_id'] == family_ids['route'] for r in player_reps)
        ti = target_index.get(pid)
        eligible = int(targets['mask'][ti].sum()) if ti is not None else 0
        reasons = []
        if nr < min_route_reps:
            reasons.append('insufficient_route_reps')
        included = not reasons
        if included:
            kept.append(int(pid))
        report_rows.append(dict(nfl_id=pid, display_name=roster.loc[pid].get('display_name',''),
            draft_year=int(roster.loc[pid,'draft_year']), n_reps=len(player_reps), n_route_reps=nr,
            n_sprint_reps=sum(r['family_id'] == family_ids['sprint'] for r in player_reps),
            n_valid_outcome_entries=eligible, has_outcome_row=ti is not None,
            has_eligible_rookie_outcomes=eligible > 0, included=included,
            exclusion_reason=';'.join(reasons)))
    if not kept:
        raise ValueError('No receivers have enough usable route reps')
    # A missing source outcome row is retained as unknown, never a zero performance label.
    outcome_shape = (len(kept), *shape[1:])
    y = np.zeros(outcome_shape, dtype=np.asarray(targets['y']).dtype)
    mask = np.zeros(outcome_shape, dtype=bool)
    counts = np.zeros(outcome_shape, dtype=np.asarray(targets['counts']).dtype)
    for i, pid in enumerate(kept):
        if pid in target_index:
            ti = target_index[pid]
            y[i], mask[i], counts[i] = targets['y'][ti], targets['mask'][ti], targets['counts'][ti]

    flat, offsets, rep_player = [], [0], []
    for i, pid in enumerate(kept):
        ordered = sorted(by_player[pid], key=lambda r:(r['family_id'],r['drill_id'],r['event_id']))
        flat.extend(ordered)
        rep_player.extend([i]*len(ordered))
        offsets.append(len(flat))
    arrays = dict(nfl_ids=np.asarray(kept,dtype=np.int64),
        draft_year=np.asarray([roster.loc[pid,'draft_year'] for pid in kept],dtype=np.int64),
        x=np.stack([r['x'] for r in flat]).astype(np.float32),
        rep_player_index=np.asarray(rep_player,dtype=np.int64),
        player_rep_offsets=np.asarray(offsets,dtype=np.int64),
        rep_event_ids=np.asarray([r['event_id'] for r in flat]),
        rep_drill_ids=np.asarray([r['drill_id'] for r in flat],dtype=np.int64),
        rep_family_ids=np.asarray([r['family_id'] for r in flat],dtype=np.int64),
        y=y, mask=mask, counts=counts,
        output_labels=np.asarray(targets['output_labels']).copy(),
        metric_names=np.asarray(targets['metric_names']).copy())
    if arrays['rep_event_ids'].dtype.hasobject:
        arrays['rep_event_ids'] = arrays['rep_event_ids'].astype(str)
    return arrays, pd.DataFrame(report_rows)


def add_participation(arrays, report, rookie_routes, threshold=20):
    """Add recorded route counts; do not infer full-season nonparticipation from absence.

    observed_threshold_mask marks recorded-dataset outcomes, not full-season coverage.
    full_season_threshold_mask marks only confirmed positives: recorded counts already
    reaching threshold. Under-threshold full-season labels remain unknown.
    """
    if threshold < 1:
        raise ValueError('Participation threshold must be positive')
    if rookie_routes.duplicated(['game_id','play_id','nfl_id']).any():
        raise ValueError('Duplicate scored rookie routes')
    if not rookie_routes.season.eq(rookie_routes.draft_year).all() or not rookie_routes.season_type.eq('REG').all():
        raise ValueError('Participation source must contain rookie regular-season routes only')
    ids = arrays['nfl_ids']
    present = set(rookie_routes.nfl_id)
    observed = np.asarray([pid in present for pid in ids])
    route_counts = np.zeros((len(ids),len(arrays['output_labels'])),dtype=np.int32)
    for j,label in enumerate(arrays['output_labels']):
        scope,name = str(label).split(':',1)
        column = {'route':'route_ran','family':'route_family'}.get(scope)
        if column is None:
            raise ValueError(f'Unknown output scope: {scope}')
        counts = rookie_routes[rookie_routes[column].eq(name)].groupby('nfl_id').size()
        route_counts[:,j] = counts.reindex(ids,fill_value=0).to_numpy()
    reaches = route_counts >= threshold
    arrays['participation_route_counts'] = route_counts
    arrays['participation_reaches_threshold'] = reaches
    arrays['participation_observed_mask'] = np.broadcast_to(observed[:,None],route_counts.shape).copy()
    arrays['participation_full_season_threshold_mask'] = reaches.copy()
    arrays['participation_threshold'] = np.asarray(threshold,dtype=np.int32)
    total = rookie_routes.groupby('nfl_id').size()
    report = report.copy()
    report['n_recorded_rookie_routes'] = report.nfl_id.map(total).fillna(0).astype(int)
    report['rookie_record_status'] = np.where(report.nfl_id.isin(present),
        'has_recorded_rookie_routes','no_recorded_rookie_routes')
    report['full_rookie_season_coverage'] = 'unknown'
    return report


def select_prediction_outputs(arrays, families=('break','short','vertical')):
    """Select family-level separation quality and volume without dropping receivers.

    Route volume counts every recorded rookie route, not just routes with usable
    separation, and has no 20-route minimum. No-record receivers remain masked.
    Full prepared outcomes remain available in y/mask for inspection.
    """
    labels = list(arrays['output_labels'])
    metrics = list(arrays['metric_names'])
    indices = [labels.index(f'family:{family}') for family in families]
    sep_index = metrics.index('separation_z')
    shape = (len(arrays['nfl_ids']),len(indices),2)
    values = np.zeros(shape,dtype=np.float32)
    valid = np.zeros(shape,dtype=bool)
    values[:,:,0] = arrays['y'][:,indices,sep_index]
    valid[:,:,0] = arrays['mask'][:,indices,sep_index]
    valid[:,:,1] = arrays['participation_observed_mask'][:,indices]
    values[:,:,1] = arrays['participation_route_counts'][:,indices]
    values[~valid] = 0.
    arrays['prediction_y'] = values
    arrays['prediction_mask'] = valid
    arrays['prediction_output_labels'] = np.asarray([f'family:{family}' for family in families])
    arrays['prediction_metric_names'] = np.asarray(['separation_z','route_count'])
    arrays['prediction_separation_counts'] = arrays['counts'][:,indices,sep_index].copy()


def prediction_table(arrays):
    """Readable selected targets: missing labels stay blank, rather than looking like zero."""
    rows = []
    for i,pid in enumerate(arrays['nfl_ids']):
        for j,label in enumerate(arrays['prediction_output_labels']):
            valid = arrays['prediction_mask'][i,j]
            value = arrays['prediction_y'][i,j]
            rows.append(dict(nfl_id=int(pid),draft_year=int(arrays['draft_year'][i]),
                route_family=str(label).split(':',1)[1],
                separation_z=float(value[0]) if valid[0] else np.nan,
                separation_valid=bool(valid[0]),
                n_separation=int(arrays['prediction_separation_counts'][i,j]),
                route_count=int(value[1]) if valid[1] else np.nan,
                route_count_valid=bool(valid[1])))
    return pd.DataFrame(rows)


def write_readme(output, metadata):
    (output/'README.md').write_text(f'''# Joined Combine inputs and rookie outcomes

Kept **{metadata['kept_players']} receivers** with **{metadata['kept_reps']} Combine
reps**. Each receiver has at least {metadata['min_route_reps']} usable route-drill
reps; rookie outcome eligibility is not required. Sprint reps are retained when available
but do not count toward that minimum. Receivers were joined by `nfl_id`, not row order.
**{metadata['players_without_eligible_outcomes']} receivers have no eligible performance
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
selected tensor has shape `{tuple(metadata['prediction_y_shape'])}`. All 106 current
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
| `x` | Float32 Combine reps, shape `{tuple(metadata['x_shape'])}` = (all retained reps, timesteps, channels). Channels are {', '.join(metadata['channels'])}. No scaling has been fitted. |
| `player_rep_offsets` | Length players+1. Player i's reps are `x[offsets[i]:offsets[i+1]]`. Reps stay grouped, without global padding. |
| `rep_player_index` | For each rep, index i into `nfl_ids`; this links every input rep back to its outcome row. |
| `rep_event_ids` | Original Combine attempt IDs; an attempt is identified by event ID and receiver ID together. |
| `rep_drill_ids`, `rep_family_ids` | Per-rep canonical drill and movement-family IDs. Decode with metadata mappings. |
| `prediction_y` | Selected targets: (players, three families, two metrics), in the label order below. This is the array intended for the upcoming LSTM dataset integration. |
| `prediction_mask` | Per-player/family/metric validity flags. False entries in prediction_y are zero placeholders, not zero performance. |
| `prediction_output_labels`, `prediction_metric_names` | Family order (break, short, vertical) and metric order (separation_z, route_count). |
| `prediction_separation_counts` | Supporting separation observation count for each player/family; no loss weights have been chosen. |
| `y` | Full original prepared outcomes, shape `{tuple(metadata['y_shape'])}` = (players, route/family outputs, metrics), preserved from the rookie target package when available; missing source rows receive zeros and false masks. |
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
''')


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets',type=Path,default=root/'outputs/rookie_context/rookie_targets.npz')
    parser.add_argument('--output',type=Path,default=root/'outputs/lstm_inputs')
    parser.add_argument('--rookie-routes',type=Path,help='Scored rookie route CSV; defaults beside --targets')
    parser.add_argument('--overwrite',action='store_true',help='Replace the generated join files in an existing output directory')
    parser.add_argument('--min-route-reps',type=int)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError('Choose a fresh output directory for the joined data')
    # Sections 1–5 build inputs only; do not import baselines, training or evaluation.
    setup = import_module('01setup')
    inputs = import_module('05rep_tensors')
    players_path = setup.DATA_DIR/'wr_only_players.csv'
    players = pd.read_csv(players_path)
    with np.load(args.targets,allow_pickle=False) as package:
        targets = {key:package[key] for key in TARGET_KEYS}
    minimum = args.min_route_reps if args.min_route_reps is not None else setup.MIN_ROUTE_REPS
    arrays, report = join_inputs_outcomes(inputs.reps,targets,players,setup.FAMILY_ID,minimum)
    rookie_path = args.rookie_routes or args.targets.with_name('rookie_routes.csv')
    target_metadata_path = args.targets.with_name('metadata.json')
    target_metadata = json.loads(target_metadata_path.read_text())
    rookie_routes = pd.read_csv(rookie_path)
    report = add_participation(arrays,report,rookie_routes,threshold=target_metadata['min_routes'])
    select_prediction_outputs(arrays,setup.ROUTE_FAMS)
    if any(r['drill_id'] not in inputs.DRILL_IDS.values() for r in inputs.reps):
        raise ValueError('Unknown drill ID')
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'joined_inputs_outcomes.npz',**arrays)
    report.to_csv(args.output/'player_join_report.csv',index=False)
    prediction_table(arrays).to_csv(args.output/'prediction_targets.csv',index=False)
    pd.DataFrame([dict(nfl_id=int(pid), output_label=str(label),
        n_recorded_routes=int(arrays['participation_route_counts'][i,j]),
        reaches_threshold=bool(arrays['participation_reaches_threshold'][i,j]),
        recorded_count_observed=bool(arrays['participation_observed_mask'][i,j]),
        full_season_threshold_known=bool(arrays['participation_full_season_threshold_mask'][i,j]))
        for i,pid in enumerate(arrays['nfl_ids'])
        for j,label in enumerate(arrays['output_labels'])]).to_csv(
            args.output/'participation_by_route.csv',index=False)
    sources = [args.targets,rookie_path,target_metadata_path,Path(__file__), setup.DATA_DIR/'wr_only_trk.csv',players_path]
    sources += [Path(__file__).with_name(name) for name in ['01setup.py','02load_data.py',
        '03data_audit.py','04cleaning_and_kinematics.py','05rep_tensors.py']]
    metadata = dict(join_key='nfl_id', kept_players=len(arrays['nfl_ids']),
        kept_reps=len(arrays['x']), roster_players=len(players),
        excluded_players=int((~report.included).sum()), min_route_reps=minimum,
        outcome_eligibility='Not required for inclusion; preserve per-metric masks',
        players_with_eligible_outcomes=int(report.loc[report.included,'has_eligible_rookie_outcomes'].sum()),
        players_without_eligible_outcomes=int((~report.loc[report.included,'has_eligible_rookie_outcomes']).sum()),
        participation_threshold=int(arrays['participation_threshold']),
        participation_population='Recorded rookie regular-season routes; no complete-season coverage assumption',
        full_season_negative_labels_available=False,
        output_selection_performed=True,
        prediction_y_shape=list(arrays['prediction_y'].shape),
        prediction_output_labels=arrays['prediction_output_labels'].tolist(),
        prediction_metric_names=arrays['prediction_metric_names'].tolist(),
        prediction_valid_counts=arrays['prediction_mask'].sum(axis=0).tolist(),
        route_count_definition='All recorded rookie regular-season routes per family, regardless of separation availability; no-record players masked unknown',
        scaling_performed=False, training_performed=False,
        evaluation_performed=False, channels=[*setup.CFG['channels'],'duration'],
        drill_ids=inputs.DRILL_IDS, family_ids=setup.FAMILY_ID,
        x_shape=list(arrays['x'].shape), y_shape=list(arrays['y'].shape),
        output_labels=arrays['output_labels'].tolist(),metric_names=arrays['metric_names'].tolist(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    write_readme(args.output,metadata)
    print(f"Joined {metadata['kept_players']} receivers / {metadata['kept_reps']} reps; "
          f"excluded {metadata['excluded_players']} receivers. Saved to {args.output}")


if __name__=='__main__':
    main()
