"""[AI-SCAFFOLD] Prepare descriptive rookie targets; never trains/evaluates a model.

Reference: all WR cohort seasons in wr_only_play, regular season by default.
Scores: rookie-season routes only. Reference deliberately includes scored players.
Future predictive evaluation must refit ContextReference on training players only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEP = 'separation_at_pass_forward'
UNKNOWN = 'UNKNOWN'
CUSHION_LABELS = ['0-2', '2-4', '4-6', '6-8', '8-10', '10+']
LEVELS = [('route_ran',), ('route_ran', 'cushion_bucket'),
          ('route_ran', 'coverage_group', 'cushion_bucket'),
          ('route_ran', 'coverage_type', 'coverage_group', 'cushion_bucket')]
# Matches section 1 without importing setup, torch, tracking, audits or experiments.
ROUTE_FAMILY = {'GO':'vertical', 'POST':'vertical', 'CORNER':'vertical', 'WHEEL':'vertical',
                'OUT':'break', 'IN':'break', 'CROSS':'break', 'SLANT':'break',
                'ANGLE':'break', 'HITCH':'short'}
METRICS = ['separation_z', 'separation_percentile', 'target_rate',
           'expected_target_probability', 'target_rate_above_expected']


def prepare_rows(play, players, games):
    required = ['game_id', 'play_id', 'nfl_id', 'route_ran', SEP, 'cushion',
                'team_coverage_man_zone', 'team_coverage_type', 'target']
    missing = set(required) - set(play.columns)
    if missing:
        raise ValueError(f'Missing play columns: {sorted(missing)}')
    if play.duplicated(['game_id', 'play_id', 'nfl_id']).any():
        raise ValueError('Duplicate player-play keys')
    if players.nfl_id.duplicated().any() or games.game_id.duplicated().any():
        raise ValueError('Player/game metadata must have unique keys')
    if not players.nfl_position.eq('WR').all():
        raise ValueError('wr_only_players must contain WRs only')
    p = play[required].merge(players[['nfl_id', 'draft_year']], on='nfl_id', how='left',
                            validate='many_to_one')
    p = p.merge(games[['game_id', 'season', 'season_type']], on='game_id', how='left',
                validate='many_to_one')
    if p[['draft_year', 'season', 'season_type']].isna().any().any():
        raise ValueError('Unknown player/game or missing draft-year/season metadata')
    p = p[p.season_type.eq('REG') & p.route_ran.notna()].copy()
    p['is_rookie_season'] = p.season.eq(p.draft_year)
    p['coverage_type'] = p.team_coverage_type.fillna(UNKNOWN)
    p['coverage_group'] = p.team_coverage_man_zone.fillna(UNKNOWN)
    cushion = pd.to_numeric(p.cushion, errors='raise')
    valid = np.isfinite(cushion) & (cushion >= 0)
    p['cushion_bucket'] = pd.cut(cushion.where(valid), [0, 2, 4, 6, 8, 10, np.inf],
                                 labels=CUSHION_LABELS, right=False).astype(object).fillna(UNKNOWN)
    # Accept CSV booleans, strings or binary numeric flags; unknown flags stay missing.
    encoded = p.target.astype('string').str.upper().map(
        {'TRUE':1., 'FALSE':0., '1':1., '0':0., '1.0':1., '0.0':0.})
    if (p.target.notna() & encoded.isna()).any():
        raise ValueError('Unrecognized target flags')
    p['target_observed'] = encoded.astype(float)
    p[SEP] = pd.to_numeric(p[SEP], errors='raise').where(lambda x: np.isfinite(x))
    p['route_family'] = p.route_ran.map(ROUTE_FAMILY)
    return p.reset_index(drop=True)


class ContextReference:
    """Hierarchical empirical separation CDFs and binary target rates.

    Each child is mixed with its parent using n/(n+prior_routes) times
    min(distinct_players/prior_players, 1). Supports are metric-specific.
    Percentiles use midrank ties; z-scores use the mixture's population moments.
    """
    def __init__(self, prior_routes=100, prior_players=15):
        if prior_routes <= 0 or prior_players <= 0:
            raise ValueError('Support parameters must be positive')
        self.prior_routes, self.prior_players = prior_routes, prior_players
        self.nodes = {}

    def fit(self, rows):
        self.nodes = {}
        for level, columns in enumerate(LEVELS):
            grouper = list(columns) if len(columns) > 1 else columns[0]
            for key, g in rows.groupby(grouper, sort=True, dropna=False):
                key = key if isinstance(key, tuple) else (key,)
                sep = g[g[SEP].notna()]
                target = g[g.target_observed.notna()]
                values = np.sort(sep[SEP].to_numpy(dtype=float))
                self.nodes[(level, key)] = dict(values=values,
                    sep_players=int(sep.nfl_id.nunique()), n_target=len(target),
                    target_players=int(target.nfl_id.nunique()),
                    target_sum=float(target.target_observed.sum()), n_routes=len(g),
                    n_players=int(g.nfl_id.nunique()))
        if not self.nodes:
            raise ValueError('No reference routes')
        return self

    def _chain(self, row):
        return [self.nodes.get((level, tuple(row[c] for c in columns)))
                if all(c in row for c in columns) else None
                for level, columns in enumerate(LEVELS)]

    def _weight(self, n, players):
        return n / (n + self.prior_routes) * min(players / self.prior_players, 1.)

    def score(self, row):
        mean = second = cdf = probability = np.nan
        sep_n = target_n = sep_players = target_players = 0
        sep_weight = target_weight = np.nan
        sep_level = target_level = -1
        value = row[SEP]
        for level, node in enumerate(self._chain(row)):
            if node is None:
                continue
            values = node['values']
            if len(values):
                w = self._weight(len(values), node['sep_players']) if np.isfinite(mean) else 1.
                m, m2 = values.mean(), np.mean(values ** 2)
                rank = ((np.searchsorted(values, value, 'left') + np.searchsorted(values, value, 'right'))
                        / (2 * len(values))) if np.isfinite(value) else np.nan
                mean = w * m + (1-w) * mean if np.isfinite(mean) else m
                second = w * m2 + (1-w) * second if np.isfinite(second) else m2
                cdf = w * rank + (1-w) * cdf if np.isfinite(cdf) else rank
                sep_n, sep_players, sep_weight, sep_level = len(values), node['sep_players'], w, level
            if node['n_target']:
                w = self._weight(node['n_target'], node['target_players']) if np.isfinite(probability) else 1.
                rate = node['target_sum'] / node['n_target']
                probability = w * rate + (1-w) * probability if np.isfinite(probability) else rate
                target_n, target_players, target_weight, target_level = (node['n_target'],
                    node['target_players'], w, level)
        std = np.sqrt(max(0., second - mean ** 2)) if np.isfinite(mean) else np.nan
        z = (value - mean) / std if np.isfinite(value) and std > 1e-8 else np.nan
        return dict(context_sep_mean=mean, context_sep_std=std, separation_z=z,
            separation_percentile=100*cdf, expected_target_probability=probability,
            sep_bucket_n=sep_n, sep_bucket_players=sep_players, sep_bucket_weight=sep_weight,
            sep_reference_level=sep_level, target_bucket_n=target_n,
            target_bucket_players=target_players, target_bucket_weight=target_weight,
            target_reference_level=target_level)

    def transform(self, rows):
        scores = pd.DataFrame([self.score(r) for r in rows.to_dict('records')], index=rows.index)
        return pd.concat([rows, scores], axis=1)

    def bucket_table(self):
        rows = []
        for (level, key), node in self.nodes.items():
            v = node['values']
            effective = self.score({**dict(zip(LEVELS[level], key)), SEP:np.nan})
            rows.append(dict(level=level, **dict(zip(LEVELS[level], key)),
                context_sep_mean=effective['context_sep_mean'], context_sep_std=effective['context_sep_std'],
                expected_target_probability=effective['expected_target_probability'],
                n_routes=node['n_routes'], n_players=node['n_players'],
                n_separation=len(v), separation_players=node['sep_players'],
                raw_sep_mean=v.mean() if len(v) else np.nan,
                raw_sep_std=v.std() if len(v) else np.nan,
                **{f'raw_sep_p{q}':np.percentile(v, q) if len(v) else np.nan for q in [10,25,50,75,90]},
                n_target=node['n_target'], target_players=node['target_players'],
                raw_target_rate=node['target_sum']/node['n_target'] if node['n_target'] else np.nan,
                sep_child_weight=self._weight(len(v), node['sep_players']) if level else 1.,
                target_child_weight=self._weight(node['n_target'], node['target_players']) if level else 1.))
        return pd.DataFrame(rows)


def summarize(scored, min_routes=20):
    rows = []
    for scope, column in [('route', 'route_ran'), ('family', 'route_family')]:
        for (pid, name), g in scored.dropna(subset=[column]).groupby(['nfl_id', column]):
            known = g[g.target_observed.notna() & g.expected_target_probability.notna()]
            n_sep = int(g.separation_z.notna().sum())
            actual = known.target_observed.mean()
            expected = known.expected_target_probability.mean()
            rows.append(dict(nfl_id=pid, draft_year=int(g.draft_year.iloc[0]), scope=scope,
                route=name, n_routes=len(g), n_games=g.game_id.nunique(), n_separation=n_sep,
                n_target=len(known), raw_separation_mean=g[SEP].mean(),
                separation_z=g.separation_z.mean(), separation_percentile=g.separation_percentile.mean(),
                target_rate=actual, expected_target_probability=expected,
                target_rate_above_expected=actual-expected,
                separation_eligible=n_sep >= min_routes, target_eligible=len(known) >= min_routes))
    return pd.DataFrame(rows)


def save_arrays(summary, ids, routes, output, min_routes):
    """Receiver-aligned outcomes, masks and counts; Combine reps join by nfl_id later."""
    labels = [f'route:{r}' for r in sorted(routes)] + [f'family:{f}' for f in sorted(set(ROUTE_FAMILY.values()))]
    y = np.zeros((len(ids), len(labels), len(METRICS)), dtype=np.float32)
    mask = np.zeros_like(y, dtype=bool)
    counts = np.zeros_like(y, dtype=np.int32)
    pi, li = {p:i for i,p in enumerate(ids)}, {r:i for i,r in enumerate(labels)}
    for row in summary.to_dict('records'):
        i, j = pi[row['nfl_id']], li[f"{row['scope']}:{row['route']}"]
        for k, metric in enumerate(METRICS):
            n = row['n_separation'] if metric.startswith('separation_') else row['n_target']
            counts[i,j,k] = n
            valid = n >= min_routes and np.isfinite(row[metric])
            mask[i,j,k] = valid
            if valid:
                y[i,j,k] = row[metric]
    np.savez_compressed(output / 'rookie_targets.npz', nfl_ids=np.asarray(ids),
                        output_labels=np.asarray(labels), metric_names=np.asarray(METRICS),
                        y=y, mask=mask, counts=counts)


def write_output_guide(output, metadata):
    """Keep the explanation beside every freshly prepared output package."""
    arrays = np.load(output / 'rookie_targets.npz', allow_pickle=False)
    shape = tuple(arrays['y'].shape)
    arrays.close()
    intro = ("# Rookie context outputs\n\n"
             f"Current reference: **{metadata['reference_rows']:,} routes from "
             f"{metadata['reference_players']} WRs**, all available regular-season seasons pooled.\n\n"
             f"Scored population: **{metadata['rookie_rows']:,} rookie-season routes from "
             f"{metadata['rookie_players']} WRs**. Target array shape: **{shape}** "
             "(receivers, route/family outputs, metrics).\n\n")
    template = Path(__file__).with_name('rookie_output_guide.md').read_text()
    (output / 'README.md').write_text(intro + template)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root/'data')
    parser.add_argument('--games', type=Path)
    parser.add_argument('--output', type=Path, default=root/'outputs'/'rookie_context')
    parser.add_argument('--min-routes', type=int, default=20)
    parser.add_argument('--prior-routes', type=int, default=100)
    parser.add_argument('--prior-players', type=int, default=15)
    args = parser.parse_args(argv)
    if args.min_routes < 1:
        parser.error('--min-routes must be positive')
    games_path = args.games or next((p for p in [args.data_dir/'games.csv',
        root/'nfl-big-data-bowl-2027'/'games.csv'] if p.exists()), None)
    if games_path is None:
        parser.error('Provide games.csv with --games')
    play_path, players_path = args.data_dir/'wr_only_play.csv', args.data_dir/'wr_only_players.csv'
    play, players, games = pd.read_csv(play_path, low_memory=False), pd.read_csv(players_path), pd.read_csv(games_path)
    reference_rows = prepare_rows(play, players, games)
    reference = ContextReference(args.prior_routes, args.prior_players).fit(reference_rows)
    rookies = reference_rows[reference_rows.is_rookie_season].copy().reset_index(drop=True)
    if rookies.empty:
        raise ValueError('No rookie regular-season routes')
    scored = reference.transform(rookies)
    summary = summarize(scored, args.min_routes)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'metadata.json').exists():
        raise FileExistsError('Choose a fresh output directory to preserve prepared data')
    reference.bucket_table().to_csv(args.output/'context_buckets.csv', index=False)
    scored.to_csv(args.output/'rookie_routes.csv', index=False)
    summary.to_csv(args.output/'rookie_receiver_summary.csv', index=False)
    save_arrays(summary, sorted(players.nfl_id.tolist()), reference_rows.route_ran.unique(), args.output, args.min_routes)
    metadata = dict(reference_population='all regular-season WR cohort routes; all available seasons pooled',
        scoring_population='regular-season routes with season == draft_year',
        reference_includes_scored_receivers=True, evaluation_performed=False,
        target_probability='P(target=True | route, coverage, cushion); context baseline, not player-specific',
        percentile='0-100, hierarchical empirical mixture CDF with midrank ties',
        cushion_intervals='[0,2),[2,4),[4,6),[6,8),[8,10),[10,infinity); unknown separate',
        hierarchy=[list(level) for level in LEVELS], prior_routes=args.prior_routes,
        prior_players=args.prior_players, min_routes=args.min_routes,
        reference_rows=len(reference_rows), reference_players=int(reference_rows.nfl_id.nunique()),
        rookie_rows=len(rookies), rookie_players=int(rookies.nfl_id.nunique()),
        target_denominator='all routes with known target, irrespective of missing separation',
        target_tensor_shape='player, route/family output, metric; invalid targets zero with mask=False',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [play_path,players_path,games_path]},
        versions=dict(numpy=np.__version__,pandas=pd.__version__))
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    write_output_guide(args.output, metadata)
    print(json.dumps({k:metadata[k] for k in ['reference_rows','reference_players','rookie_rows','rookie_players']},indent=2))
    print('Prepared data:',args.output)


if __name__ == '__main__':
    main()
