"""Reproducible player-level evaluation for the numbered pipeline.

[AI-SCAFFOLD] AI-assisted implementation; retain in the course disclosure.
"""
import argparse
import hashlib
import platform
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scipy
import sklearn
import matplotlib
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

setup = import_module('01setup')

OFFICIAL = ['forty', 'three_cone', 'short_shuttle', 'vertical', 'broad_jump',
            'combine_height', 'combine_weight']


class FeatureScaler:
    """Training medians, fixed missing flags, and training standardization.

    Completely missing training columns receive zero, with an explicit missing flag.
    """
    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.
                                for c in x.T])
        z = self._impute(x)
        self.mean, self.std = z.mean(0), z.std(0)
        self.std[self.std < 1e-6] = 1.
        return self

    def _impute(self, x):
        missing = ~np.isfinite(x)
        return np.concatenate([np.where(missing, self.median, x), missing.astype(float)], axis=1)

    def transform(self, x):
        return ((self._impute(np.asarray(x, dtype=float)) - self.mean) / self.std).astype(np.float32)

    def state(self):
        return {k: getattr(self, k).tolist() for k in ['median', 'mean', 'std']}


def official_features(ids, results, players):
    if results['nfl_id'].duplicated().any() or players['nfl_id'].duplicated().any():
        raise ValueError('Expected one results/player row per nfl_id')
    a = results.set_index('nfl_id').reindex(ids)[OFFICIAL].to_numpy(dtype=float)
    pick = players.set_index('nfl_id').reindex(ids)[['draft_overall_pick']].to_numpy(dtype=float)
    return a, pick


def kinematic_features(ids, by_player, channels):
    """Mean per-rep summaries, separately by sprint/route; absent families stay missing."""
    names = ['peak_speed', 'time_to_90pct', 'peak_accel', 'hardest_braking', 'peak_sideways']
    rows = []
    for pid in ids:
        row = []
        for family in setup.FAMILIES:
            summaries = []
            for rep in by_player[pid]:
                if rep['family_id'] != setup.FAMILY_ID[family]:
                    continue
                x = rep['x']
                speed = x[:, channels.index('speed' if 'speed' in channels else 's')]
                duration = float(x[0, -1])
                # Derivatives are reconstructed from elapsed time for raw-channel ablations.
                accel = (x[:, channels.index('acc_tan')] if 'acc_tan' in channels else
                         np.gradient(speed, duration / (len(speed) - 1)))
                side = (x[:, channels.index('acc_cent')] if 'acc_cent' in channels else
                        x[:, channels.index('turn')] * np.pi / 180 * speed if 'turn' in channels else
                        speed * np.gradient(np.unwrap(np.arctan2(
                            x[:, channels.index('dir_sin')], x[:, channels.index('dir_cos')])),
                            duration / (len(speed) - 1)))
                reached = np.flatnonzero(speed >= 0.9 * speed.max())
                summaries.append([speed.max(), reached[0] / (len(speed) - 1) * duration,
                                  accel.max(), accel.min(), np.abs(side).max()])
            row.extend(np.mean(summaries, axis=0) if summaries else [np.nan] * len(names))
        rows.append(row)
    return np.asarray(rows), [f'{f}_{n}' for f in setup.FAMILIES for n in names]


def make_splits(years, cfg, mode):
    n = len(years)
    if mode == 'kfold':
        if not 2 <= cfg['n_folds'] <= n:
            raise ValueError('n_folds must be between 2 and the cohort size')
        outer = KFold(cfg['n_folds'], shuffle=True, random_state=setup.SEED).split(np.arange(n))
    elif mode == 'draft_class':
        if pd.isna(years).any() or len(np.unique(years)) < 2:
            raise ValueError('Draft-class evaluation requires at least two known classes')
        outer = [(np.flatnonzero(years != y), np.flatnonzero(years == y)) for y in np.unique(years)]
    else:
        raise ValueError(mode)
    folds = []
    for fold, (tr, te) in enumerate(outer):
        if len(tr) < 3:
            raise ValueError('Too few training players for inner validation')
        perm = np.random.default_rng(setup.SEED + fold).permutation(tr)
        nv = max(1, int(np.ceil(len(tr) * .15)))
        folds.append((tr, te, perm[nv:], perm[:nv]))
    return folds


def adjusted_targets(play, ids, training_ids, cfg, target_module):
    """Remove route-type means fitted on inner-train players, shared by every model.

    For a route unseen in training, use its training family mean; never use test outcomes.
    If the entire family is absent in training, fail with an actionable error.
    """
    p = play[play['nfl_id'].isin(training_ids)].dropna(subset=[setup.SEP_COL, 'route_ran']).copy()
    p = p[np.isfinite(p[setup.SEP_COL])].copy()
    p['family'] = p['route_ran'].map(setup.ROUTE_FAMILY)
    p = p.dropna(subset=['family'])
    means = p.groupby('route_ran')[setup.SEP_COL].mean().to_dict()
    fam_means = p.groupby('family')[setup.SEP_COL].mean().to_dict()
    fallback = []
    for route, family in setup.ROUTE_FAMILY.items():
        if route not in means:
            if family not in fam_means:
                raise ValueError(f'No training outcomes for family {family}; reduce folds or target threshold')
            means[route] = fam_means[family]
            fallback.append(route)
    targets = target_module.build_targets(play, ids, cfg, means)
    arrays = [np.stack([targets[pid][k] for pid in ids]) for k in range(3)]
    return (*arrays, {'route_means': means, 'fallback_routes': fallback})


def ridge_predict(x, tr, te, y, mask, alpha=10.):
    scaler = FeatureScaler().fit(x[tr])
    z = scaler.transform(x)
    pred = np.empty((len(te), y.shape[1]), dtype=np.float32)
    for k in range(y.shape[1]):
        train = tr[mask[tr, k] > 0]
        if len(train) < 2:
            raise ValueError(f'Fewer than two training targets for output {k}')
        pred[:, k] = Ridge(alpha=alpha, solver='svd').fit(z[train], y[train, k]).predict(z[te])
    return pred, scaler.state()


def neural_predict(cfg, ids, by_player, attrs, y, mask, weights, inner, val, test, fold, out,
                   label, seed_index):
    data = import_module('08dataset_and_batching')
    models = import_module('09model')
    train = import_module('10training')
    seed = setup.SEED + 100 * fold + seed_index
    setup.set_seed(seed)
    scaler = FeatureScaler().fit(attrs[inner])
    z = scaler.transform(attrs)
    scalers = data.fit_scalers([ids[i] for i in inner], by_player, z[inner])
    def loader(idx, shuffle=False):
        return DataLoader(data.PlayerDataset(idx, ids, by_player, z, y, mask, weights, scalers),
                          batch_size=cfg['batch_size'], shuffle=shuffle, collate_fn=data.collate)
    model = models.PlayerModel(by_player[ids[0]][0]['x'].shape[1],
                               len(import_module('05rep_tensors').DRILL_IDS), z.shape[1],
                               y.shape[1], cfg).to(setup.DEVICE)
    loss = train.fit(model, loader(inner, True), loader(val), cfg)
    pred = train.predict(model, loader(test))
    stem = out / f'{label}_fold{fold}_seed{seed_index}'
    torch.save(model.state_dict(), str(stem) + '.pt')
    state = dict(seed=seed, validation_loss=loss, attributes=scaler.state(),
                 channel_mean=scalers[0].tolist(), channel_std=scalers[1].tolist(),
                 attr_mean=scalers[2].tolist(), attr_std=scalers[3].tolist())
    Path(str(stem) + '_preprocessing.json').write_text(json.dumps(state, indent=2))
    # Evaluate a single held-out player at a time to preserve rep identity and discard padding queries.
    attention = []
    model.eval()
    if cfg['pool'] in ('weighted', 'attention'):
        with torch.no_grad():
            for j in test:
                b = next(iter(loader([j])))
                model(train.to_dev(b))
                w = model.pool.last_weights[0].cpu().numpy()
                if w.ndim == 2:
                    w = w.mean(axis=0)  # all queries are valid in this one-player batch
                for rep, weight in zip(by_player[ids[j]], w):
                    attention.append(dict(nfl_id=ids[j], event_id=rep['event_id'],
                                          drill_id=rep['drill_id'], weight=float(weight)))
        pd.DataFrame(attention).to_csv(str(stem) + '_pool_weights.csv', index=False)
    return pred


def intervals(predictions, y, mask, score, n_boot, seed=0):
    """Paired player bootstrap; identical resampled player indices for every model."""
    rng = np.random.default_rng(seed)
    point = {name: score(p, y, mask) for name, p in predictions.items()}
    samples = {name: [] for name in predictions}
    for _ in range(n_boot):
        ix = rng.integers(len(y), size=len(y))
        for name, p in predictions.items():
            samples[name].append(score(p[ix], y[ix], mask[ix])[['spearman', 'r2']].to_numpy())
    rows = []
    for name, table in point.items():
        for k, r in table.iterrows():
            for metric_idx, metric in enumerate(['spearman', 'r2']):
                vals = np.asarray(samples[name])[:, k, metric_idx] if n_boot else np.array([])
                valid = vals[np.isfinite(vals)]
                lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
                rows.append(dict(model=name, family=r['family'], n=r['n'], metric=metric,
                                 value=r[metric], ci_low=lo, ci_high=hi, bootstrap_valid=len(valid)))
        if name != 'A':
            for k, family in enumerate(setup.ROUTE_FAMS):
                for mi, metric in enumerate(['spearman', 'r2']):
                    vals = (np.asarray(samples[name])[:, k, mi] - np.asarray(samples['A'])[:, k, mi]
                            if n_boot else np.array([]))
                    valid = vals[np.isfinite(vals)]
                    lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
                    rows.append(dict(model=f'{name}-A', family=family, n=point[name].iloc[k]['n'],
                                     metric=metric, value=point[name].iloc[k][metric] - point['A'].iloc[k][metric],
                                     ci_low=lo, ci_high=hi, bootstrap_valid=len(valid)))
    return pd.DataFrame(rows)


def run_evaluation(cfg, output, mode='kfold', neural=False, ablations=False, n_boot=500):
    load = import_module('02load_data')
    repmod = import_module('05rep_tensors')
    targets = import_module('06targets')
    baseline = import_module('07baselines')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists():
        raise FileExistsError(f'{output} already has a run; choose a new --output directory')
    all_reps = repmod.build_reps(import_module('04cleaning_and_kinematics').wr_trk,
                                cfg['T'], cfg['channels'])
    raw_targets = targets.build_targets(load.play, load.wr_ids, cfg)
    ids, by_player = baseline.make_player_table(all_reps, raw_targets, cfg=cfg)
    if not ids:
        raise ValueError('No eligible players')
    if cfg.get('require_all_families', False):
        ids = [pid for pid in ids if {r['family_id'] for r in by_player[pid]} == set(setup.FAMILY_ID.values())]
    ids = sorted(ids)
    if not ids:
        raise ValueError('No eligible players after matching drill families')
    # Save a reproducible visual check of the rebuilt, unstandardized reps.
    fig = repmod.plot_reps_per_family(all_reps, channels=cfg['channels'])
    fig.savefig(output / 'rep_traces.png', dpi=160, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    a, pick = official_features(ids, load.results, load.players)
    b, b_names = kinematic_features(ids, by_player, cfg['channels'])
    years = load.results.set_index('nfl_id').loc[ids, 'draft_year'].to_numpy()
    folds = make_splits(years, cfg, mode)
    config = dict(cfg=cfg, mode=mode, seed=setup.SEED, neural=neural, ablations=ablations,
                  bootstrap=n_boot, official_columns=OFFICIAL, kinematic_columns=b_names,
                  route_families=setup.ROUTE_FAMS, drill_ids=repmod.DRILL_IDS,
                  target='route-type residual; means fitted on inner-train players',
                  regular_season_only=load.REGULAR_SEASON_ONLY,
                  versions={m.__name__:m.__version__ for m in [np,pd,torch,scipy,sklearn,matplotlib]},
                  python=platform.python_version(), device=str(setup.DEVICE),
                  source_sha256={str(p.relative_to(setup.DATA_DIR.parent)):
                      hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [*load.WR_FILES.values(), load.GAMES_PATH]})
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    pd.DataFrame(dict(nfl_id=ids, draft_year=years)).to_csv(output / 'cohort.csv', index=False)
    split_rows = []
    for fold, (tr, te, inner, val) in enumerate(folds):
        for role, indices in [('train', inner), ('validation', val), ('test', te)]:
            split_rows.extend(dict(fold=fold, nfl_id=ids[i], role=role) for i in indices)
    pd.DataFrame(split_rows).to_csv(output / 'folds.csv', index=False)
    shape = (len(ids), len(setup.ROUTE_FAMS))
    features = dict(A=a, B=b, draft_only=pick, A_draft=np.concatenate([a, pick], axis=1))
    variants = {}
    if neural:
        variants = {'C':dict(use_attrs=False), 'C_A':dict(use_attrs=True)}
        if ablations:
            variants.update({f'C_A_{pool}':dict(pool=pool, use_attrs=True)
                             for pool in ['max', 'weighted', 'attention']})
            variants.update({f'C_A_{family}_only':dict(use_attrs=True, rep_family=family)
                             for family in setup.FAMILIES})
    config['model_variants'] = variants
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    predictions = {name:np.full(shape, np.nan, dtype=np.float32)
                   for name in ['mean', *features, *variants]}
    seed_predictions = {name:np.full((cfg['n_seeds'], *shape), np.nan, dtype=np.float32)
                        for name in variants}
    y_oof, mask_oof, count_oof = (np.zeros(shape, np.float32) for _ in range(3))
    fold_states = []
    for fold, (tr, te, inner, val) in enumerate(folds):
        print(f'Evaluation fold {fold + 1}/{len(folds)}: {len(tr)} train, {len(te)} test', flush=True)
        y, mask, weights, adjustment = adjusted_targets(load.play, ids, [ids[i] for i in inner], cfg, targets)
        if (mask[inner].sum(axis=0) < 2).any():
            raise ValueError(f'Fold {fold}: fewer than two inner-training targets in a family')
        y_oof[te], mask_oof[te], count_oof[te] = y[te], mask[te], weights[te] ** 2
        for k in range(shape[1]):
            available = tr[mask[tr, k] > 0]
            if len(available) < 2:
                raise ValueError(f'Fold {fold}: insufficient training targets for {setup.ROUTE_FAMS[k]}')
            predictions['mean'][te, k] = y[available, k].mean()
        state = dict(fold=fold, adjustment=adjustment, scalers={})
        for name, x in features.items():
            predictions[name][te], state['scalers'][name] = ridge_predict(x, tr, te, y, mask)
        for name, override in variants.items():
            local_cfg = {**cfg, **override}
            selected = by_player
            if 'rep_family' in override:
                fid = setup.FAMILY_ID[override['rep_family']]
                # Preserve the comparison cohort; a missing family is an explicit failure.
                selected = {pid:[r for r in by_player[pid] if r['family_id'] == fid] for pid in ids}
                missing = [pid for pid in ids if not selected[pid]]
                if missing:
                    raise ValueError(f'{name}: {len(missing)} players lack reps; use a matched cohort with --require-all-families')
            for seed in range(cfg['n_seeds']):
                seed_predictions[name][seed, te] = neural_predict(local_cfg, ids, selected, a, y, mask,
                    weights, inner, val, te, fold, output, name, seed)
            predictions[name][te] = seed_predictions[name][:, te].mean(axis=0)
        fold_states.append(state)
        (output / 'fold_preprocessing.json').write_text(json.dumps(fold_states, indent=2))
        np.savez_compressed(output / 'checkpoint.npz', y=y_oof, mask=mask_oof, **predictions)
    for name, pred in predictions.items():
        table = pd.DataFrame(dict(nfl_id=ids))
        for k, family in enumerate(setup.ROUTE_FAMS):
            table[f'{family}_prediction'] = pred[:, k]
            table[f'{family}_target'] = y_oof[:, k]
            table[f'{family}_valid'] = mask_oof[:, k]
            table[f'{family}_route_count'] = np.rint(count_oof[:, k]).astype(int)
        table.to_csv(output / f'{name}_oof.csv', index=False)
    for name, pred in seed_predictions.items():
        np.save(output / f'{name}_seed_oof.npy', pred)
    metrics = intervals(predictions, y_oof, mask_oof, baseline.score, n_boot)
    metrics.to_csv(output / 'metrics.csv', index=False)
    seed_metrics = []
    for name, preds in seed_predictions.items():
        for seed, pred in enumerate(preds):
            table = baseline.score(pred, y_oof, mask_oof)
            table.insert(0, 'seed_index', seed)
            table.insert(0, 'model', name)
            seed_metrics.append(table)
    if seed_metrics:
        pd.concat(seed_metrics).to_csv(output / 'seed_metrics.csv', index=False)
    interpretation(predictions, y_oof, mask_oof, count_oof, ids, pick, a, metrics, output)
    print(metrics.to_string(index=False))
    return metrics


def interpretation(predictions, y, mask, counts, ids, pick, attrs, metrics, output):
    """Held-out residual tables and figures, with draft slot and route eligibility."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for name, pred in predictions.items():
        for i, pid in enumerate(ids):
            for k, family in enumerate(setup.ROUTE_FAMS):
                if mask[i, k]:
                    rows.append(dict(model=name, nfl_id=pid, family=family, draft_pick=pick[i, 0],
                                     route_count=int(round(counts[i, k])), missing_attributes=int((~np.isfinite(attrs[i])).sum()),
                                     target=y[i, k], prediction=pred[i, k], residual=y[i, k]-pred[i, k]))
    residuals = pd.DataFrame(rows)
    residuals.to_csv(output / 'residuals.csv', index=False)
    compare = 'C_A' if 'C_A' in predictions else 'B'
    fig, axes = plt.subplots(1, len(setup.ROUTE_FAMS), figsize=(12, 4), squeeze=False)
    for k, (family, ax) in enumerate(zip(setup.ROUTE_FAMS, axes[0])):
        valid = mask[:, k] > 0
        ax.scatter(predictions['A'][valid, k], predictions[compare][valid, k], s=20)
        ax.set(title=family, xlabel='A OOF prediction', ylabel=f'{compare} OOF prediction')
    fig.tight_layout()
    fig.savefig(output / 'prediction_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(len(setup.ROUTE_FAMS), 3, figsize=(12, 9), squeeze=False)
    for k, family in enumerate(setup.ROUTE_FAMS):
        subset = residuals[(residuals['model'] == compare) & (residuals['family'] == family)]
        for ax, column in zip(axes[k], ['draft_pick', 'route_count', 'missing_attributes']):
            ax.scatter(subset[column], subset['residual'], s=16, alpha=.7)
            ax.axhline(0, color='gray', lw=.6)
            ax.set(xlabel=column, ylabel=f'{family}: target - prediction')
    fig.suptitle(f'{compare}: held-out residuals')
    fig.tight_layout()
    fig.savefig(output / 'residual_diagnostics.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metric, ax in zip(['spearman', 'r2'], axes):
        selected = metrics[(metrics['metric'] == metric) & ~metrics['model'].str.contains('-A', regex=False)]
        labels = [f'{r.model}/{r.family}' for r in selected.itertuples()]
        positions = np.arange(len(selected))
        ax.scatter(positions, selected['value'], s=16)
        ax.vlines(positions, selected['ci_low'], selected['ci_high'], alpha=.5)
        ax.axhline(0, color='gray', lw=.6)
        ax.set(title=f'{metric}: player-bootstrap 95% intervals', xticks=positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(output / 'model_comparison.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs' / 'evaluation')
    parser.add_argument('--neural', action='store_true', help='Run C and C+A alongside baselines')
    parser.add_argument('--ablations', action='store_true', help='Also run pooling and family-only variants')
    parser.add_argument('--require-all-families', action='store_true', help='Restrict every model to a matched family-complete cohort')
    parser.add_argument('--mode', choices=['kfold', 'draft_class'], default='kfold')
    parser.add_argument('--folds', type=int, default=setup.CFG['n_folds'])
    parser.add_argument('--epochs', type=int, default=setup.CFG['epochs'])
    parser.add_argument('--seeds', type=int, default=setup.CFG['n_seeds'])
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--min-routes', type=int, default=setup.MIN_ROUTES)
    parser.add_argument('--min-route-reps', type=int, default=setup.MIN_ROUTE_REPS)
    parser.add_argument('--timesteps', type=int, default=setup.CFG['T'])
    parser.add_argument('--channels', choices=setup.CHANNEL_SETS, default='derived')
    args = parser.parse_args(argv)
    if min(args.epochs, args.seeds, args.min_routes, args.min_route_reps) < 1 or args.bootstrap < 0:
        parser.error('epochs, seeds and thresholds must be positive; bootstrap must be nonnegative')
    if args.ablations and not args.neural:
        parser.error('--ablations requires --neural')
    cfg = {**setup.CFG, 'epochs':args.epochs, 'n_seeds':args.seeds, 'n_folds':args.folds,
           'min_routes_per_family':args.min_routes, 'min_reps_per_player':args.min_route_reps,
           'T':args.timesteps, 'channels':setup.CHANNEL_SETS[args.channels],
           'require_all_families':args.require_all_families}
    # Small LSTMs are much faster without CPU thread oversubscription.
    torch.set_num_threads(1)
    run_evaluation(cfg, args.output, args.mode, args.neural, args.ablations, args.bootstrap)


if __name__ == '__main__':
    main()
