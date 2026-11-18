"""[AI-SCAFFOLD] One seeded player holdout run for rookie quality and volume.

Train-only context references/scalers, leave-one-player-out training references,
validation early stopping, and one untouched test set. No tuning or cross-validation.
"""
import argparse
import copy
import hashlib
from importlib import import_module
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader, Dataset

from prepare_rookie_context import ContextReference, prepare_rows, summarize


def player_split(n, seed):
    if n < 10:
        raise ValueError('At least ten players are required for a 60/20/20 holdout')
    order = np.random.default_rng(seed).permutation(n)
    size = max(1, round(.2*n))
    return order[2*size:], order[size:2*size], order[:size]


class OutcomeScaler:
    """Count log1p followed by per-output train-only standardization."""
    def fit(self, y, mask, train):
        z = self.transform_units(y)
        self.mean = np.zeros(y.shape[1:])
        self.std = np.ones(y.shape[1:])
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                observed = train[mask[train,j,k]]
                if len(observed) < 2:
                    raise ValueError(f'Insufficient training targets for family {j}, metric {k}')
                values = z[observed,j,k]
                self.mean[j,k] = values.mean()
                self.std[j,k] = max(values.std(),1e-6)
        return self

    @staticmethod
    def transform_units(y):
        z = np.asarray(y,dtype=float).copy()
        if (z[:,:,1]<0).any():
            raise ValueError('Route counts must be nonnegative')
        z[:,:,1] = np.log1p(z[:,:,1])
        return z

    def transform(self, y, mask):
        z = (self.transform_units(y)-self.mean)/self.std
        return np.where(mask,z,0).astype(np.float32)

    def inverse(self, z):
        y = np.asarray(z,dtype=float)*self.std+self.mean
        y[:,:,1] = np.maximum(0,np.expm1(y[:,:,1]))
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite inverse-transformed predictions')
        return y


def build_split_targets(rows, ids, train, families, min_routes=20, prior_routes=100, prior_players=15):
    """Keep volume labels for observed players; rebuild separation without held-out references."""
    train_ids = set(ids[train])
    reference_rows = rows[rows.nfl_id.isin(train_ids)]
    reference = ContextReference(prior_routes,prior_players).fit(reference_rows)
    y = np.zeros((len(ids),len(families),2),dtype=np.float32)
    mask = np.zeros_like(y,dtype=bool)
    counts = np.zeros_like(y,dtype=np.int32)
    for i,pid in enumerate(ids):
        rookie = rows[rows.nfl_id.eq(pid) & rows.is_rookie_season]
        if rookie.empty:
            continue
        local_ref = reference
        if pid in train_ids:
            local_ref = ContextReference(prior_routes,prior_players).fit(
                reference_rows[reference_rows.nfl_id.ne(pid)])
        scored = local_ref.transform(rookie)
        table = summarize(scored,min_routes)
        family_table = table[table.scope.eq('family')].set_index('route')
        for j,family in enumerate(families):
            volume = int(rookie.route_family.eq(family).sum())
            y[i,j,1], mask[i,j,1], counts[i,j,1] = volume, True, volume
            if family in family_table.index:
                row = family_table.loc[family]
                counts[i,j,0] = int(row.n_separation)
                if row.separation_eligible and np.isfinite(row.separation_z):
                    y[i,j,0], mask[i,j,0] = row.separation_z, True
        if (i+1)%20==0:
            print(f'Prepared split-specific outcomes for {i+1}/{len(ids)} players',flush=True)
    return y,mask,counts,reference


def fit_input_scaler(package, train):
    x = package['x'][np.isin(package['rep_player_index'],train)]
    return x.mean(axis=(0,1)), np.maximum(x.std(axis=(0,1)),1e-6)


class JoinedDataset(Dataset):
    def __init__(self, package, idx, y, mask, channel_mean, channel_std):
        self.p,self.idx,self.y,self.mask = package,idx,y,mask
        self.mean,self.std = channel_mean,channel_std
    def __len__(self):
        return len(self.idx)
    def __getitem__(self,index):
        i = self.idx[index]
        start,end = self.p['player_rep_offsets'][i:i+2]
        return dict(x=torch.tensor((self.p['x'][start:end]-self.mean)/self.std,dtype=torch.float32),
            drill=torch.tensor(self.p['rep_drill_ids'][start:end],dtype=torch.long),
            fam=torch.tensor(self.p['rep_family_ids'][start:end],dtype=torch.long),
            attrs=torch.empty(0),y=torch.tensor(self.y[i].reshape(-1)),
            ym=torch.tensor(self.mask[i].reshape(-1),dtype=torch.float32),w=torch.ones(6))


def balanced_loss(pred,y,mask):
    n = mask.sum(0)
    valid = n>0
    if not valid.any():
        return None
    return ((((pred-y)**2)*mask).sum(0)[valid]/n[valid]).mean()


def move(batch,device):
    return {k:v.to(device) for k,v in batch.items()}


@torch.no_grad()
def epoch_loss(model,loader,device):
    model.eval()
    num,den = torch.zeros(6,device=device),torch.zeros(6,device=device)
    for batch in loader:
        b = move(batch,device)
        num += (((model(b)-b['y'])**2)*b['ym']).sum(0)
        den += b['ym'].sum(0)
    valid = den>0
    if not valid.any():
        raise ValueError('Partition has no observed labels')
    return (num[valid]/den[valid]).mean().item()


def metric_table(actual,pred,mask,baseline,families):
    rows = []
    for j,family in enumerate(families):
        for k,metric in enumerate(['separation_z','route_count']):
            m = mask[:,j,k]
            y,p,b = actual[m,j,k],pred[m,j,k],baseline[m,j,k]
            def metrics(v):
                if not len(y):
                    return dict(mae=np.nan,rmse=np.nan,r2=np.nan,spearman=np.nan)
                total = ((y-y.mean())**2).sum()
                return dict(mae=float(np.abs(y-v).mean()),rmse=float(np.sqrt(((y-v)**2).mean())),
                    r2=float(1-((y-v)**2).sum()/total) if total>0 else np.nan,
                    spearman=float(spearmanr(y,v).statistic) if len(y)>1 and np.ptp(y)>0 and np.ptp(v)>0 else np.nan)
            rows.append(dict(family=family,metric=metric,n=int(m.sum()),**metrics(p),
                             **{f'baseline_{key}':value for key,value in metrics(b).items()}))
    return pd.DataFrame(rows)


def plots(out,history,predictions,families):
    fig,ax = plt.subplots(figsize=(7,4))
    for column,label in [('train_loss','Train'),('validation_loss','Validation')]:
        ax.plot(history.epoch,history[column],label=label)
    ax.set(xlabel='Epoch',ylabel='Balanced masked MSE (standardized targets)',title='One LSTM run: learning curve')
    ax.legend();fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=160);plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(12,7),squeeze=False)
    for j,family in enumerate(families):
        data = predictions[predictions.route_family.eq(family)]
        for k,metric in enumerate(['separation_z','route_count']):
            valid = data[f'{metric}_valid']
            actual = data.loc[valid,f'{metric}_actual']
            predicted = data.loc[valid,f'{metric}_predicted']
            ax=axes[k,j];ax.scatter(actual,predicted,s=28,alpha=.8)
            if len(actual):
                limits=[min(actual.min(),predicted.min()),max(actual.max(),predicted.max())]
                ax.plot(limits,limits,color='gray',linestyle=':',label='Perfect prediction')
                ax.axhline(data.loc[valid,f'{metric}_baseline'].iloc[0],color='orange',linestyle='--',label='Train mean')
            ax.set(title=f'{family}: {metric} (n={len(actual)})',xlabel='Observed',ylabel='Predicted')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Held-out players only; counts are recorded routes, not verified season totals')
    fig.tight_layout();fig.savefig(out/'heldout_predictions.png',dpi=160);plt.close(fig)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=root/'outputs/lstm_inputs/joined_inputs_outcomes.npz')
    parser.add_argument('--output',type=Path,default=root/'outputs/lstm_single_run')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=15)
    args=parser.parse_args(argv)
    if min(args.epochs,args.patience)<1:
        parser.error('Epochs and patience must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose a fresh output directory')
    torch.set_num_threads(1)
    setup=import_module('01setup');setup.set_seed(args.seed)
    model_module=import_module('09model');collate=import_module('08dataset_and_batching').collate
    with np.load(args.inputs,allow_pickle=False) as loaded:
        package={key:loaded[key] for key in loaded.files}
    join_meta_path=args.inputs.with_name('metadata.json')
    join_meta=json.loads(join_meta_path.read_text())
    ids=package['nfl_ids'];families=setup.ROUTE_FAMS
    if package['prediction_output_labels'].tolist()!=[f'family:{f}' for f in families] or package['prediction_metric_names'].tolist()!=['separation_z','route_count']:
        raise ValueError('Joined prediction axes do not match the requested family outcomes')
    train,val,test=player_split(len(ids),args.seed)
    players_path=setup.DATA_DIR/'wr_only_players.csv';play_path=setup.DATA_DIR/'wr_only_play.csv'
    games_path=next(p for p in [setup.DATA_DIR/'games.csv',root/'nfl-big-data-bowl-2027/games.csv'] if p.exists())
    players=pd.read_csv(players_path)
    rows=prepare_rows(pd.read_csv(play_path,low_memory=False),players,pd.read_csv(games_path))
    target_meta_path=root/'outputs/rookie_context/metadata.json'
    target_meta=json.loads(target_meta_path.read_text())
    y,mask,counts,reference=build_split_targets(rows,ids,train,families,
        min_routes=target_meta['min_routes'],prior_routes=target_meta['prior_routes'],prior_players=target_meta['prior_players'])
    # Volume is reference-independent and must exactly match the prepared join.
    np.testing.assert_array_equal(y[:,:,1],package['prediction_y'][:,:,1])
    np.testing.assert_array_equal(mask[:,:,1],package['prediction_mask'][:,:,1])
    scaler=OutcomeScaler().fit(y,mask,train);z=scaler.transform(y,mask)
    ch_mean,ch_std=fit_input_scaler(package,train)
    cfg={**setup.CFG,'use_attrs':False,'pool':'mean','epochs':args.epochs,'patience':args.patience}
    def loader(indices,shuffle=False):
        return DataLoader(JoinedDataset(package,indices,z,mask,ch_mean,ch_std),
                          batch_size=cfg['batch_size'],shuffle=shuffle,collate_fn=collate)
    train_dl,val_dl,test_dl=loader(train,True),loader(val),loader(test)
    device=setup.DEVICE
    model=model_module.PlayerModel(package['x'].shape[-1],len(join_meta['drill_ids']),0,6,cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    best,best_epoch,best_state,bad=float('inf'),0,None,0
    history=[]
    print(f'Players: {len(train)} train / {len(val)} validation / {len(test)} test; device={device}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        for batch in train_dl:
            b=move(batch,device);opt.zero_grad()
            loss=balanced_loss(model(b),b['y'],b['ym'])
            if loss is None:
                continue
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        training=epoch_loss(model,loader(train),device)
        validation=epoch_loss(model,val_dl,device)
        if not np.isfinite([training,validation]).all():
            raise ValueError('Nonfinite training/validation loss')
        history.append(dict(epoch=epoch,train_loss=training,validation_loss=validation))
        if validation<best-1e-5:
            best,best_epoch,bad=validation,epoch,0
            best_state=copy.deepcopy(model.state_dict())
        else:
            bad+=1
        if epoch==1 or epoch%10==0:
            print(f'Epoch {epoch}: train={training:.4f}, validation={validation:.4f}',flush=True)
        if bad>=args.patience:
            break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        predicted_z=torch.cat([model(move(b,device)).cpu() for b in test_dl]).numpy().reshape(-1,3,2)
    predicted=scaler.inverse(predicted_z)
    baseline=np.zeros_like(predicted)
    for j in range(3):
        for k in range(2):
            baseline[:,j,k]=y[train[mask[train,j,k]],j,k].mean()
    metrics=metric_table(y[test],predicted,mask[test],baseline,families)
    names=players.set_index('nfl_id').display_name
    prediction_rows=[]
    for i,idx in enumerate(test):
        for j,family in enumerate(families):
            row=dict(nfl_id=int(ids[idx]),display_name=names.loc[ids[idx]],route_family=family)
            for k,metric in enumerate(['separation_z','route_count']):
                valid=bool(mask[idx,j,k])
                row.update({f'{metric}_actual':float(y[idx,j,k]) if valid else np.nan,
                    f'{metric}_predicted':float(predicted[i,j,k]),f'{metric}_valid':valid,
                    f'{metric}_baseline':float(baseline[i,j,k])})
            prediction_rows.append(row)
    predictions=pd.DataFrame(prediction_rows);history=pd.DataFrame(history)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out/'metrics.csv',index=False);predictions.to_csv(out/'test_predictions.csv',index=False)
    history.to_csv(out/'training_history.csv',index=False)
    role=np.full(len(ids),'train',dtype='<U10');role[val]='validation';role[test]='test'
    pd.DataFrame(dict(nfl_id=ids,role=role,has_count_label=mask[:,:,1].any(1),
        has_separation_label=mask[:,:,0].any(1))).to_csv(out/'player_split.csv',index=False)
    np.savez_compressed(out/'split_targets.npz',nfl_ids=ids,y=y,mask=mask,counts=counts,
        output_labels=np.asarray(families),metric_names=np.asarray(['separation_z','route_count']))
    np.savez_compressed(out/'preprocessing.npz',channel_mean=ch_mean,channel_std=ch_std,
        outcome_mean=scaler.mean,outcome_std=scaler.std)
    reference.bucket_table().to_csv(out/'training_reference_buckets.csv',index=False)
    torch.save(model.state_dict(),out/'model.pt')
    sources=[args.inputs,join_meta_path,players_path,play_path,games_path,target_meta_path,Path(__file__),
             Path(__file__).with_name('09model.py'),Path(__file__).with_name('prepare_rookie_context.py')]
    summary=dict(seed=args.seed,split_sizes=dict(train=len(train),validation=len(val),test=len(test)),
        best_epoch=best_epoch,epochs_run=len(history),best_validation_loss=best,
        config=cfg,device=str(device),baseline='Per-output arithmetic mean of observed training targets in original units',
        loss='Equal-output masked MSE on train-standardized separation z and log1p counts',
        context_reference='Training players only, all their regular-season seasons; leave-one-player-out for each training target',
        target_scaler='Fitted on observed training labels only',input_scaler='Fitted on training-player reps only',
        count_caveat='Recorded rookie routes only; no-record players masked; full-season coverage unknown',
        interpretation='One exploratory random-player holdout, not a draft-class or chronological test; no hyperparameter sweep',
        prediction_family_order=families,prediction_metric_order=['separation_z','route_count'],
        test_valid_counts=mask[test].sum(0).tolist(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        versions=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__))
    (out/'run.json').write_text(json.dumps(summary,indent=2))
    plots(out,history,predictions,families)
    (out/'README.md').write_text(f'''# One LSTM training and evaluation run

Seed {args.seed}; {len(train)} training / {len(val)} validation / {len(test)} test
players. Mean pooling, shared 32-unit LSTM, no official attributes. Best epoch
{best_epoch} of {len(history)} epochs run; selected solely by validation loss.

- `metrics.csv`: held-out per-family/metric sample size, MAE, RMSE, R² and Spearman;
  baseline columns use the training arithmetic mean. Smaller errors are better;
  negative R² means worse than the test-set mean. Baseline Spearman is undefined
  for constant predictions. Counts are evaluated in raw recorded-route units.
- `test_predictions.csv`: named test receivers, actual values, predictions, masks
  and baseline predictions. Missing actual values are blank and excluded from metrics.
- `learning_curve.png`, `training_history.csv`: train and validation losses in
  standardized/log-count units, not raw count error. No test loss drove stopping.
- `heldout_predictions.png`: observed versus predicted outcomes; gray dotted line
  is perfect agreement; orange dashed line is the training-mean baseline.
- `player_split.csv`: each receiver's role and availability of outcome labels.
- `split_targets.npz`: rebuilt targets/masks/counts, aligned to nfl_ids; held-out
  separation scores differ from full-population descriptive scores by design.
- `training_reference_buckets.csv`: context distributions fitted on training
  players only. Training-player scores use references refitted without that player.
- `model.pt`, `preprocessing.npz`: best checkpoint and train-only scaling parameters.
- `run.json`: settings, source hashes, split sizes and interpretation limits.

Count targets have no 20-route minimum; separation retains that minimum. Players
without recorded rookie routes remain in the split and receive predictions, but
have no scored labels. All input players contribute to training-only input scaling;
only observed targets contribute to loss. Counts use log1p before training-only
standardization and are converted back and clipped at zero for reporting. Each
family/metric has equal loss weight when present. These results come from one
small random-player split, not cross-validation or prospective rookie-class testing.
''')
    print(f'Best epoch={best_epoch}; results={out}',flush=True)
    print(metrics.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
