"""Controlled pooling and drill-family comparisons on shared player folds.

[AI-SCAFFOLD] AI-assisted implementation.
"""
from importlib import import_module
from pathlib import Path


def run_experiments(mode="kfold", output=None, cfg=None):
    """Require both drill families for all models, including the baselines.

    Freshly rebuild reps, targets and folds for every invocation. Experiment and
    seed predictions are persisted by evaluation.py, along with paired intervals.
    """
    from evaluation import run_evaluation
    cfg = dict(import_module('01setup').CFG if cfg is None else cfg)
    cfg['require_all_families'] = True
    destination = Path(output) if output else Path(__file__).resolve().parents[1] / 'outputs' / 'ablations'
    return run_evaluation(cfg, destination, mode, neural=True, ablations=True)
