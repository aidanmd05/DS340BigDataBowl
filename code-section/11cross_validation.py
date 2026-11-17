"""Player-grouped cross-validation through the shared evaluation harness.

[AI-SCAFFOLD] AI-assisted implementation; see evaluation.py for fold logic.
"""
from importlib import import_module
from pathlib import Path
import pandas as pd


def run_cv(cfg, attrs=None, mode="kfold", output=None):
    """Run shared baselines + neural CV and return the C+A OOF matrix.

    Attributes are constructed from official columns by the harness; callers should
    omit attrs. Supply a new output directory for each run to preserve artifacts.
    """
    if attrs is not None:
        raise ValueError("Omit attrs: evaluation constructs aligned official attributes with fold-local imputation")
    from evaluation import run_evaluation
    destination = Path(output) if output else Path(__file__).resolve().parents[1] / 'outputs' / 'neural'
    run_evaluation(cfg, destination, mode=mode, neural=True)
    table = pd.read_csv(destination / 'C_A_oof.csv')
    families = import_module('01setup').ROUTE_FAMS
    return table[[f'{f}_prediction' for f in families]].to_numpy()
