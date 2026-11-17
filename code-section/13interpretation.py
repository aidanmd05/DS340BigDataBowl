"""Held-out interpretation artifacts are produced by evaluation.interpretation.

[AI-SCAFFOLD] AI-assisted implementation.

Each evaluation saves residuals with route volume, draft slot and attribute opt-outs;
A versus C+A predictions (A versus B for baseline-only runs); and model metrics
with player-bootstrap intervals. Pooling experiments save held-out rep weights.
Weights are shared across route outputs and are descriptive, not causal importance.
Family-only ablations use a matched cohort and identical folds for every model.
"""
from evaluation import interpretation
