# One LSTM training and evaluation run

Seed 0; 64 training / 21 validation / 21 test
players. Mean pooling, shared 32-unit LSTM, no official attributes. Best epoch
51 of 66 epochs run; selected solely by validation loss.

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
