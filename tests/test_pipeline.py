"""Behavior checks for leakage boundaries and the evaluation contract."""
import contextlib
from importlib import import_module
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code-section'))
from evaluation import FeatureScaler, make_splits, adjusted_targets


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Existing notebook-style audit executes on import; suppress routine tables.
        with contextlib.redirect_stdout(io.StringIO()):
            cls.setup = import_module('01setup')
            cls.reps = import_module('05rep_tensors')
            cls.clean = import_module('04cleaning_and_kinematics')
            cls.targets = import_module('06targets')
            cls.baseline = import_module('07baselines')
            cls.model = import_module('09model')
            cls.train = import_module('10training')
        torch.set_num_threads(1)

    def test_feature_scaler_training_only_and_all_missing(self):
        train = np.array([[1., np.nan], [3., np.nan]])
        scaler = FeatureScaler().fit(train)
        before = scaler.state()
        out = scaler.transform(np.array([[99999., 7.], [np.nan, np.nan]]))
        self.assertEqual(before, scaler.state())
        self.assertEqual(out.shape, (2, 4))
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(scaler.median.tolist(), [2., 0.])
        self.assertEqual(out[1, 0], 0.)

    def test_player_folds_disjoint_and_complete(self):
        years = np.repeat([2023, 2024, 2025], 10)
        for mode in ['kfold', 'draft_class']:
            folds = make_splits(years, {**self.setup.CFG, 'n_folds':3}, mode)
            tested = []
            for outer, test, inner, val in folds:
                self.assertFalse(set(outer) & set(test))
                self.assertFalse(set(inner) & set(val))
                self.assertEqual(set(outer), set(inner) | set(val))
                tested.extend(test)
                if mode == 'draft_class':
                    self.assertFalse(set(years[outer]) & set(years[test]))
            self.assertEqual(sorted(tested), list(range(len(years))))

    def test_target_adjustment_never_fits_test_outcomes(self):
        rows = []
        for pid in [1, 2, 3]:
            for route in ['GO', 'OUT', 'HITCH']:
                for _ in range(2):
                    rows.append(dict(nfl_id=pid, route_ran=route,
                                     separation_at_pass_forward=float(pid)))
        play = pd.DataFrame(rows)
        cfg = {**self.setup.CFG, 'min_routes_per_family':2}
        y, mask, w, fitted = adjusted_targets(play, [1, 2, 3], [1, 2], cfg, self.targets)
        changed = play.copy()
        changed.loc[changed.nfl_id == 3, self.setup.SEP_COL] += 1000
        y2, _, _, fitted2 = adjusted_targets(changed, [1, 2, 3], [1, 2], cfg, self.targets)
        self.assertEqual(fitted, fitted2)
        np.testing.assert_allclose(y[:2], y2[:2])
        np.testing.assert_allclose(y2[2] - y[2], 1000.)
        self.assertTrue((mask == 1).all())
        np.testing.assert_allclose(w, np.sqrt(2))
        self.assertEqual(self.targets.build_targets(play, [1], {**cfg, 'min_routes_per_family':3}), {})

    def test_route_eligibility_excludes_sprints(self):
        route = self.setup.FAMILY_ID['route']
        sprint = self.setup.FAMILY_ID['sprint']
        reps = [dict(nfl_id=1, family_id=sprint) for _ in range(10)]
        reps += [dict(nfl_id=1, family_id=route) for _ in range(2)]
        reps += [dict(nfl_id=2, family_id=route) for _ in range(3)]
        ids, _ = self.baseline.make_player_table(reps, {1:None, 2:None}, cfg=self.setup.CFG)
        self.assertEqual(ids, [2])

    def test_qc_rejects_duplicate_times_and_bad_channels(self):
        base = pd.DataFrame(dict(event_id=[1]*20, nfl_id=[1]*20,
            drill_type=['SKILL_DRILLS_WR']*20, time=pd.date_range('2023-01-01', periods=20, freq='100ms'),
            s=[2.]*20, dir=[30.]*20, a=[1.]*20))
        base.loc[5, 'time'] = base.loc[4, 'time']
        d = self.clean.add_kinematics(base)
        self.assertTrue(self.clean.attempt_qc(d).iloc[0]['bad_time'])
        d.loc[0, 'acc_cent'] = np.inf
        self.assertTrue(self.clean.attempt_qc(d).iloc[0]['bad_channels'])

    def test_resampling_uses_sorted_actual_time(self):
        drill = next(iter(self.reps.DRILL_IDS))
        d = pd.DataFrame(dict(event_id=[999]*5, nfl_id=[1]*5,
            drill_type=['SKILL_DRILLS_WR']*5, drill_name=[drill]*5, draft_year=[2023]*5,
            time=pd.to_datetime([0, 100, 300, 600, 1000], unit='ms'), speed=[1., 2., 4., 7., 11.]))
        rep = self.reps.build_reps(d.iloc[::-1], T=5, channels=['speed'])[0]
        np.testing.assert_allclose(rep['x'][:, 0], [1., 3.5, 6., 8.5, 11.])
        np.testing.assert_allclose(rep['x'][:, -1], 1.)

    def test_pool_padding_and_rep_order_invariance(self):
        torch.manual_seed(5)
        h = torch.randn(2, 3, 8)
        mask = torch.ones(2, 3, dtype=torch.bool)
        padded = torch.cat([h, torch.randn(2, 4, 8) * 100], dim=1)
        pmask = torch.cat([mask, torch.zeros(2, 4, dtype=torch.bool)], dim=1)
        for kind in ['mean', 'max', 'weighted', 'attention']:
            pool = self.model.Pool(kind, 8).eval()
            with torch.no_grad():
                expected = pool(h, mask)
                torch.testing.assert_close(pool(padded, pmask), expected)
                torch.testing.assert_close(pool(h[:, [2, 0, 1]], mask), expected)

    def test_global_validation_loss_uses_weight_denominator(self):
        class Echo(torch.nn.Module):
            def forward(self, b):
                return b['prediction']
        batches = [dict(prediction=torch.tensor([[1.]]), y=torch.tensor([[0.]]),
                        ym=torch.tensor([[1.]]), w=torch.tensor([[1.]])),
                   dict(prediction=torch.tensor([[3.], [3.]]), y=torch.zeros(2, 1),
                        ym=torch.ones(2, 1), w=torch.tensor([[10.], [10.]]))]
        self.assertAlmostEqual(self.train.eval_loss(Echo(), batches), 181 / 21, places=6)

    def test_scoring_rejects_missing_predictions_and_handles_constants(self):
        y = np.ones((6, 3), dtype=np.float32)
        mask = np.ones_like(y)
        result = self.baseline.score(y.copy(), y, mask)
        self.assertTrue(result[['spearman', 'r2']].isna().all().all())
        pred = y.copy()
        pred[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.baseline.score(pred, y, mask)
        mask[0, 0] = 0
        self.assertEqual(self.baseline.score(pred, y, mask).iloc[0]['n'], 5)

    def test_ridge_imputes_without_touching_test_statistics(self):
        from evaluation import ridge_predict
        x = np.array([[1., np.nan], [2., np.nan], [3., 4.], [4., 5.], [1e6, np.nan]])
        y = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 3))
        pred, state = ridge_predict(x, np.arange(4), np.array([4]), y, np.ones_like(y))
        self.assertEqual(state['median'], [2.5, 4.5])
        self.assertTrue(np.isfinite(pred).all())

    def test_paired_bootstrap_identical_models_have_zero_difference(self):
        from evaluation import intervals
        y = np.tile(np.arange(8, dtype=np.float32)[:, None], (1, 3))
        metrics = intervals({'A':y, 'copy':y.copy()}, y, np.ones_like(y), self.baseline.score, 12)
        paired = metrics[metrics.model == 'copy-A']
        np.testing.assert_allclose(paired[['value', 'ci_low', 'ci_high']], 0.)

    def test_model_ignores_padded_reps(self):
        torch.manual_seed(2)
        cfg = {**self.setup.CFG, 'hidden':8, 'pool':'attention', 'per_family_encoders':True}
        model = self.model.PlayerModel(4, 2, 2, 3, cfg).eval()
        batch = dict(x=torch.randn(1, 2, 5, 4), drill=torch.tensor([[0, 1]]),
                     fam=torch.tensor([[0, 1]]), mask=torch.ones(1, 2, dtype=torch.bool),
                     attrs=torch.randn(1, 2))
        padded = dict(batch)
        padded['x'] = torch.cat([batch['x'], torch.randn(1, 2, 5, 4) * 100], dim=1)
        for key in ['drill', 'fam']:
            padded[key] = torch.cat([batch[key], torch.zeros(1, 2, dtype=torch.long)], dim=1)
        padded['mask'] = torch.tensor([[True, True, False, False]])
        with torch.no_grad():
            torch.testing.assert_close(model(batch), model(padded))


if __name__ == '__main__':
    unittest.main()
