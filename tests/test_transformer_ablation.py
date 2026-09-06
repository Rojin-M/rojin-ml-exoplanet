"""Check controlled normalization bypass, auxiliary learning, and deferred test evaluation."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

import test_cnn_auxiliary as aux_fixture
from training.cnn import train_kepler_cnn as cnn
from training.transformer import train_kepler_transformer as transformer


class TransformerAblationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = aux_fixture.AuxiliaryCNNTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.cfg = transformer.TrainConfig(base_dir=str(f.root), dataset_path=str(f.v0),
            splits_path=str(f.splits), manifest_path=str(f.manifest), output_root=str(f.root/'transformer'),
            epochs=2, batch_size=6, num_workers=0, device='cpu', threshold_grid_size=5,
            d_model=16, num_heads=4, global_depth=2, local_depth=2, mlp_ratio=2.)
        self.kwargs = dict(global_seq_len=32, local_seq_len=16, scalar_features=7,
            **{key:getattr(self.cfg,key) for key in ('d_model','num_heads','global_depth','local_depth',
                'mlp_ratio','dropout','attention_dropout','token_dropout','scalar_dropout')})

    def test_normalization_bypass_preserves_initial_weights_and_rng(self):
        cnn.set_seed(42); original = transformer.ExoFormer(**self.kwargs)
        original_rng = torch.get_rng_state().clone()
        cnn.set_seed(42); bypass = transformer.ExoFormer(**self.kwargs, use_input_norm=False)
        self.assertTrue(torch.equal(original_rng,torch.get_rng_state()))
        for key,value in original.state_dict().items():
            self.assertTrue(torch.equal(value,bypass.state_dict()[key]))
        arrays,_ = cnn.load_training_data(self.cfg)
        values = [torch.from_numpy(arrays[key][:6]) for key in ('x_global','x_local','x_scalar','x_flat')]
        original.eval(); bypass.eval()
        self.assertFalse(torch.allclose(original(*values),bypass(*values)))
        for branch in (bypass.global_branch,bypass.local_branch):
            self.assertFalse(branch.input_norm.weight.requires_grad)
            self.assertEqual(branch.encoder.layers[0].self_attn.dropout, self.cfg.dropout)

    def test_deferred_test_and_offline_logging(self):
        tracker = MagicMock(); tracker.config = {}; tracker.summary = {}
        cfg = replace(self.cfg,evaluate_test=False)
        seen=[]; actual=transformer.evaluate_split
        def evaluate(model,loader,*args,**kwargs):
            indices=set(loader.dataset.indices.tolist())
            self.assertFalse(indices & set(range(18,24)))
            seen.append(indices)
            return actual(model,loader,*args,**kwargs)
        with patch.object(transformer,'tracking_run') as tracking, patch.object(transformer,'evaluate_split',side_effect=evaluate), redirect_stdout(io.StringIO()):
            tracking.return_value.__enter__.return_value=tracker
            run=transformer.train_model(cfg)
        self.assertTrue(seen)
        self.assertFalse((run/'test_predictions.csv').exists())
        summary=json.loads((run/'summary.json').read_text())
        self.assertFalse(summary['test_evaluated']); self.assertNotIn('test_metrics',summary)
        self.assertFalse(any(k.startswith('test/') for call in tracker.log.call_args_list for k in call.args[0]))
        history=pd.read_csv(run/'history.csv')
        self.assertEqual(history.iloc[0].learning_rate,cfg.learning_rate)
        self.assertIn('val_accuracy',history)
        self.assertEqual(tracker.config['effective_attention_dropout'],cfg.dropout)

    def test_auxiliary_training_and_checkpoint_reload(self):
        f=self.fixture
        cfg=replace(self.cfg,dataset_path=str(f.v1),splits_path=f.cfg.splits_path,
            manifest_path=f.cfg.manifest_path,use_aux_branch=True,use_input_norm=False,evaluate_test=False)
        with redirect_stdout(io.StringIO()): run=transformer.train_model(cfg)
        checkpoint=torch.load(run/'best_model.pt',map_location='cpu')
        cnn.set_seed(cfg.seed); model=transformer.ExoFormer(**checkpoint['model_config'])
        self.assertFalse(torch.equal(model.aux_branch.net[0].weight,checkpoint['model_state']['aux_branch.net.0.weight']))
        model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
        self.assertEqual(checkpoint['aux_preprocessing'],json.loads((run/'aux_preprocessing.json').read_text()))
        arrays,_=cnn.load_training_data(cfg)
        with np.load(cfg.dataset_path,allow_pickle=True) as stored:
            np.testing.assert_array_equal(arrays['x_aux'],stored['X_aux'])
        indices=arrays['val_idx']; values=[torch.from_numpy(arrays[key][indices]) for key in ('x_global','x_local','x_scalar','x_flat')]
        aux=torch.from_numpy(arrays['x_aux'][indices])
        prediction=model(*values,aux)
        self.assertFalse(torch.allclose(prediction,model(*values,aux+torch.arange(11))))
        prediction.sum().backward()
        self.assertGreater(float(model.aux_branch.net[0].weight.grad.abs().sum()),0)
        saved=pd.read_csv(run/'val_predictions.csv')
        np.testing.assert_allclose(torch.sigmoid(prediction).detach().numpy(),saved.probability,rtol=1e-6)
        with self.assertRaisesRegex(ValueError,'required'):model(*values)

    def test_cli_requires_auxiliary_paths_and_preserves_legacy_defaults(self):
        with patch('sys.argv',['train']):
            args=transformer.parse_args()
            self.assertTrue(args.use_input_norm);self.assertTrue(args.evaluate_test)
        with patch('sys.argv',['train','--aux_branch']):
            with self.assertRaisesRegex(ValueError,'explicit'):transformer.build_config(transformer.parse_args())
        with patch('sys.argv',['train','--no_input_norm','--skip_test','--no_amp','--base_dir',str(self.fixture.root),
            '--dataset_path',self.cfg.dataset_path,'--manifest_path',self.cfg.manifest_path,'--splits_path',self.cfg.splits_path]):
            cfg=transformer.build_config(transformer.parse_args())
            self.assertFalse(cfg.use_input_norm);self.assertFalse(cfg.evaluate_test)
            self.assertFalse(cfg.use_amp)

    def test_deferred_checkpoint_evaluation_preserves_training_record(self):
        from scripts.experiments.evaluate_transformer_checkpoint import evaluate
        cfg=replace(self.cfg,evaluate_test=False)
        with redirect_stdout(io.StringIO()):
            run=transformer.train_model(cfg)
            before=(run/'summary.json').read_bytes()
            result=evaluate(run,'cpu')
        self.assertEqual((run/'summary.json').read_bytes(),before)
        self.assertTrue(result['validation_checkpoint_replay_matches'])
        self.assertEqual(result['test_size'],6)
        saved=pd.read_csv(run/'test_predictions.csv')
        self.assertEqual(saved.candidate_id.tolist(),[f'K{i:05d}.01' for i in range(18,24)])
        self.assertAlmostEqual(cnn.compute_metrics(saved.y.to_numpy(),saved.probability.to_numpy(),
            result['best_threshold'])['pr_auc'],result['test_metrics']['pr_auc'])
        with self.assertRaisesRegex(ValueError,'deferred'):evaluate(run,'cpu')


if __name__ == '__main__':
    unittest.main()
