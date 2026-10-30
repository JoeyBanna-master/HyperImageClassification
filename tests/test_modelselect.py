"""Run: python -m unittest discover -s tests -p 'test_modelselect.py' -v."""

import gc
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from main import build_parser
from functions.experiments import get_model
from functions.modelselect import (
    SQSFormerClassifier, build_model, get_model_names, load_model_checkpoint,
)
from model.demunet.model import MyModel


def model_args(name, dataset='HUST2013', classes=15, windows=11):
    return SimpleNamespace(
        model=name, datasets=dataset, classes=classes, windows=windows,
        batch_size=2, device=torch.device('cpu'), learning_rate=0.005,
        momentum=0.9, weight_decay=0.0001, epochs=2,
    )


class ModelSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_registry_and_parser_choices(self):
        expected = {'s2mamba', 'demunet', 'cpformer', 'ssftm', 'hirwkv',
                    'hybridformer', 'sqsformer', 'morphformer',
                    'basemodelv1', 'basemodelRestoreMixer', 'scab_baseline'}
        self.assertEqual(set(get_model_names()), expected)
        parser = build_parser()
        self.assertEqual(parser.parse_args(['--exp_name', 'check']).model, 's2mamba')
        for name in expected:
            self.assertEqual(parser.parse_args(['--exp_name', 'check', '--model', name]).model, name)
        for excluded in ('fbcanet', 'fbscanet', 'state_fusion', 's2mamba_gdn'):
            with self.subTest(excluded=excluded), patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(['--exp_name', 'check', '--model', excluded])

    def test_importing_registry_and_cli_does_not_import_models(self):
        program = (
            'import sys; import main; '
            'assert not any(name == "ssftm_model" or name.startswith("model.") '
            'for name in sys.modules); '
            'assert "mamba_ssm" not in sys.modules; '
            'assert "mmcv" not in sys.modules'
        )
        subprocess.run([sys.executable, '-c', program], check=True, timeout=30)

    def test_experiment_keeps_optimizer_and_scheduler(self):
        args = model_args('cpformer')
        model, optimizer, scheduler = get_model(args)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], args.weight_decay)
        self.assertEqual(optimizer.param_groups[0]['momentum'], args.momentum)
        self.assertIs(scheduler.optimizer, optimizer)
        self.assertEqual(next(model.parameters()).device, args.device)

    def test_sqsformer_adaptation_and_old_class_path_checkpoint(self):
        args = model_args('sqsformer')
        model = build_model(args).eval()
        patches = torch.randn(2, 144, 11, 11)
        with torch.no_grad():
            original_outputs = model.backbone(patches)
            actual = model(patches)
        self.assertEqual(actual.shape, (2, 15))
        self.assertEqual(len(original_outputs), 3)
        torch.testing.assert_close(actual, original_outputs[0])
        self.assertTrue(actual.isfinite().all())
        from functions.experiments import SQSFormerClassifier as old_class_path
        self.assertIs(old_class_path, SQSFormerClassifier)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'sqsformer.pth'
            # Emulate a checkpoint saved before the wrapper moved to modelselect.
            with patch.object(SQSFormerClassifier, '__module__', 'functions.experiments'):
                torch.save(model, checkpoint)
            # Verify in a FRESH interpreter, as in a spawned worker/parent load.
            program = (
                'import sys, torch; from functions.modelselect import load_model_checkpoint; '
                'torch.set_num_threads(1); '
                'model=load_model_checkpoint(sys.argv[1], "sqsformer").eval(); '
                'assert model(torch.zeros(1,144,11,11)).shape == (1,15)'
            )
            subprocess.run([sys.executable, '-c', program, str(checkpoint)],
                           check=True, timeout=40)
            with torch.no_grad():
                restored = load_model_checkpoint(checkpoint, 'sqsformer').eval()
                torch.testing.assert_close(restored(patches), actual)

    def test_demunet_houston_and_pu_dimensions_and_partial_batches(self):
        for dataset, bands, attention_dim, classes, window in (
            ('HUST2013', 144, 136, 15, 11), ('PU', 103, 96, 9, 7),
        ):
            with self.subTest(dataset=dataset):
                model = build_model(model_args('demunet', dataset, classes, window)).eval()
                self.assertEqual((model.ic, model.ix), (bands, attention_dim))
                self.assertEqual(model.mamba.mamba.config.d_model, bands)
                self.assertEqual(model.conv.out_channels, attention_dim)
                self.assertEqual(model.encoder1.encoder.layers[0].self_attn.embed_dim, attention_dim)
                self.assertEqual(model.ver_conv.out_channels, bands)
                with torch.no_grad():
                    for batch_size in (2, 1):
                        logits = model(torch.randn(batch_size, bands, window, window))
                        self.assertEqual(logits.shape, (batch_size, classes))
                        self.assertTrue(logits.isfinite().all())
                del model, logits
                gc.collect()

    def test_demunet_strict_multiple_and_backward(self):
        # ic=16 is divisible by 8: strict '< ic' requires ix=8, not 16.
        model = MyModel(batch_size=2, in_channels=16, num_classes=3, patch_size=5)
        self.assertEqual(model.ix, 8)
        logits = model(torch.randn(2, 16, 5, 5))
        nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        for layer in (model.conv, model.ver_conv):
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(layer.weight.grad.isfinite().all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_demunet_invalid_configuration(self):
        for bands in (0, 8):
            with self.assertRaisesRegex(ValueError, 'greater than 8'):
                MyModel(in_channels=bands)
        model = MyModel(batch_size=2, in_channels=16, num_classes=3, patch_size=5).eval()
        for patches, message in (
            (torch.randn(1, 15, 5, 5), 'expects 16 bands'),
            (torch.randn(1, 16, 7, 7), 'expects a 5x5'),
            (torch.randn(3, 16, 5, 5), 'exceeds configured maximum'),
        ):
            with self.assertRaisesRegex(ValueError, message):
                model(patches)

    def test_cuda_only_models_fail_before_import_on_cpu(self):
        for name in ('s2mamba', 'ssftm', 'hirwkv'):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, name):
                build_model(model_args(name))

    def test_selected_dependency_error_names_the_model(self):
        def fail_to_import(args, config):
            raise ModuleNotFoundError('missing_selected_dependency')
        with patch.dict('functions.modelselect.MODEL_BUILDERS', {'cpformer': fail_to_import}):
            with self.assertRaisesRegex(RuntimeError, 'cpformer.*missing_selected_dependency'):
                build_model(model_args('cpformer'))

    def test_ssftm_checkpoint_prepares_dynamic_module(self):
        with patch('functions.modelselect._load_ssftm_module') as prepare, \
             patch('functions.modelselect.torch.load', return_value='loaded') as load:
            self.assertEqual(load_model_checkpoint('example.pth', 'ssftm'), 'loaded')
            prepare.assert_called_once_with()
            load.assert_called_once_with('example.pth', map_location='cpu', weights_only=False)


if __name__ == '__main__':
    unittest.main()
