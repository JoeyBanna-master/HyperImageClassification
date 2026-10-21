"""Run: python -m unittest discover -s tests -p 'test_state_fusion.py' -v"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from main import build_parser
from functions.experiments import make_loaders
from functions.train_function_new import train
from model.StateFusionHSI import (
    StateFusionHSI, SpectralFeatureExtractor, SpatialFeatureExtractor,
)


class StateFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def test_houston_shapes_and_gradients_through_all_states(self):
        model = StateFusionHSI()
        states, gates, handles = [], [], []

        def capture_state(module, inputs, outputs):
            outputs[1].retain_grad()
            states.append(outputs[1])

        for block in model.blocks:
            handles.append(block.register_forward_hook(capture_state))
            handles.append(block.feature_fusion.branch_gate.register_forward_hook(
                lambda module, inputs, output: gates.append(output)
            ))
        logits = model(torch.randn(3, 144, 11, 11))
        self.assertEqual(logits.shape, (3, 15))
        nn.functional.cross_entropy(logits, torch.tensor([1, 4, 12])).backward()
        for state in states:
            self.assertEqual(state.shape, (3, 64, 11, 11))
            self.assertTrue(torch.isfinite(state.grad).all())
            self.assertGreater(state.grad.abs().sum().item(), 0)
        for gate in gates:
            torch.testing.assert_close(gate.sum(dim=1), torch.ones(3, 11, 11))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        for handle in handles:
            handle.remove()

    def test_state_is_carried_and_changes_later_predictions(self):
        model = StateFusionHSI(feature_dim=8).eval()
        patch_input = torch.randn(2, 144, 11, 11)
        outputs, next_inputs = [], []
        first_hook = model.blocks[0].register_forward_hook(
            lambda module, inputs, result: outputs.append(result)
        )
        next_hook = model.blocks[1].register_forward_pre_hook(
            lambda module, inputs: next_inputs.append(inputs)
        )
        with torch.no_grad():
            reference = model(patch_input)
        self.assertIs(outputs[0][1], next_inputs[0][1])
        first_hook.remove()
        next_hook.remove()

        # Change ONLY the carried state, keeping the first block's content fixed.
        def perturb_carried_state(module, inputs, result):
            content, state = result
            channel_offset = torch.arange(8, dtype=state.dtype).view(1, 8, 1, 1)
            return content, state + channel_offset

        handle = model.blocks[0].register_forward_hook(perturb_carried_state)
        with torch.no_grad():
            changed = model(patch_input)
        handle.remove()
        self.assertGreater((changed - reference).abs().max().item(), 1e-6)

    def test_extractors_have_different_spatial_dependencies(self):
        content = torch.randn(1, 8, 7, 7)
        changed_neighbor = content.clone()
        changed_neighbor[:, 0, 3, 4] += 5
        spectral = SpectralFeatureExtractor(8).eval()
        spatial = SpatialFeatureExtractor(8).eval()
        with torch.no_grad():
            torch.testing.assert_close(
                spectral(content)[:, :, 3, 3], spectral(changed_neighbor)[:, :, 3, 3]
            )
            spatial_difference = spatial(content)[:, :, 3, 3] - spatial(changed_neighbor)[:, :, 3, 3]
        self.assertGreater(spatial_difference.abs().max().item(), 1e-6)

    def test_batch_independence_and_checkpoint_roundtrip(self):
        model = StateFusionHSI(feature_dim=8).eval()
        samples = torch.randn(3, 144, 9, 13)
        with torch.no_grad():
            expected = model(samples)
            model(torch.randn_like(samples))  # Previous patches cannot change state.
            separate = torch.cat([model(sample.unsqueeze(0)) for sample in samples])
        torch.testing.assert_close(expected, separate, rtol=1e-5, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pth'
            torch.save(model, checkpoint)
            restored = torch.load(checkpoint, weights_only=False).eval()
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(samples))

    def test_bad_input_and_other_band_counts(self):
        model = StateFusionHSI(feature_dim=8)
        for invalid in (torch.randn(2, 144, 10, 11), torch.randn(2, 143, 11, 11),
                        torch.randn(2, 144, 11)):
            with self.assertRaises(ValueError):
                model(invalid)
        alternative = StateFusionHSI(in_channels=103, num_classes=9, feature_dim=8)
        self.assertEqual(alternative(torch.randn(1, 103, 7, 7)).shape, (1, 9))

    def test_entrypoint_and_complete_evaluation_loaders(self):
        args = build_parser().parse_args(['--exp_name', 'unit_test', '--batch_size', '2'])
        args.device = torch.device('cpu')
        self.assertFalse(hasattr(args, 'feature_dim'))
        self.assertFalse(hasattr(args, 'num_blocks'))
        dataset = TensorDataset(torch.randn(5, 144, 11, 11), torch.arange(5))
        train_loader, val_loader, test_loader = make_loaders(args, [dataset] * 3, 0)
        self.assertEqual(sum(len(labels) for _, labels in train_loader), 4)
        for loader in (val_loader, test_loader):
            self.assertEqual(torch.cat([labels for _, labels in loader]).tolist(), list(range(5)))

    def test_best_validation_weights_are_restored(self):
        model = nn.Linear(1, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        args = SimpleNamespace(device=torch.device('cpu'), classes=2)

        def train_one_epoch(args, loader, current_model, criterion, optimizer, epoch):
            with torch.no_grad():
                current_model.weight.fill_(epoch + 1.0)
            optimizer.step()
            return {'loss': 1.0, 'accuracy': 0.5, 'num_samples': 2}

        validations = [
            {'loss': 0.2, 'accuracy': 0.9, 'num_samples': 2},
            {'loss': 0.8, 'accuracy': 0.5, 'num_samples': 2},
        ]
        with patch('functions.train_function_new.train_epoch', side_effect=train_one_epoch), \
             patch('functions.train_function_new.validate', side_effect=validations), \
             redirect_stdout(io.StringIO()):
            best = train(args, 0, model, None, None, None, optimizer, scheduler, 2)
        torch.testing.assert_close(best.weight, torch.ones_like(best.weight))
        self.assertEqual(args.best_epoch, 1)
        self.assertEqual(args.best_val_accuracy, 0.9)


if __name__ == '__main__':
    unittest.main()
