"""CPU correctness checks; no real datasets or GPU required.

Run from the project root: python -m unittest discover -s tests -v
"""
import copy
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch_dct as DCT
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset

from model.FBCANet import DFIM
from Mydataset.High_Dataset import MyDataset
from functions.val_function import validate
from functions.test_function import test as evaluate
from functions.experiments import execute, make_loaders, select_devices
from Whole_FBCANet import build_parser


def legacy_high_frequency(x, ratio):
    h, w = x.shape[-2:]
    weight = torch.ones((h, w), device=x.device)
    weight[:int(h * ratio[0]), :int(w * ratio[1])] = 0
    return DCT.idct_2d(DCT.dct_2d(x, norm='ortho') * weight, norm='ortho')


def legacy_dfim(module, x):
    # Reproduce the previous independent transforms in both branches.
    spatial = module.spatial(x, high_frequency=legacy_high_frequency(x, module.spatial.ratio))
    channel = module.channel(x, high_frequency=legacy_high_frequency(x, module.channel.ratio))
    return module.out(spatial + channel)


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_dct_outputs_and_gradients(self):
        torch.manual_seed(9)
        new = DFIM(32).double().train()
        new.spatial.alpha.data.fill_(0.7)  # Exercise the spatial gradient, not its zero-init shortcut.
        old = copy.deepcopy(new)
        x = torch.randn(2, 32, 11, 11, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        reference = legacy_dfim(old, x)
        module = importlib.import_module('model.FBCANet')
        with patch.object(module.DCT, 'dct_2d', wraps=DCT.dct_2d) as dct:
            actual = new(y)
            self.assertEqual(dct.call_count, 1)
        torch.testing.assert_close(actual, reference, rtol=1e-8, atol=1e-9)
        upstream = torch.randn_like(actual)
        reference.backward(upstream)
        actual.backward(upstream)
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-7, atol=1e-8)
        for before, after in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(before.grad, after.grad, rtol=1e-7, atol=1e-8)
        keys = set(old.state_dict())
        self.assertEqual(keys, set(new.state_dict()))
        cache = new._frequency_mask
        new(y.detach())
        self.assertIs(cache, new._frequency_mask)
        new(torch.randn(2, 32, 9, 9, dtype=torch.float64))
        self.assertEqual(new._frequency_mask.shape[-2:], (9, 9))
        new.float()(y.detach().float())
        self.assertEqual(new._frequency_mask.dtype, torch.float32)
        restored = DFIM(32)
        restored.load_state_dict(new.state_dict(), strict=True)

    def test_non_dct_branch(self):
        model = DFIM(32, isdct=False).eval()
        x = torch.randn(2, 32, 11, 11)
        torch.testing.assert_close(model(x), model.out(model.spatial(x) + model.channel(x)))

    def test_dataset_and_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            samples = np.arange(120, dtype=np.float64).reshape(5, 3, 2, 4)
            labels = np.arange(5, dtype=np.int32)
            np.save(directory / 'x.npy', samples)
            np.save(directory / 'y.npy', labels)
            dataset = MyDataset(directory / 'x.npy', directory / 'y.npy')
            self.assertEqual(dataset.datas.dtype, torch.float32)
            self.assertEqual(dataset.labels.dtype, torch.int64)
            np.testing.assert_array_equal(dataset.datas.numpy(), samples.astype(np.float32))
            raw = MyDataset(directory / 'x.npy', directory / 'y.npy', pre_transform=False)
            self.assertIsInstance(raw.datas, np.ndarray)
            args = build_parser().parse_args(['--batch_size', '2'])
            args.device = torch.device('cpu')
            first = make_loaders(args, [dataset], 11)[0]
            second = make_loaders(args, [dataset], 11)[0]
            self.assertEqual(len(first), 2)  # Preserve drop_last behavior.
            for left, right in zip(first, second):
                torch.testing.assert_close(left[0], right[0])
                torch.testing.assert_close(left[1], right[1])
            args.num_workers = 1
            threaded = make_loaders(args, [dataset], 11)[0]
            self.assertEqual(sum(len(y) for _, y in threaded), 4)
            del threaded

    def test_evaluation_metrics_and_empty_loaders(self):
        logits = torch.tensor([[3., 1., 0.], [0., 3., 1.], [0., 2., 3.],
                               [0., 4., 1.], [3., 0., 1.]])
        labels = torch.tensor([0, 2, 2, 1, 1])
        model = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3))
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        expected = logits.argmax(1).numpy()
        self.assertEqual(validate(model, loader), round(metrics.accuracy_score(labels, expected) * 100, 2))
        args = SimpleNamespace(device=torch.device('cpu'))
        oa, aa, kappa, _, class_acc, elapsed = evaluate(args, 0, loader, model)
        expected_class = metrics.recall_score(labels, expected, average=None, zero_division=0)
        self.assertEqual(oa, metrics.accuracy_score(labels, expected))
        self.assertEqual(aa, expected_class.mean())
        self.assertEqual(kappa, metrics.cohen_kappa_score(labels, expected))
        np.testing.assert_array_equal(class_acc, expected_class)
        self.assertGreaterEqual(elapsed, 0)
        empty = DataLoader(TensorDataset(logits[:0], labels[:0]), batch_size=2)
        with self.assertRaises(ValueError):
            validate(model, empty)
        with self.assertRaises(ValueError):
            evaluate(args, 0, empty, model)

    def test_device_validation(self):
        with patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(select_devices('0,1'), ['cuda:0', 'cuda:1'])
            for invalid in ('0,0', '2', '-1', 'x', ''):
                with self.assertRaises(ValueError):
                    select_devices(invalid)

    def test_serial_and_spawned_experiments_match(self):
        # Use tiny synthetic data with all 15 classes; keep the actual FBCANet.
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            folder = directory / 'datasets' / 'Houston'
            folder.mkdir(parents=True)
            rng = np.random.default_rng(5)
            for split in ('train', 'val', 'test'):
                np.save(folder / f'10%X{split}WindowSize11.npy',
                        rng.normal(size=(15, 144, 11, 11)).astype(np.float32))
                np.save(folder / f'10%y{split}WindowSize11.npy', np.arange(15))
            args = build_parser().parse_args(['--iters', '2', '--epochs', '2', '--batch_size', '15'])
            try:
                os.chdir(directory)
                serial = execute(args, ['cpu'])
                spawned = execute(args, ['cpu', 'cpu'])
                summaries = [json.loads((p / 'summary.json').read_text(encoding='utf-8')) for p in (serial, spawned)]
                self.assertEqual(summaries[0]['mean'], summaries[1]['mean'])
                self.assertEqual(summaries[0]['sample_std'], summaries[1]['sample_std'])
                for index in range(2):
                    left = torch.load(serial / f'iter_{index:03d}' / 'model.pth', weights_only=False)
                    right = torch.load(spawned / f'iter_{index:03d}' / 'model.pth', weights_only=False)
                    for name, tensor in left.state_dict().items():
                        torch.testing.assert_close(tensor, right.state_dict()[name], rtol=0, atol=0)
                    self.assertTrue((spawned / f'iter_{index:03d}' / 'experiment.log').is_file())
                self.assertTrue((spawned / 'best_model.pth').is_file())
                self.assertEqual([r['seed'] for r in summaries[0]['experiments']], [0, 1])
            finally:
                os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
