"""Run: python -m unittest discover -s tests -p 'test_baseline_features.py' -v."""

import copy
import unittest

import torch
from torch.nn import functional as F

from model.basemodel.AdaptiveConv import (
    AdaptiveDilatedConv2d,
    HSIAdaptiveSpatialConv,
    MultiScaleFeatureExtractorWithAdaDR,
)


class BaselineFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(37)

    def test_integer_dilation_matches_replication_then_zero_padding(self):
        # Only the nearest border pixel is replicated. Further samples see
        # zeros, matching DCNv2 applied to the explicitly padded input.
        x = torch.randn(2, 2, 4, 7, dtype=torch.float64)
        for dilation in (1, 2, 3, 5):
            with self.subTest(dilation=dilation):
                layer = AdaptiveDilatedConv2d(2, 3, initial_dilation=2).double()
                with torch.no_grad():
                    layer.conv_offset.bias.fill_((dilation - 1) / 2)
                    layer.bias.copy_(torch.tensor([0.2, -0.4, 0.7]))
                expected = F.conv2d(
                    F.pad(x, (1, 1, 1, 1), mode='replicate'),
                    layer.weight * 0.5,
                    layer.bias,
                    padding=dilation - 1,
                    dilation=dilation,
                )
                torch.testing.assert_close(
                    layer(x), expected, rtol=1e-12, atol=1e-12,
                )

    def test_initial_dilation_and_half_mask(self):
        layer = AdaptiveDilatedConv2d(1, 2, initial_dilation=3).double()
        x = torch.randn(1, 1, 5, 7, dtype=torch.float64)
        raw = layer.conv_offset(layer.PAD(x))
        torch.testing.assert_close(
            1 + raw.abs() * 3,
            torch.full_like(raw, 3 + 3 * layer.epsilon),
        )
        torch.testing.assert_close(
            layer.conv_mask(layer.PAD(x)).sigmoid(),
            torch.full((1, 9, 5, 7), 0.5, dtype=torch.float64),
        )

    def test_band_and_batch_order_and_band_isolation(self):
        layer = HSIAdaptiveSpatialConv(out_channels=3).double()
        with torch.no_grad():
            layer.adadr.conv_offset.weight.fill_(0.01)
            layer.adadr.conv_mask.weight.fill_(0.02)
        x = torch.randn(2, 1, 3, 5, 7, dtype=torch.float64)
        output = layer(x)
        expected = torch.stack([
            torch.stack([
                layer.adadr(x[b, :, band].unsqueeze(0))[0]
                for band in range(3)
            ], dim=1)
            for b in range(2)
        ])
        self.assertEqual(output.shape, (2, 3, 3, 5, 7))
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(layer(x.flip(0)), output.flip(0))
        order = [2, 0, 1]
        torch.testing.assert_close(layer(x[:, :, order]), output[:, :, order])

        changed = x.clone()
        changed[1, :, 2] += 2
        actual = layer(changed)
        torch.testing.assert_close(actual[0], output[0])
        torch.testing.assert_close(actual[1, :, :2], output[1, :, :2])
        self.assertGreater((actual[1, :, 2] - output[1, :, 2]).abs().max().item(), 0)

        differentiable = x.clone().requires_grad_()
        layer(differentiable)[1, :, 2].square().mean().backward()
        self.assertEqual(differentiable.grad[0].abs().sum().item(), 0)
        self.assertEqual(differentiable.grad[1, :, :2].abs().sum().item(), 0)
        self.assertGreater(differentiable.grad[1, :, 2].abs().sum().item(), 0)

    def test_four_outputs_and_learnable_offset_mask_gradients(self):
        model = MultiScaleFeatureExtractorWithAdaDR(embed_dim=3)
        x = torch.randn(2, 5, 5, 7, requires_grad=True)
        features = model(x)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertEqual(feature.shape, (2, 3, 5, 5, 7))
            self.assertTrue(torch.isfinite(feature).all())
        sum(feature.square().mean() for feature in features).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        adadr = model.spatial_medium[0].adadr
        for predictor in (adadr.conv_offset, adadr.conv_mask):
            for parameter in predictor.parameters():
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        for branch in (model.spectral, model.spatial_small, model.spatial_large):
            self.assertGreater(branch[0].weight.grad.abs().sum().item(), 0)

    def test_optional_torchvision_outputs_and_gradients(self):
        # torchvision is a test oracle only, never a runtime dependency.
        try:
            from torchvision.ops import deform_conv2d
            deform_conv2d(
                torch.ones(1, 1, 3, 3), torch.zeros(1, 18, 1, 1),
                torch.ones(1, 1, 3, 3),
            )
        except (ImportError, OSError, RuntimeError, NotImplementedError) as exc:
            self.skipTest(f'torchvision deform_conv2d unavailable: {exc}')

        for raw_bias in (-0.63, 0.71):
            for h, w in ((4, 7), (1, 3), (3, 1)):
                with self.subTest(raw_bias=raw_bias, shape=(h, w)):
                    actual = AdaptiveDilatedConv2d(2, 3).double()
                    with torch.no_grad():
                        actual.conv_offset.weight.normal_(std=0.012)
                        actual.conv_offset.bias.fill_(raw_bias)
                        actual.conv_mask.weight.normal_(std=0.07)
                        actual.conv_mask.bias.normal_(std=0.11)
                    reference = copy.deepcopy(actual)
                    x = torch.randn(2, 2, h, w, dtype=torch.float64, requires_grad=True)
                    x_reference = x.detach().clone().requires_grad_()
                    padded = reference.PAD(x_reference)
                    delta = reference.conv_offset(padded).abs() * reference.initial_dilation
                    offsets = delta * reference.dilated_offset.reshape(1, 18, 1, 1)
                    expected = deform_conv2d(
                        padded, offsets, reference.weight, reference.bias,
                        padding=(0, 0), dilation=(1, 1),
                        mask=reference.conv_mask(padded).sigmoid(),
                    )
                    result = actual(x)
                    torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-11)
                    upstream = torch.randn_like(result)
                    result.backward(upstream)
                    expected.backward(upstream)
                    torch.testing.assert_close(
                        x.grad, x_reference.grad, rtol=1e-10, atol=1e-11,
                    )
                    for name, parameter in actual.named_parameters():
                        torch.testing.assert_close(
                            parameter.grad,
                            dict(reference.named_parameters())[name].grad,
                            rtol=1e-10, atol=1e-11,
                            msg=lambda message, name=name: f'{name}: {message}',
                        )


if __name__ == '__main__':
    unittest.main()
