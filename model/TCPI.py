"""Target-Conditioned Progressive Inference for HSI center-pixel classification.

Architecture contract: docs/proposed_architecture_spec.md.
Input: [batch, bands, height, width]. Output: [batch, num_classes] logits.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ConvObservationEncoder(nn.Module):
    """F_obs: [batch, group_size, H, W] -> [batch, H * W, dim]."""

    def __init__(self, group_size: int, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(group_size, dim, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x_group: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.conv(x_group))
        return features.flatten(2).transpose(1, 2)


class MLPQueryInitializer(nn.Module):
    """F_q: the complete center spectrum [batch, bands] -> q0 [batch, dim]."""

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, center_spectrum: torch.Tensor) -> torch.Tensor:
        return self.mlp(center_spectrum)


class CrossAttentionRetriever(nn.Module):
    """R(Zg, q_prev): [batch, N, dim], [batch, dim] -> Eg [batch, dim]."""

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, observations: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        evidence, _ = self.attention(
            query=query.unsqueeze(1),
            key=observations,
            value=observations,
            need_weights=False,
        )
        return evidence.squeeze(1)


class GRUStateUpdater(nn.Module):
    """U(q_prev, Eg): two [batch, dim] tensors -> qg [batch, dim]."""

    def __init__(self, dim: int):
        super().__init__()
        self.cell = nn.GRUCell(input_size=dim, hidden_size=dim)

    def forward(self, state: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.cell(evidence, state)


class TCPI(nn.Module):
    """Shared observation encoding with sequential, state-conditioned retrieval.

    Each group contains ``group_size`` consecutive original bands. Only the
    final incomplete group is zero-padded; q0 always uses the unpadded complete
    center spectrum. Spatial dimensions must be positive and odd so the patch
    has one unambiguous center pixel.

    All five modules can be supplied independently through keyword arguments.
    Replacements must honor the tensor contracts documented above; the
    classifier maps [batch, dim] to [batch, num_classes]. Each module instance
    is reused across stages. State is local to one forward call.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        group_size: int = 10,
        dim: int = 64,
        *,
        observation_encoder: Optional[nn.Module] = None,
        query_initializer: Optional[nn.Module] = None,
        evidence_retriever: Optional[nn.Module] = None,
        state_updater: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
    ):
        super().__init__()
        for name, value in (
            ('in_channels', in_channels), ('num_classes', num_classes),
            ('group_size', group_size), ('dim', dim),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.group_size = group_size
        self.dim = dim
        self.num_groups = (in_channels + group_size - 1) // group_size
        self.observation_encoder = (
            observation_encoder if observation_encoder is not None
            else ConvObservationEncoder(group_size, dim)
        )
        self.query_initializer = (
            query_initializer if query_initializer is not None
            else MLPQueryInitializer(in_channels, dim)
        )
        self.evidence_retriever = (
            evidence_retriever if evidence_retriever is not None
            else CrossAttentionRetriever(dim)
        )
        self.state_updater = (
            state_updater if state_updater is not None else GRUStateUpdater(dim)
        )
        self.classifier = (
            classifier if classifier is not None else nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError('TCPI expects [batch, bands, height, width] input')
        if x.shape[1] != self.in_channels:
            raise ValueError(f'Expected {self.in_channels} bands, got {x.shape[1]}')
        height, width = x.shape[-2:]
        if height < 1 or width < 1 or height % 2 == 0 or width % 2 == 0:
            raise ValueError('Patch height and width must be positive and odd')

        # Initialize once from the ORIGINAL complete center spectrum.
        center_spectrum = x[:, :, height // 2, width // 2]
        state = self.query_initializer(center_spectrum)

        for x_group in x.split(self.group_size, dim=1):
            missing_bands = self.group_size - x_group.shape[1]
            if missing_bands:
                x_group = F.pad(x_group, (0, 0, 0, 0, 0, missing_bands))
            observations = self.observation_encoder(x_group)
            # Eg = R(Zg, q_(g-1)): the evolving state controls acquisition.
            evidence = self.evidence_retriever(observations, state)
            # qg = U(q_(g-1), Eg): reuse qg as the NEXT stage's query.
            state = self.state_updater(state, evidence)

        return self.classifier(state)
