"""S2Mamba with a bidirectional Gated DeltaNet spectral branch.

Input: [batch, bands, patch, patch]; output: [batch, num_classes].
PCS, SMG, residual connections and pooling follow model/s2mamba.py.
Only the BSS sequence mixer is replaced; tokens remain learned channels,
not raw wavelength bands. Both directions share one GatedDeltaNet.

PyTorch reference scans work without timm, einops, Mamba or FLA. On CUDA,
the auto backends prefer installed mamba_ssm / FLA kernels. Reference scans
are intended for correctness checks, not a claim of training throughput.

GDN recurrence: https://arxiv.org/abs/2412.06464
Local reference: GatedDeltaNet-main/lit_gpt/gated_delta_rule_ops/chunk.py
"""

from functools import lru_cache
import importlib
import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["S2Mamba_GDN", "s2mamba_GDN", "BSS_GDN", "GatedDeltaNet"]


@lru_cache(maxsize=None)
def _optional_kernel(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, function_name, None)


@lru_cache(maxsize=None)
def _fla_layout_options(kernel):
    # Older FLA releases defaulted to head-first tensors; current releases
    # accept only [B, L, heads, dim] and have removed this parameter.
    return {"head_first": False} if "head_first" in inspect.signature(kernel).parameters else {}


def _selective_scan_reference(u, delta, A, B, C, D, delta_bias):
    """Original PCS selective scan, with FP32 states and grouped B/C.

    u/delta: [B, directions * channels, L]; B/C: [B, directions, N, L].
    Uses the same exp(delta*A) and delta*B*u discretization as Mamba's
    selective_scan_ref; the D*u skip is retained.
    """
    batch, features, length = u.shape
    directions, state_size = B.shape[1:3]
    channels = features // directions
    u = u.float().reshape(batch, directions, channels, length)
    delta = F.softplus(delta.float() + delta_bias.float()[None, :, None])
    delta = delta.reshape(batch, directions, channels, length)
    A = A.float().reshape(directions, channels, state_size)
    B, C = B.float(), C.float()
    D = D.float().reshape(directions, channels)
    state = u.new_zeros(batch, directions, channels, state_size)
    outputs = []
    for t in range(length):
        dt, value = delta[..., t], u[..., t]
        state = (
            torch.exp(dt[..., None] * A[None]) * state
            + dt[..., None] * B[..., t].unsqueeze(2) * value[..., None]
        )
        outputs.append((state * C[..., t].unsqueeze(2)).sum(-1) + D * value)
    return torch.stack(outputs, dim=-1).reshape(batch, features, length)


def _gated_delta_rule_reference(q, k, v, beta, g):
    """Gated delta recurrence with [B, L, heads, dim] tensors.

    S <- exp(g) * S
    S <- S + beta * k * (v - S^T k)^T
    y <- S^T (q / sqrt(key_dim))

    Each call starts from zero. FP32 accumulation also supports half/bfloat16
    inputs; float64 is retained for mathematical gradient checks.
    """
    output_dtype = v.dtype
    state_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
    q, k, v, beta, g = (tensor.to(state_dtype) for tensor in (q, k, v, beta, g))
    batch, length, heads, key_dim = q.shape
    state = v.new_zeros(batch, heads, key_dim, v.shape[-1])
    q = q * (key_dim ** -0.5)
    outputs = []
    for t in range(length):
        state = state * g[:, t].exp()[..., None, None]
        key = k[:, t]
        prediction = (state * key[..., None]).sum(-2)
        correction = beta[:, t, :, None] * (v[:, t] - prediction)
        state = state + key[..., None] * correction[..., None, :]
        outputs.append((state * q[:, t, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1).to(output_dtype)


class DropPath(nn.Module):
    """Per-sample stochastic depth with the same scaling as timm.DropPath."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_path probability must be in [0, 1)")
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 1), nn.GELU())

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class PCS(nn.Module):
    """S2Mamba's original four-direction spatial mixer, with a reference scan."""

    def __init__(self, d_model, d_state=16, dropout=0.0, backend="auto"):
        super().__init__()
        if backend not in ("auto", "torch", "mamba"):
            raise ValueError("pcs_backend must be 'auto', 'torch', or 'mamba'")
        self.backend = backend
        self.d_model, self.d_state = d_model, d_state
        self.d_inner, self.dt_rank = 2 * d_model, math.ceil(d_model / 16)
        self.K = 4
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1,
                                groups=self.d_inner, bias=True)
        self.act = nn.SiLU()
        projections = [nn.Linear(self.d_inner, self.dt_rank + 2 * d_state,
                                  bias=False) for _ in range(self.K)]
        self.x_proj_weight = nn.Parameter(torch.stack([p.weight for p in projections]))
        dt_projections = [self._dt_init(self.dt_rank, self.d_inner)
                          for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([p.weight for p in dt_projections]))
        self.dt_projs_bias = nn.Parameter(torch.stack([p.bias for p in dt_projections]))
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_logs = nn.Parameter(A.log().repeat(self.K * self.d_inner, 1))
        self.Ds = nn.Parameter(torch.ones(self.K * self.d_inner))
        self.A_logs._no_weight_decay = True
        self.Ds._no_weight_decay = True
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _dt_init(rank, channels):
        projection = nn.Linear(rank, channels, bias=True)
        nn.init.uniform_(projection.weight, -(rank ** -0.5), rank ** -0.5)
        dt = torch.exp(torch.rand(channels) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp_min(1e-4)
        with torch.no_grad():
            projection.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        return projection

    def forward_core(self, x):
        batch, channels, height, width = x.shape
        length = height * width
        forward = torch.stack((x.flatten(2), x.transpose(2, 3).flatten(2)), dim=1)
        sequences = torch.cat((forward, forward.flip(-1)), dim=1)
        projected = torch.einsum("bkdl,kcd->bkcl", sequences, self.x_proj_weight)
        dt, B, C = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dt = torch.einsum("bkrl,kdr->bkdl", dt, self.dt_projs_weight)
        u = sequences.reshape(batch, self.K * channels, length).float()
        dt = dt.reshape(batch, self.K * channels, length).float()
        A = -self.A_logs.float().exp()
        B, C = B.float().contiguous(), C.float().contiguous()
        D, dt_bias = self.Ds.float(), self.dt_projs_bias.float().flatten()
        kernel = None
        if self.backend != "torch" and x.is_cuda:
            kernel = _optional_kernel("mamba_ssm.ops.selective_scan_interface",
                                      "selective_scan_fn")
        if self.backend == "mamba" and kernel is None:
            raise RuntimeError("pcs_backend='mamba' requires CUDA and a working mamba_ssm installation")
        if kernel is None:
            out = _selective_scan_reference(u, dt, A, B, C, D, dt_bias)
        else:
            out = kernel(u.contiguous(), dt.contiguous(), A, B, C, D,
                         delta_bias=dt_bias, delta_softplus=True)
        out = out.reshape(batch, self.K, channels, length)
        reverse = out[:, 2:4].flip(-1)
        column = out[:, 1].reshape(batch, channels, width, height).transpose(2, 3).flatten(2)
        reverse_column = reverse[:, 1].reshape(batch, channels, width, height).transpose(2, 3).flatten(2)
        y = out[:, 0] + reverse[:, 0] + column + reverse_column
        y = y.transpose(1, 2).reshape(batch, height, width, channels)
        return self.out_norm(y).to(x.dtype)

    def forward(self, x):
        content, gate = self.in_proj(x).chunk(2, dim=-1)
        content = self.act(self.conv2d(content.permute(0, 3, 1, 2).contiguous()))
        y = self.forward_core(content) * F.gelu(gate)
        return self.dropout(self.out_proj(y))


class GatedDeltaNet(nn.Module):
    """GDN sequence layer without language-model caches or Lit-GPT imports.

    Q/K/V causal short convolutions, L2-normalized keys/queries, Mamba-style
    decay, sigmoid beta and per-head RMSNorm/SiLU output gating follow the
    local NVLabs layer's default path. Returns a tensor [B, L, hidden_size].
    """

    def __init__(self, hidden_size=128, num_heads=4, expand_k=1.0,
                 expand_v=1.0, conv_size=4, norm_eps=1e-5, backend="auto"):
        super().__init__()
        if backend not in ("auto", "torch", "fla"):
            raise ValueError("gdn_backend must be 'auto', 'torch', or 'fla'")
        if hidden_size < 1 or num_heads < 1 or conv_size < 1:
            raise ValueError("GDN hidden_size, num_heads and conv_size must be positive")
        if not all(math.isfinite(ratio) and ratio > 0 for ratio in (expand_k, expand_v)):
            raise ValueError("GDN expansion ratios must be finite and positive")
        self.key_dim, self.value_dim = int(hidden_size * expand_k), int(hidden_size * expand_v)
        if (self.key_dim < num_heads or self.value_dim < num_heads
                or self.key_dim % num_heads or self.value_dim % num_heads):
            raise ValueError("GDN key_dim and value_dim must be positive multiples of num_heads")
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_qk_dim, self.head_v_dim = self.key_dim // num_heads, self.value_dim // num_heads
        self.backend, self.norm_eps = backend, norm_eps
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.gk_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.q_conv1d = self._short_conv(self.key_dim, conv_size)
        self.k_conv1d = self._short_conv(self.key_dim, conv_size)
        self.v_conv1d = self._short_conv(self.value_dim, conv_size)
        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(1e-4, 16).log())
        dt = torch.exp(torch.rand(num_heads) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.norm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    @staticmethod
    def _short_conv(channels, size):
        return nn.Conv1d(channels, channels, size, padding=size - 1,
                         groups=channels, bias=False)

    @staticmethod
    def _convolve(x, conv):
        # Right padding is cropped: output at t sees no future channel tokens.
        return F.silu(conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2))

    def _scan(self, q, k, v, beta, g):
        kernel = None
        if self.backend != "torch" and q.is_cuda:
            kernel = _optional_kernel("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")
        if self.backend == "fla" and kernel is None:
            raise RuntimeError("gdn_backend='fla' requires CUDA and a working flash-linear-attention installation")
        if kernel is None:
            return _gated_delta_rule_reference(q, k, v, beta, g)
        output_dtype = v.dtype
        # FLA kernels use half/bfloat16 Q/K/V even with an FP32 training loop.
        kernel_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        out, _ = kernel(q=q.to(kernel_dtype).contiguous(),
                        k=k.to(kernel_dtype).contiguous(),
                        v=v.to(kernel_dtype).contiguous(),
                        beta=beta.contiguous(), g=g.contiguous(),
                        initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=False, **_fla_layout_options(kernel))
        return out.to(output_dtype)

    def forward(self, hidden_states):
        if (hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size
                or hidden_states.shape[1] == 0):
            raise ValueError(f"GDN expects [B, L>0, {self.hidden_size}]")
        batch, length, _ = hidden_states.shape
        q = self._convolve(self.q_proj(hidden_states), self.q_conv1d)
        k = self._convolve(self.k_proj(hidden_states), self.k_conv1d)
        v = self._convolve(self.v_proj(hidden_states), self.v_conv1d)
        q = q.reshape(batch, length, self.num_heads, self.head_qk_dim)
        k = k.reshape(batch, length, self.num_heads, self.head_qk_dim)
        v = v.reshape(batch, length, self.num_heads, self.head_v_dim)
        q = F.normalize(q.float(), dim=-1, eps=1e-6).to(v.dtype)
        k = F.normalize(k.float(), dim=-1, eps=1e-6).to(v.dtype)
        g = -self.A_log.float().exp() * F.softplus(self.gk_proj(hidden_states).float()
                                                  + self.dt_bias.float())
        beta = self.b_proj(hidden_states).float().sigmoid()
        out = self._scan(q, k, v, beta, g)
        normalized = out.float() * torch.rsqrt(out.float().square().mean(-1, keepdim=True)
                                                + self.norm_eps)
        normalized = normalized * self.norm_weight.float()
        gate = self.g_proj(hidden_states).reshape(batch, length, self.num_heads, self.head_v_dim)
        out = (normalized * F.silu(gate.float())).to(v.dtype)
        return self.o_proj(out.reshape(batch, length, self.value_dim))


class BSS_GDN(nn.Module):
    """Original BSS wrapper with shared-parameter forward/backward GDN scans."""

    def __init__(self, d_model, d_spectral=121, dropout=0.0,
                 gdn_hidden_size=128, gdn_num_heads=4, gdn_expand_k=1.0,
                 gdn_expand_v=1.0, gdn_conv_size=4, gdn_backend="auto"):
        super().__init__()
        self.d_model, self.d_inner, self.d_spectral = d_model, 2 * d_model, d_spectral
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1,
                                groups=self.d_inner, bias=True)
        self.act = nn.SiLU()
        self.spec_in = nn.Linear(d_spectral, gdn_hidden_size, bias=False)
        self.gdn = GatedDeltaNet(hidden_size=gdn_hidden_size, num_heads=gdn_num_heads,
                                expand_k=gdn_expand_k, expand_v=gdn_expand_v,
                                conv_size=gdn_conv_size, backend=gdn_backend)
        self.spec_out = nn.Linear(gdn_hidden_size, d_spectral, bias=False)
        self.out_norm_spectral = nn.LayerNorm(d_spectral)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def reset_sequence_parameters(self):
        """Give the new adapter/GDN projections a usable initial signal scale.

        The baseline's std=0.02 initialization is retained for the outer BSS
        projections. Applying it to every new GDN projection compounded signal
        shrinkage, so the spectral LayerNorm operated in its epsilon-dominated
        regime. Xavier initialization is restricted to the new sequence core.
        Decay parameters, short convolutions and normalization stay unchanged.
        """
        for module in (self.spec_in, self.spec_out, *self.gdn.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_spectral(self, x):
        batch, channels, height, width = x.shape
        if height * width != self.d_spectral:
            raise ValueError(f"BSS_GDN requires {self.d_spectral} spatial positions")
        # Token axis = learned channel; feature axis = all spatial positions.
        tokens = self.spec_in(x.flatten(2))
        # Batch packing shares parameters, but recurrence/short-conv states stay
        # independent for each sample and direction.
        both = self.gdn(torch.cat((tokens, tokens.flip(1)), dim=0))
        forward, backward = both.chunk(2, dim=0)
        y = self.spec_out(forward + backward.flip(1))
        y = self.out_norm_spectral(y)
        return y.transpose(1, 2).reshape(batch, height, width, channels)

    def forward(self, x):
        content, gate = self.in_proj(x).chunk(2, dim=-1)
        content = self.act(self.conv2d(content.permute(0, 3, 1, 2).contiguous()))
        y = self.forward_spectral(content) * F.gelu(gate)
        return self.dropout(self.out_proj(y))


class FeedForward_softmax(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, 2), nn.Softmax(dim=-1))

    def forward(self, x):
        return self.net(x)


class SMG(nn.Module):
    def __init__(self, hidden_dim, drop_path=0.0):
        super().__init__()
        self.FFN_softmax = FeedForward_softmax(2 * hidden_dim)
        self.threshold = 0.1
        self.drop_path = DropPath(drop_path)

    def forward(self, spatial, spectral):
        gate = self.FFN_softmax(torch.cat((spatial, spectral), dim=-1))
        gate = torch.where(gate < self.threshold, torch.zeros_like(gate), gate)
        return self.drop_path(gate[..., :1] * spatial + gate[..., 1:] * spectral)


class MambaBlock_GDN(nn.Module):
    def __init__(self, hidden_dim, d_state, d_spectral, drop_path,
                 attn_drop, norm_layer, pcs_backend, gdn_options):
        super().__init__()
        self.ln = norm_layer(hidden_dim)
        self.PCS = PCS(hidden_dim, d_state=d_state, dropout=attn_drop, backend=pcs_backend)
        self.BSS = BSS_GDN(hidden_dim, d_spectral=d_spectral, dropout=attn_drop, **gdn_options)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = self.ln(x)
        return self.drop_path(self.PCS(x)), self.drop_path(self.BSS(x))


class MambaLayer_GDN(nn.Module):
    def __init__(self, dim, depth, d_state, d_spectral, drop_path,
                 attn_drop, norm_layer, pcs_backend, gdn_options):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock_GDN(dim, d_state, d_spectral, drop_path[i], attn_drop,
                           norm_layer, pcs_backend, gdn_options)
            for i in range(depth)
        ])
        # Preserve the baseline SMG's use of attn_drop as its DropPath rate.
        self.fusion = SMG(dim, drop_path=attn_drop)

    def forward(self, x):
        for block in self.blocks:
            spatial, spectral = block(x)
            x = x + self.fusion(spatial, spectral)
        return x


class S2Mamba_GDN(nn.Module):
    """S2Mamba classifier with a GDN BSS branch and unchanged spatial topology.

    Baseline arguments keep their meaning; d_state only controls PCS now.
    gdn_hidden_size controls the H*W -> GDN -> H*W adapter width.
    Stages must have equal widths: the baseline has no stage transitions.
    gdn_init='xavier' initializes only new adapter/GDN projections at Xavier
    scale; 'legacy' reproduces the first version's uniform std=0.02 scheme.
    """

    def __init__(self, patch=11, in_chans=154, num_classes=16, depths=(1,),
                 dims=(64,), d_state=16, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 gdn_hidden_size=128, gdn_num_heads=4, gdn_expand_k=1.0,
                 gdn_expand_v=1.0, gdn_conv_size=4, gdn_backend="auto",
                 pcs_backend="auto", gdn_init="xavier"):
        super().__init__()
        if patch < 1 or in_chans < 1:
            raise ValueError("patch and in_chans must be positive")
        depths = tuple(depths)
        if not depths or any(depth < 1 for depth in depths):
            raise ValueError("depths must contain positive block counts")
        if isinstance(dims, int):
            dims = tuple(dims * 2 ** i for i in range(len(depths)))
        else:
            dims = tuple(dims)
        if len(dims) != len(depths) or any(dim < 1 for dim in dims):
            raise ValueError("dims must contain one positive width per stage")
        if len(set(dims)) != 1:
            raise ValueError("S2Mamba has no stage transitions; all dims must be equal")
        d_state = math.ceil(dims[0] / 6) if d_state is None else d_state
        if d_state < 1:
            raise ValueError("d_state must be positive")
        if not all(0.0 <= rate < 1.0 for rate in (drop_rate, attn_drop_rate, drop_path_rate)):
            raise ValueError("dropout/drop_path rates must be in [0, 1)")
        if gdn_init not in ("xavier", "legacy"):
            raise ValueError("gdn_init must be 'xavier' or 'legacy'")
        self.gdn_init = gdn_init
        self.patch, self.in_chans = patch, in_chans
        self.num_classes, self.num_layers = num_classes, len(depths)
        self.embed_dim, self.num_features, self.dims = dims[0], dims[-1], dims
        self.patch_embed = PatchEmbed(in_chans, self.embed_dim)
        # drop_rate remains accepted for baseline compatibility; the baseline
        # does not use it in forward. attn_drop_rate controls branch dropout.
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        options = dict(gdn_hidden_size=gdn_hidden_size, gdn_num_heads=gdn_num_heads,
                       gdn_expand_k=gdn_expand_k, gdn_expand_v=gdn_expand_v,
                       gdn_conv_size=gdn_conv_size, gdn_backend=gdn_backend)
        self.layers = nn.ModuleList()
        offset = 0
        for dim, depth in zip(dims, depths):
            self.layers.append(MambaLayer_GDN(
                dim, depth, d_state, patch ** 2, dpr[offset:offset + depth],
                attn_drop_rate, norm_layer, pcs_backend, options))
            offset += depth
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        if gdn_init == "xavier":
            for layer in self.layers:
                for block in layer.blocks:
                    block.BSS.reset_sequence_parameters()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if (x.ndim != 4 or x.shape[1] != self.in_chans
                or tuple(x.shape[2:]) != (self.patch, self.patch)):
            raise ValueError(f"S2Mamba_GDN expects [B, {self.in_chans}, {self.patch}, {self.patch}]")
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        features = self.norm(x.flatten(1, 2))
        return self.head(self.avgpool(features.transpose(1, 2)).flatten(1))


# Also support the exact model name requested by the user as a constructor.
s2mamba_GDN = S2Mamba_GDN
