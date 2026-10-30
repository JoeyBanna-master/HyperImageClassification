"""模型注册、按需导入和分类输出适配。

Add a builder and one MODEL_BUILDERS entry to register another model.
Builders keep the previous experiment settings; training settings stay in
experiments.py. Importing this module never imports a model/CUDA extension.
"""

import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import torch
from torch import nn

from Config_for_datasets.Config import CONFIG_HSIC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = 's2mamba'


class SQSFormerClassifier(nn.Module):
    """SQSFormer 返回 (logits,left,right)，统一为 [B,K] logits。"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, patches):
        return self.backbone(patches)[0]


def _load_ssftm_module():
    """保留已有 ssftm_model 模块名，支持完整模型序列化和旧权重加载。"""
    if 'ssftm_model' in sys.modules:
        return sys.modules['ssftm_model']
    model_directory = PROJECT_ROOT / 'model' / 'SSFTM-main'
    original_search_path = sys.path.copy()
    try:
        sys.path.insert(0, str(model_directory))
        importlib.import_module('_C')
        module_spec = importlib.util.spec_from_file_location(
            'ssftm_model', model_directory / 'model.py'
        )
        module = importlib.util.module_from_spec(module_spec)
        sys.modules['ssftm_model'] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop('ssftm_model', None)
            raise
        return module
    finally:
        sys.path[:] = original_search_path


def _build_s2mamba(args, dataset_config):
    if torch.device(args.device).type != 'cuda':
        raise RuntimeError('s2mamba 的原始 selective_scan 实现需要 CUDA；请选择可用 GPU')
    # 显式检查依赖：原模型的 try/except 会吞掉导入失败，随后变成 NameError。
    importlib.import_module('mamba_ssm.ops.selective_scan_interface')
    from model.s2mamba import S2Mamba
    return S2Mamba(
        in_chans=dataset_config['bands'], patch=args.windows,
        num_classes=dataset_config['num_classes'], depths=[1], dims=[64],
        drop_path_rate=0.1, attn_drop_rate=0.1,
    )


def _build_demunet(args, dataset_config):
    from model.demunet.model import MyModel
    return MyModel(
        batch_size=args.batch_size, bilinear=True,
        in_channels=dataset_config['bands'],
        num_classes=dataset_config['num_classes'], patch_size=args.windows,
    )


def _build_cpformer(args, dataset_config):
    from model.cpformer1 import CPFormer
    return CPFormer(
        in_channels=dataset_config['bands'], num_classes=dataset_config['num_classes'],
        dim=256 if args.datasets == 'IP' else 64,
    )


def _build_ssftm(args, dataset_config):
    if torch.device(args.device).type != 'cuda':
        raise RuntimeError('ssftm 的树扫描 CUDA 扩展需要 GPU')
    module = _load_ssftm_module()
    return module.GrootV(
        in_chans=dataset_config['bands'], num_classes=dataset_config['num_classes'],
        channels=128, depths=[1], act_layer='SiLU', norm_layer='LN',
        spe_fusion='concat', use_ssf=True,
        scan_strategy='both_spacausal_spenoncausal',
    )


def _build_hirwkv(args, dataset_config):
    if torch.device(args.device).type != 'cuda':
        raise RuntimeError('hirwkv 的 WKV CUDA 扩展需要 GPU')
    module = importlib.import_module('model.Hi-RWKV-main.model.HiRWKV_model')
    backbone = module.HiRWKV(
        in_channels=dataset_config['bands'], hidden_dim=128,
        num_classes=dataset_config['num_classes'], group_num=1,
        dataset_name=args.datasets,
    )
    return nn.Sequential(backbone, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))


def _build_hybridformer(args, dataset_config):
    from model.HybridFormer.vit_SS import ViT
    return ViT(
        image_size=args.windows, patch_size=[3, 5],
        num_classes=dataset_config['num_classes'], channels=dataset_config['bands'],
        dim=100, depth=4, heads=2, mlp_dim=2048, dropout=0.2, emb_dropout=0.2,
    )


def _build_sqsformer(args, dataset_config):
    from model.SQSFormer.src.models.SQSFormer import SQSFormer
    params_path = PROJECT_ROOT / 'model' / 'SQSFormer' / 'src' / 'params_use' / 'sqsformer.json'
    params = json.loads(params_path.read_text(encoding='utf-8'))
    params['data'].update(
        spectral_size=dataset_config['bands'], num_classes=dataset_config['num_classes'],
        patch_size=args.windows,
    )
    return SQSFormerClassifier(SQSFormer(params))


def _build_morphformer(args, dataset_config):
    # 原模块在导入时会执行 CPU summary，隔离打印和随机数消耗。
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        from model.morphFormer import model as morphformer
    morphformer.patchsize = args.windows
    backbone = morphformer.CNN(
        FM=16, NC=dataset_config['bands'], Classes=dataset_config['num_classes'], HSIOnly=True,
    )
    classifier = backbone.out3
    backbone.out3 = nn.Identity()
    return nn.Sequential(backbone, classifier)
def _build_basemodelv1(args, dataset_config):
    if torch.device(args.device).type != 'cuda':
        raise RuntimeError(
            'hsi_hybrid 的 Mamba 模块需要 CUDA；请选择可用 GPU'
        )

    # 检查 Mamba 依赖
    importlib.import_module('mamba_ssm.modules.mamba_simple')

    from model.basemodelv1 import HSIHybridBaseline

    return HSIHybridBaseline(
        num_bands=dataset_config['bands'],
        num_classes=dataset_config['num_classes'],
        patch_size=args.windows,
        dim=64,
        num_heads=4,
    )
    
def _build_basemodelRestoreMixer(args, dataset_config):
    if torch.device(args.device).type != 'cuda':
        raise RuntimeError(
            'hsi_hybrid 的 Mamba 模块需要 CUDA；请选择可用 GPU'
        )

    # 检查 Mamba 依赖
    importlib.import_module('mamba_ssm.modules.mamba_simple')

    from model.basemodelRestoreMixer import HSIHybridBaseline
   
    return HSIHybridBaseline(
        num_bands=dataset_config['bands'],
        num_classes=dataset_config['num_classes'],
        patch_size=args.windows,
        dim=64,
        num_heads=4,
    )

def _build_scab_baseline(args, dataset_config):
    from model.basemodel.scab_baseline import CrossFusionBaseline

    return CrossFusionBaseline(
        num_bands=dataset_config['bands'],
        num_classes=dataset_config['num_classes'],
        dim=32,
        num_groups=4,
    )
def _build_base_c(args, dataset_config):
    from model.basemodel.base_c import CrossFusionBaseline

    return CrossFusionBaseline(
        num_bands=dataset_config['bands'],
        num_classes=dataset_config['num_classes'],
        dim=32,
        num_groups=4,
    )

# 只注册用户指定的模型。结构超参数在对应 builder 中维护。
MODEL_BUILDERS = {
    's2mamba': _build_s2mamba,
    'demunet': _build_demunet,
    'cpformer': _build_cpformer,
    'ssftm': _build_ssftm,
    'hirwkv': _build_hirwkv,
    'hybridformer': _build_hybridformer,
    'sqsformer': _build_sqsformer,
    'morphformer': _build_morphformer,
    'basemodelv1':_build_basemodelv1,
    'basemodelRestoreMixer':_build_basemodelRestoreMixer,
    'scab_baseline': _build_scab_baseline,
    'base_c':_build_base_c,
}



def get_model_names():
    return tuple(MODEL_BUILDERS)


def build_model(args):
    """按照 --model 创建 [B,C,H,W] -> [B,K] 分类模型，并移动到目标设备。"""
    if args.model not in MODEL_BUILDERS:
        raise ValueError(f'Unknown model: {args.model}; choose from {get_model_names()}')
    if args.datasets not in CONFIG_HSIC:
        raise ValueError(f'Unknown dataset: {args.datasets}')
    dataset_config = CONFIG_HSIC[args.datasets]
    if args.classes != dataset_config['num_classes']:
        raise ValueError('--classes must match the selected dataset num_classes')
    if args.windows < 5 or args.windows % 2 == 0:
        raise ValueError('Registered patch models require an odd --windows of at least 5')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'{args.model}: CUDA was requested but is unavailable')
    try:
        model = MODEL_BUILDERS[args.model](args, dataset_config)
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(f'Cannot build {args.model}: {error}') from error
    return model.to(args.device)


def load_model_checkpoint(checkpoint_path, model_name):
    """加载本项目可信的完整模型；主进程也要能找到子进程中的动态模块。"""
    if model_name == 'ssftm':
        try:
            _load_ssftm_module()
        except (ImportError, OSError) as error:
            raise RuntimeError(f'Cannot load ssftm checkpoint: {error}') from error
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
