"""Independent, seeded repetitions with one process per selected GPU."""
import contextlib
import copy
import csv
from datetime import datetime
import json
import logging
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from model.s2mamba import S2Mamba
# from model.s2mamba_GDN import S2Mamba_GDN
# from model.demunet.model import MyModel
# from model.FBCANet import FBCANet
# from model.cpformer1 import CPFormer

# SSFTM 的导入配置保留供手动切换；使用 TCPI 时无需加载其 CUDA 扩展。
# import importlib.util
# _ssftm_dir = Path(__file__).resolve().parents[1] / "model" / "SSFTM-main"
# _original_sys_path = sys.path.copy()
# try:
#     sys.path.insert(0, str(_ssftm_dir))
#     import torch
#     import _C
#     _spec = importlib.util.spec_from_file_location(
#         "ssftm_model", _ssftm_dir / "model.py"
#     )
#     _ssftm = importlib.util.module_from_spec(_spec)
#     sys.modules["ssftm_model"] = _ssftm
#     _spec.loader.exec_module(_ssftm)
# finally:
#     sys.path[:] = _original_sys_path
# GrootV = _ssftm.GrootV

import numpy as np
import torch
from torch.utils.data import DataLoader
from fvcore.nn import parameter_count_table

from Config_for_datasets.Config import CONFIG_HSIC
from Mydataset.High_Dataset import MyDataset


from .train_function_new import train
from .test_function import test


def select_devices(gpus):
    if gpus is None:
        return ['cuda:0' if torch.cuda.is_available() else 'cpu']
    try:
        indices = [int(value.strip()) for value in gpus.split(',')]
    except ValueError as exc:
        raise ValueError('--gpus must be comma-separated visible GPU indices') from exc
    if len(indices) != len(set(indices)):
        raise ValueError('--gpus must not contain duplicate indices')
    count = torch.cuda.device_count()
    if not indices or any(index < 0 or index >= count for index in indices):
        raise ValueError(f'GPU index out of range: {count} CUDA device(s) visible')
    return [f'cuda:{index}' for index in indices]


def seed_experiment(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_loader_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def load_datasets(args):
    config = CONFIG_HSIC[args.datasets]
    if args.classes != config['num_classes']:
        raise ValueError('--classes must match the selected dataset num_classes')
    result = []
    for split in ('train', 'val', 'test'):
        data = MyDataset(config[f'{split}_data_path'], config[f'{split}_label_path'])
        expected = (config['bands'], args.windows, args.windows)
        if tuple(data.datas.shape[1:]) != expected:
            raise ValueError(f'{split} shape {tuple(data.datas.shape)}; expected [N, {expected}]')
        if data.labels.ndim != 1 or len(data) < args.batch_size:
            raise ValueError(f'{split} requires 1D labels and at least batch_size samples (drop_last=True)')
        if data.labels.min() < 0 or data.labels.max() >= args.classes:
            raise ValueError(f'{split} labels must be in [0, {args.classes - 1}]')
        result.append(data)
    return result


def make_loaders(args, datasets, seed):
    loaders = []
    for offset, dataset in enumerate(datasets):
        options = dict(batch_size=args.batch_size, shuffle=True, drop_last=True,
                       num_workers=args.num_workers,
                       pin_memory=args.device.type == 'cuda' and not args.no_pin_memory,
                       generator=torch.Generator().manual_seed(seed + offset),
                       worker_init_fn=seed_loader_worker)
        if args.num_workers:
            options.update(persistent_workers=True, prefetch_factor=args.prefetch_factor,
                           multiprocessing_context='spawn')
        loaders.append(DataLoader(dataset, **options))
    return loaders


def get_model(args):
    #FBCANet 网络代码
    # model = FBCANet(in_channels=CONFIG_HSIC[args.datasets]['bands'],
    #                num_classes=args.classes, dim=64, reduction=16,
    #                patch_size=args.windows).to(args.device)
    #demunet 网络代码
    #model = MyModel(batch_size=args.batch_size, bilinear=True).to(args.device)
#S2Mamba 网络代码
    model = S2Mamba(
    in_chans=CONFIG_HSIC[args.datasets]['bands'],
    patch=args.windows,
    num_classes=args.classes,
    depths=[1],
    dims=[64],
    drop_path_rate=0.1,
    attn_drop_rate=0.1,
).to(args.device)

# s2mamba_GDN：保留 PCS/SMG，将 BSS 改为共享参数的双向 GatedDeltaNet。
    # model = S2Mamba_GDN(
    #     in_chans=CONFIG_HSIC[args.datasets]['bands'],
    #     patch=args.windows,
    #     num_classes=args.classes,
    #     depths=[1],
    #     dims=[64],
    #     drop_path_rate=0.1,
    #     attn_drop_rate=0.1,
    #     gdn_hidden_size=128,
    #     gdn_num_heads=4,
    #     gdn_init='xavier',  # 'legacy' 可复现第一版初始化；修改后需重新训练。
    #     pcs_backend='auto',
    #     gdn_backend='auto',
    # ).to(args.device)

# CPFormer 网络代码
    # model = CPFormer(
    #     in_channels=CONFIG_HSIC[args.datasets]["bands"],
    #     num_classes=args.classes,
    #     dim=256 if args.datasets == "IP" else 64,
    # ).to(args.device)

#SSFTM 网络代码
    # model = GrootV(
    #     in_chans=CONFIG_HSIC[args.datasets]["bands"],
    #     num_classes=args.classes,
    #     channels=128,
    #     depths=[1],
    #     act_layer="SiLU",
    #     norm_layer="LN",
    #     spe_fusion="concat",
    #     use_ssf=True,
    #     scan_strategy="both_spacausal_spenoncausal",
    # ).to(args.device)


    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, dampening=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - epoch / args.epochs)
    return model, optimizer, scheduler
def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def run_worker(args, device, iteration_ids, run_dir):
    args = copy.copy(args)
    args.device = torch.device(device)
    torch.set_num_threads(args.cpu_threads)
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)
    started = time.perf_counter()
    datasets = load_datasets(args)
    initialization_seconds = time.perf_counter() - started
    for index in iteration_ids:
        directory = Path(run_dir) / f'iter_{index:03d}'
        directory.mkdir()
        with (directory / 'experiment.log').open('w', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                logging.basicConfig(stream=log, level=logging.INFO, force=True)
                try:
                    seed = args.seed + index
                    seed_experiment(seed)
                    loaders = make_loaders(args, datasets, seed)
                    if args.device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(args.device)
                        torch.cuda.synchronize(args.device)
                    started = time.perf_counter()
                    model, optimizer, scheduler = get_model(args)
                    criterion = torch.nn.CrossEntropyLoss().to(args.device)
                    model = train(args, index, model, loaders[0], loaders[1], criterion,
                                  optimizer, scheduler, args.epochs,
                                  history_path=directory / 'epoch_metrics.jsonl')
                    oa, aa, kappa, report, class_acc, test_seconds, diagnostics = test(
                        args, index, loaders[2], model, return_diagnostics=True)
                    if args.device.type == 'cuda':
                        torch.cuda.synchronize(args.device)
                    elapsed = time.perf_counter() - started
                    result = dict(iteration=index, seed=seed, device=device,
                                  oa=float(oa), aa=float(aa), kappa=float(kappa),
                                  class_accuracy=class_acc.tolist(), report=report,
                                  test_seconds=test_seconds, experiment_seconds=elapsed,
                                  data_initialization_seconds=initialization_seconds,
                                  epoch_times=args.epoch_times,
                                  training_history=args.training_history,
                                  accuracy_unit='fraction',
                                  confusion_matrix_axes={'rows': 'true', 'columns': 'predicted'},
                                  confusion_matrix_labels=diagnostics['class_labels'],
                                  test_confusion_matrix=diagnostics['confusion_matrix'],
                                  test_class_support=diagnostics['class_support'],
                                  test_prediction_counts=diagnostics['prediction_counts'],
                                  peak_cuda_bytes=(torch.cuda.max_memory_allocated(args.device)
                                                   if args.device.type == 'cuda' else 0))
                    # Full CPU model retains the previous checkpoint format and is portable.
                    torch.save(model.cpu(), directory / 'model.pth')
                    write_json(directory / 'metrics.json', result)
                    with (directory / 'test_confusion_matrix.csv').open(
                            'w', newline='', encoding='utf-8') as stream:
                        writer = csv.writer(stream)
                        writer.writerow(['true_class'] + [f'pred_{c}' for c in diagnostics['class_labels']])
                        for label, row in zip(diagnostics['class_labels'], diagnostics['confusion_matrix']):
                            writer.writerow([label] + row)
                    del model, optimizer, scheduler, loaders, criterion
                except BaseException:
                    traceback.print_exc()
                    raise
                finally:
                    # Remove handlers before the redirected log file is closed.
                    root_logger = logging.getLogger()
                    for handler in root_logger.handlers[:]:
                        root_logger.removeHandler(handler)
                        handler.close()
        print(f'Finished experiment {index} on {device}: OA={oa:.6f}', flush=True)


def dispatch(args, devices, run_dir):
    devices = devices[:args.iters]
    assignments = [list(range(offset, args.iters, len(devices))) for offset in range(len(devices))]
    if len(devices) == 1:
        run_worker(args, devices[0], assignments[0], run_dir)
        return
    context = mp.get_context('spawn')
    processes = []
    try:
        for device, indices in zip(devices, assignments):
            process = context.Process(target=run_worker, args=(args, device, indices, str(run_dir)))
            process.start()
            processes.append(process)
        while True:
            for process in processes:
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f'Experiment worker exited with {process.exitcode}; inspect logs in {run_dir}')
            if all(process.exitcode == 0 for process in processes):
                break
            time.sleep(0.2)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()


def summarize(args, run_dir, total_seconds):
    results = []
    for index in range(args.iters):
        result = json.loads((run_dir / f'iter_{index:03d}' / 'metrics.json').read_text(encoding='utf-8'))
        if result['iteration'] != index:
            raise ValueError('Experiment result ID mismatch')
        if len(result['class_accuracy']) != args.classes:
            raise ValueError(f'Experiment {index} did not report all classes; check evaluation sample coverage')
        if not np.isfinite([result['oa'], result['aa'], result['kappa']]).all():
            raise ValueError(f'Experiment {index} has non-finite metrics')
        results.append(result)
    best = max(results, key=lambda result: result['oa'])  # first wins ties, as before
    scores = np.array([[r['oa'], r['aa'], r['kappa']] for r in results])
    classes = np.array([r['class_accuracy'] for r in results])
    # A sample standard deviation needs >=2 repeats; use null for a smoke run.
    score_std = scores.std(axis=0, ddof=1).tolist() if len(results) > 1 else None
    class_std = classes.std(axis=0, ddof=1).tolist() if len(results) > 1 else None
    best_path = run_dir / f"iter_{best['iteration']:03d}" / 'model.pth'
    shutil.copyfile(best_path, run_dir / 'best_model.pth')
    legacy_path = Path(args.output_dir) / args.datasets / f'{args.exp_name}_ab{{{args.ablation}}}{args.datasets}.pth'
    shutil.copyfile(best_path, legacy_path)
    summary = dict(best_iteration=best['iteration'], metrics_order=['OA', 'AA', 'Kappa'],
                   mean=scores.mean(axis=0).tolist(), sample_std=score_std,
                   class_mean=classes.mean(axis=0).tolist(), class_sample_std=class_std,
                   mean_test_seconds=float(np.mean([r['test_seconds'] for r in results])),
                   total_seconds=total_seconds, experiments=results)
    write_json(run_dir / 'summary.json', summary)
    # This file was just produced locally by this run; do not load untrusted checkpoints.
    model = torch.load(best_path, map_location='cpu', weights_only=False)
    print(parameter_count_table(model))
    print(best['report'])
    for name, mean, std in zip(summary['metrics_order'], summary['mean'], score_std or [None] * 3):
        spread = f'{std * 100:.2f}' if std is not None else 'N/A (one repeat)'
        print(f'{name}: {mean * 100:.2f} +/- {spread}')
    for index, mean in enumerate(summary['class_mean']):
        spread = f'{class_std[index] * 100:.2f}' if class_std is not None else 'N/A (one repeat)'
        print(f'Class {index + 1}: {mean * 100:.2f} +/- {spread}')
    print(f"Mean test time: {summary['mean_test_seconds']:.3f}s; total: {total_seconds:.3f}s")
    print(f'Results: {run_dir}\nModel: {legacy_path}')
    return summary


def execute(args, devices):
    if args.datasets not in CONFIG_HSIC:
        raise ValueError(f'Unknown dataset: {args.datasets}')
    if not devices or args.iters < 1:
        raise ValueError('At least one device and one experiment are required')
    # Fail before launching processes if a configured input file is absent.
    for name, value in CONFIG_HSIC[args.datasets].items():
        if name in {f'{split}_{kind}_path' for split in ('train', 'val', 'test') for kind in ('data', 'label')}:
            if not Path(value).is_file():
                raise FileNotFoundError(f'Missing {name}: {Path(value).resolve()}')
    run_id = args.exp_name + datetime.now().strftime('%m%d')
    run_dir = Path(args.output_dir).resolve() / args.datasets / 'runs' / run_id
    run_dir.mkdir(parents=True)
    write_json(run_dir / 'config.json', dict(arguments=vars(args), devices=devices,
                                           torch_version=torch.__version__,
                                           cuda_version=torch.version.cuda))
    print(f'Experiment logs and outputs: {run_dir}', flush=True)
    started = time.perf_counter()
    dispatch(args, devices, run_dir)
    summarize(args, run_dir, time.perf_counter() - started)
    return run_dir
