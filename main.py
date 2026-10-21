
import argparse
from functions.experiments import execute, select_devices
from functions.modelselect import DEFAULT_MODEL, get_model_names


def experiment_name(value):
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError('exp_name must not be empty or whitespace')
    return value


def build_parser():
    parser = argparse.ArgumentParser("HSI patch classification")
    parser.add_argument('--exp_name', type=experiment_name, required=True, help='experiment name (required)')
    parser.add_argument('--model', choices=get_model_names(), default=DEFAULT_MODEL,
                        help=f'classification model (default: {DEFAULT_MODEL})')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='checkpoints direction')
    # Training Setting
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--iters', type=int, default=10, help='number of experiment repeats')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--seed', type=int, default=0, help='training seed')
    parser.add_argument('--step_size', type=int, default=10, help='step_size')
    parser.add_argument('--gamma', type=float, default=0.4, help='gamma')

    parser.add_argument('--data_root', type=str, default='./datasets/', help='datasets dir')
    parser.add_argument('--classes', type=int, default=15, help='datasets classes')
    parser.add_argument('--datasets', type=str, default='HUST2013', help='path to the datasets')
    parser.add_argument('--ablation', type=str, default='1', help='ablation')
    parser.add_argument('--windows', type=int, default=11, help='size of patches')
    parser.add_argument('--group_size', type=int, default=10,
                        help='consecutive spectral bands per TCPI stage; zero-pad the last group')
    parser.add_argument('--cutout', action='store_true', help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')


    parser.add_argument('--gpus', type=str, default=None,
                        help='comma-separated visible GPU indices, e.g. 0,1; one experiment per GPU')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers per loader/process')
    parser.add_argument('--no-pin-memory', action='store_true', help='disable pinned memory on CUDA')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='prefetched batches per loader worker')
    parser.add_argument('--cpu_threads', type=int, default=1, help='PyTorch CPU threads per experiment process')
    parser.add_argument('--output_dir', default='model_pth', help='model and experiment output directory')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    for name in ('batch_size', 'epochs', 'iters', 'prefetch_factor', 'cpu_threads', 'group_size'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.num_workers < 0:
        parser.error('num_workers must be non-negative')
    try:
        devices = select_devices(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))
    execute(args, devices)


if __name__ == '__main__':
    main()
