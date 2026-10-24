"""Launch one forecasting experiment through the public reproduction entrypoint."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    'web': ('quito_hour', 'extended_wikipedia', 'tailedts_202401'),
    'general': ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather'),
}


def build_command(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Extra reproduce.py options can follow --, e.g. -- --no_use_gpu '
        '--save_predictions. Relative paths are resolved from your current directory.'
    ))
    parser.add_argument('--framework', choices=('raft', 'pir', 'knn_mts'), default='raft')
    parser.add_argument('--data-type', choices=DATASETS, required=True)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Web: root containing dataset directories; General: directory containing CSV files')
    parser.add_argument('--dataset', required=True,
                        choices=DATASETS['web'] + DATASETS['general'])
    parser.add_argument('--method', choices=('BASE', 'RAFT', 'PIR', 'KNN', 'SHARP', 'CREST', 'S&C'), required=True)
    parser.add_argument('--scale', type=int, help='Web cohort label (default: 200000)')
    parser.add_argument('--pred-len', type=int, help='Forecast horizon; otherwise use the dataset default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, help='New output directory; otherwise generate a run directory')
    parser.add_argument('--dry-run', action='store_true', help='Check configuration without retrieval or training')
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='Additional reproduce.py arguments after --')
    args = parser.parse_args(argv)
    if args.dataset not in DATASETS[args.data_type]:
        parser.error('--dataset does not belong to --data-type')
    if args.data_type == 'general' and args.scale is not None:
        parser.error('--scale applies only to web datasets')
    if not args.data_dir.is_dir():
        parser.error('--data-dir must be an existing directory')
    extra = args.extra
    if extra:
        if extra[0] != '--':
            parser.error('additional reproduce.py arguments must follow --')
        extra = extra[1:]

    if args.framework != 'raft':
        command = [sys.executable, str(ROOT / 'scripts/train_model.py'),
                   '--framework', args.framework, '--data-type', args.data_type,
                   '--data-dir', str(args.data_dir.resolve()), '--dataset', args.dataset,
                   '--method', args.method, '--seed', str(args.seed)]
        for flag, value in (('--scale', args.scale), ('--pred-len', args.pred_len),
                            ('--output', args.output.resolve() if args.output else None)):
            if value is not None:
                command += [flag, str(value)]
        if args.dry_run:
            command.append('--dry-run')
        return command + (['--'] + extra if extra else [])
    if args.method == 'BASE':
        args.method = 'RAFT'
    if args.method not in ('RAFT', 'SHARP', 'CREST', 'S&C'):
        parser.error('RAFT framework requires BASE/RAFT, SHARP, CREST or S&C')

    command = [sys.executable, str(ROOT / 'reproduce.py'),
               '--dataset', args.dataset, '--method', args.method, '--seed', str(args.seed)]
    if args.data_type == 'web':
        command += ['--canonical_root', str(args.data_dir.resolve()),
                    '--scale', str(args.scale if args.scale is not None else 200000)]
    else:
        command += ['--root_path', str(args.data_dir.resolve())]
    if args.pred_len is not None:
        command += ['--pred_len', str(args.pred_len)]
    if args.output is not None:
        command += ['--output', str(args.output.resolve())]
    if args.dry_run:
        command.append('--dry_run')
    return command + extra


def main():
    command = build_command()
    print('Running: ' + shlex.join(command), flush=True)
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
