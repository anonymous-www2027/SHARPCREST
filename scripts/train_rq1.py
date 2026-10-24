"""Run selected PIR/kNN-MTS RQ1 training conditions with the published defaults."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reproduction.model_data import GENERAL, WEB_LENGTHS, HORIZONS, general_csv
from src.reproduction.model_sweep import main as run_sweep


def method_name(value):
    aliases = {'BASE': 'BASE', 'SHARP': 'SHARP', 'CREST': 'CREST',
               'SHARP&CREST': 'S&C', 'SHARP+CREST': 'S&C', 'S&C': 'S&C'}
    try:
        return aliases[value.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError('Choose BASE, SHARP, CREST or SHARP&CREST')


def dataset_name(value):
    names = {name.lower(): name for name in (*GENERAL, *WEB_LENGTHS)}
    names.update(wiki='extended_wikipedia', tailedts='tailedts_202401')
    try:
        return names[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError('Unknown RQ1 dataset: '+value)


def build_sweep_args(argv=None):
    """Validate the complete selection before starting any training run."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework', choices=['pir', 'knn_mts'], required=True)
    p.add_argument('--suite', choices=['web', 'general'], required=True)
    p.add_argument('--methods', '--method', nargs='+', type=method_name,
                   default=['BASE', 'SHARP', 'CREST', 'S&C'],
                   help="BASE SHARP CREST 'SHARP&CREST'; default: all four")
    p.add_argument('--datasets', nargs='+', type=dataset_name,
                   help='Default: all datasets in the selected suite')
    p.add_argument('--horizons', nargs='+', type=int, help='Default: all RQ1 horizons for each dataset')
    p.add_argument('--seeds', nargs='+', type=int,
                   help='Default: web 0/1/42; general 0/1/2 except ETTm1 0/1/10')
    p.add_argument('--data-dir', type=Path, help='Bundled General/Web root or an external data root')
    p.add_argument('--tailedts-data-dir', type=Path,
                   help='External root containing tailedts_202401/; Wiki/Quito can remain bundled')
    p.add_argument('--output-root', type=Path, help='Default: outputs/rq1-<framework>-<suite>')
    p.add_argument('--dry-run', action='store_true', help='Print commands without creating outputs or training')
    a = p.parse_args(argv)
    available = tuple(WEB_LENGTHS) if a.suite == 'web' else GENERAL
    datasets = a.datasets or available
    if any(ds not in available for ds in datasets):
        p.error('--datasets must belong to the selected suite')
    if a.tailedts_data_dir and (a.suite != 'web' or 'tailedts_202401' not in datasets):
        p.error('--tailedts-data-dir requires TailedTS in the selected web datasets')
    for ds in datasets:
        allowed = HORIZONS[ds] if a.suite == 'web' else [96, 192, 336, 720]
        if a.horizons and any(h not in allowed for h in a.horizons):
            p.error(f'{ds} supports horizons {allowed}; select it separately for other horizons')
    if a.seeds and any(s < 0 or s >= 2**32 for s in a.seeds):
        p.error('Seeds must be in [0, 2**32)')
    for values in (datasets, a.methods, a.horizons or [], a.seeds or []):
        if len(values) != len(set(values)):
            p.error('Duplicate dataset/method/horizon/seed selections are not allowed')
    data = (a.data_dir or ROOT/'data'/a.suite).resolve()
    if not a.dry_run:
        for ds in datasets:
            if a.suite == 'general':
                try:
                    general_csv(data, ds)
                except FileNotFoundError as exc:
                    p.error(str(exc))
            else:
                source = a.tailedts_data_dir.resolve() if ds == 'tailedts_202401' and a.tailedts_data_dir else data
                if (source/ds/'package.json').is_file():
                    present = (source/ds/'200000.parquet').is_file()
                else:
                    directories = [ds]
                    if ds.startswith('quito_'):
                        manifest = json.loads((ROOT/'configs/cohorts'/ds/'200000.json').read_text())
                        directories = {row[1] for row in manifest['selected_identities']}
                    present = all((source/name/file).is_file() for name in directories
                                  for file in ('series_index.parquet', 'metadata.json'))
                if not present:
                    p.error(f'Missing data for {ds} under {source}; set --data-dir or --tailedts-data-dir')
    out = (a.output_root or ROOT/'outputs'/f'rq1-{a.framework}-{a.suite}').resolve()
    result = ['--framework', a.framework, '--suite', a.suite, '--data-dir', str(data),
              '--output-root', str(out), '--datasets', *datasets, '--methods', *a.methods]
    if a.tailedts_data_dir:
        result += ['--tailedts-data-dir', str(a.tailedts_data_dir.resolve())]
    if a.horizons:
        result += ['--horizons', *map(str, a.horizons)]
    if a.seeds:
        result += ['--seeds', *map(str, a.seeds)]
    if not a.dry_run:
        result.append('--execute')
    return result


def main(argv=None):
    run_sweep(build_sweep_args(argv))


if __name__ == '__main__':
    main()
