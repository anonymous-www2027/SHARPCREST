"""Portable launcher, grid, and release-export checks without research files."""

import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from src.reproduction.config import METHODS, ROOT, parse_args
from scripts.train_forecasting import DATASETS, build_command


@pytest.mark.parametrize('dataset', DATASETS['web'] + DATASETS['general'])
@pytest.mark.parametrize('method', METHODS)
def test_training_launcher_routes_data_and_method(tmp_path, dataset, method):
    kind = 'web' if dataset in DATASETS['web'] else 'general'
    directory = tmp_path / 'data with spaces'
    directory.mkdir()
    command = build_command(['--data-type', kind, '--data-dir', str(directory),
                             '--dataset', dataset, '--method', method,
                             '--', '--no_use_gpu', '--save_predictions'])
    args = parse_args(command[2:])
    assert Path(args.canonical_root if kind == 'web' else args.root_path) == directory
    assert (args.retrieval_backend, args.retrieval_controller) == METHODS[method]
    assert args.is_training == 1 and not args.retrieval_only and not args.prepare_only
    assert args.save_predictions and not args.use_gpu


@pytest.mark.parametrize('external', [False, True])
def test_grid_preserves_bundled_default_or_explicit_root(tmp_path, external):
    command = [sys.executable, '-m', 'src.reproduction.sweep', '--suite', 'web',
               '--datasets', 'extended_wikipedia', '--horizons', '7', '--seeds', '42',
               '--methods', 'SHARP', '--output-root', str(tmp_path / 'runs')]
    if external:
        command += ['--data-root', str(tmp_path / 'external')]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    child = shlex.split(output.strip())
    args = parse_args(child[2:])
    if external:
        assert Path(args.canonical_root) == tmp_path / 'external/web/processed'
    else:
        assert '--data_root' not in child
        if not Path('data/web/processed').exists():
            assert Path(args.canonical_root) == ROOT / 'data/web'


def test_export_contains_only_manifest_files(tmp_path):
    destination = tmp_path / 'public copy'
    command = [sys.executable, str(ROOT / 'scripts/export_public.py'), '--output', str(destination)]
    summary = json.loads(subprocess.check_output(command, text=True))
    expected = set((ROOT / 'PUBLIC_FILES.txt').read_text().splitlines())
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
    assert actual == expected and summary['files'] == len(expected)
    assert not (destination / '.git').exists()
    assert subprocess.run(command, capture_output=True).returncode != 0


@pytest.mark.parametrize('dataset', ['extended_wikipedia', 'ETTh1', 'Exchange'])
def test_repository_defaults_work_outside_checkout(tmp_path, dataset):
    output = subprocess.check_output([
        sys.executable, str(ROOT / 'reproduce.py'), '--dataset', dataset,
        '--method', 'S&C', '--dry_run', '--no_use_gpu',
    ], cwd=tmp_path, text=True)
    args = json.loads(output)['arguments']
    assert Path(args['data_root']) == ROOT / 'data'
    if dataset == 'extended_wikipedia':
        assert Path(args['canonical_root']) == ROOT / 'data/web'
    else:
        folder = 'ETT' if dataset.startswith('ETT') else 'experiment'
        assert Path(args['root_path']) == ROOT / 'data/general' / folder
        assert (Path(args['root_path']) / args['data_path']).is_file()


@pytest.mark.parametrize('preset', ['rq1_web200k.json', 'rq1_general.json'])
def test_presets_resolve_repository_data(tmp_path, monkeypatch, preset):
    monkeypatch.chdir(tmp_path)
    args = parse_args(['--config', str(ROOT / 'configs' / preset), '--no_use_gpu'])
    assert Path(args.data_root) == ROOT / 'data'
    if preset == 'rq1_web200k.json':
        assert Path(args.canonical_root) == ROOT / 'data/web'
    else:
        assert (Path(args.root_path) / args.data_path).is_file()


def test_export_uses_only_new_module_layout():
    names = set((ROOT / 'PUBLIC_FILES.txt').read_text().splitlines())
    legacy = {'Datasets', 'models', 'layers', 'data_provider', 'exp', 'common',
              'utils', 'reproduction', 'experiments'}
    assert not any(Path(name).parts[0] in legacy for name in names)
    assert 'run.py' not in names and 'src/run.py' in names
    assert {'data/README.md', 'src/__init__.py', 'configs/legacy_frozen_cohorts.json'} <= names
