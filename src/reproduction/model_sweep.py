"""Print or execute the complete PIR/kNN-MTS RQ1 grid with shared backbones."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from src.reproduction.model_data import ROOT, GENERAL, WEB_LENGTHS, HORIZONS
from src.reproduction.models import METHODS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework', choices=['pir', 'knn_mts'], required=True)
    p.add_argument('--suite', choices=['general', 'web'], required=True)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--tailedts-data-dir', type=Path, help='Separate external root containing tailedts_202401/')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--horizons', nargs='+', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    available = tuple(WEB_LENGTHS) if a.suite == 'web' else GENERAL
    datasets = a.datasets or available
    if any(ds not in available for ds in datasets):
        p.error('Dataset does not belong to suite')
    if a.tailedts_data_dir and (a.suite != 'web' or 'tailedts_202401' not in datasets):
        p.error('--tailedts-data-dir requires TailedTS in the selected web datasets')
    root = a.output_root.resolve()
    for ds in datasets:
        data_dir = (a.tailedts_data_dir if ds == 'tailedts_202401' and a.tailedts_data_dir
                    else a.data_dir or ROOT/'data'/a.suite).resolve()
        seeds = a.seeds or ([0,1,42] if a.suite == 'web' else [0,1,10] if ds == 'ETTm1' else [0,1,2])
        for h in a.horizons or (HORIZONS[ds] if a.suite == 'web' else [96,192,336,720]):
            for seed in seeds:
                for method in a.methods:
                    output = root/ds/f'H{h}'/f'seed{seed}'/method.replace('&','_')
                    command = [sys.executable, str(ROOT/'scripts/train_model.py'), '--framework', a.framework,
                        '--data-type', a.suite, '--dataset', ds, '--pred-len', str(h), '--seed', str(seed),
                        '--method', method, '--data-dir', str(data_dir),
                        '--output', str(output), '--cache-dir', str(root/'cache')]
                    print(shlex.join(command), flush=True)
                    if a.execute:
                        if output.exists():
                            from src.preprocessing.io import file_hash
                            receipt = output/'completion.json'
                            if not receipt.is_file():
                                raise RuntimeError(f'Incomplete output retained: {output}; choose a fresh output root')
                            protocol_path = output/'public_protocol.json'
                            if not protocol_path.is_file():
                                raise RuntimeError(f'Missing public protocol: {output}')
                            protocol = json.loads(protocol_path.read_text())
                            expected = dict(framework=a.framework, dataset=ds, horizon=h, seed=seed, method=method, learning_rate=.0001, crest_weights=[.5,.25,.25])
                            if any(protocol.get(k) != v for k,v in expected.items()):
                                raise RuntimeError(f'Output protocol mismatch: {output}')
                            record = json.loads(receipt.read_text())
                            if record['state'] != 'complete' or any(file_hash(output/n) != digest for n,digest in record['hashes'].items()):
                                raise RuntimeError(f'Output validation failed: {output}')
                            continue
                        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
