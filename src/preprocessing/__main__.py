"""Command-line entry point for canonical conversion and TSF cohort preparation."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    convert = commands.add_parser('convert-quito-hour', help='Convert one Quito-Hour raw Parquet source')
    convert.add_argument('--source', type=Path, required=True)
    convert.add_argument('--dataset', choices=['quitobench_hour', 'quito_corpus_hour'], required=True)
    convert.add_argument('--output', type=Path, required=True, help='New canonical dataset directory')
    convert.add_argument('--source-sha256', help='Expected raw file SHA256; reject a different source')
    convert.add_argument('--source-version', help='Raw dataset release identifier to record, e.g. v20260315')
    convert.add_argument('--sample-series', type=int, help='Bounded conversion smoke test only')
    convert.add_argument('--max-partition-rows', type=int, default=2000000)
    convert.add_argument('--target-values-per-part', type=int, default=50000000)
    build = commands.add_parser('prepare', help='Build nested TSF-proportional cohorts from canonical panels')
    build.add_argument('--canonical-root', type=Path, required=True,
                       help='Root containing dataset directories with index, metadata and part Parquet files')
    build.add_argument('--dataset', required=True, help='Preset name or custom canonical dataset directory name')
    build.add_argument('--scales', type=int, nargs='+', required=True, help='Candidate targets, e.g. 100000 200000 400000 800000')
    build.add_argument('--output', type=Path, required=True, help='New output, or identical interrupted request to resume')
    build.add_argument('--protocol', type=Path, help='JSON overrides of split, period, L/H, anchor_series, selection_seed, etc.')
    build.add_argument('--L', '--seq-len', dest='L', type=int, help='Reference input length for candidate sizing')
    build.add_argument('--H', '--pred-len', dest='H', type=int, help='Reference prediction length for candidate sizing')
    build.add_argument('--period', type=int, help='Seasonal period in observations for TSF')
    for name in ('train-end', 'valid-end', 'test-end'):
        build.add_argument('--' + name, type=int, help='Exclusive split end, counted from the start of each series')
    build.add_argument('--channel', type=int, help='Zero-based channel for TSF and selected-value hashes (default: 0)')
    build.add_argument('--selection-seed', type=int, help='Cohort sampling seed (default: 0)')
    build.add_argument('--anchor-series', type=int, help='Optional 100K series count; custom datasets default to automatic sizing')
    build.add_argument('--workers', type=int, default=1, help='TSF worker processes; queues are bounded')
    build.add_argument('--batch-size', type=int, default=256)
    inputs = build.add_mutually_exclusive_group()
    inputs.add_argument('--tsf-root', type=Path, help='Optional canonical root preserving missing values for TSF')
    inputs.add_argument('--tsf-table', type=Path, help='Verified generated or published population TSF table')
    build.add_argument('--require-published', action='store_true', help='Fail unless generated IDs and raw hashes match bundled paper cohorts')
    pack = commands.add_parser('package-paper', help='Package separate 100K, 200K, 400K and 800K Parquet files')
    pack.add_argument('--canonical-root', type=Path, required=True)
    pack.add_argument('--dataset', choices=['extended_wikipedia', 'quito_hour'], required=True)
    pack.add_argument('--output', type=Path, required=True, help='Root for new compact dataset directories')
    a = parser.parse_args(argv)
    if a.command == 'convert-quito-hour':
        from .quito import convert_quito
        result = convert_quito(source=a.source, dataset_name=a.dataset, output_dir=a.output,
            sample_series=a.sample_series, max_partition_rows=a.max_partition_rows,
            target_values_per_part=a.target_values_per_part, expected_source_sha256=a.source_sha256,
            source_version=a.source_version)
        print(json.dumps(dict(status='PASS', output=str(a.output), num_series=result['num_series'])))
    elif a.command == 'package-paper':
        from .package import export_paper
        print(json.dumps(export_paper(a.canonical_root, a.dataset, a.output), indent=2))
    else:
        from .prepare import prepare
        overrides = json.loads(a.protocol.read_text()) if a.protocol else {}
        for name in ('L', 'H', 'period', 'channel', 'selection_seed', 'anchor_series'):
            if getattr(a, name) is not None:
                overrides[name] = getattr(a, name)
        for name in ('train_end', 'valid_end', 'test_end'):
            if getattr(a, name) is not None:
                overrides.setdefault('split', {})[name] = getattr(a, name)
        if a.test_end is not None:
            overrides['split']['expected_length'] = a.test_end
        result = prepare(a.canonical_root, a.dataset, a.scales, a.output,
            overrides=overrides,
            workers=a.workers, batch_size=a.batch_size, tsf_root=a.tsf_root,
            tsf_table=a.tsf_table, require_published=a.require_published)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
