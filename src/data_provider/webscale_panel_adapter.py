"""SHARP adapter for canonical Parquet panels.

The reusable data structures live in ``src.common.webscale_ts`` and deliberately
know nothing about Parquet or forecasting models.  This module is the thin
application boundary that reads the canonical preprocessing format and builds
the three explicit time splits used by SHARP.
"""

from collections import OrderedDict
from dataclasses import dataclass
import json
import hashlib
import time
from pathlib import Path

import numpy as np

from src.utils.panel_crest import validate_panel_crest_options

from src.common.webscale_ts import (
    InMemorySeriesBank,
    PanelWindowDataset,
    SeriesSubset,
    WindowManifest,
)


DEFAULT_PANEL_GROUPED_ROOT = (
    str(Path(__file__).resolve().parents[2] / 'data/web/processed_grouped')
)


PANEL_SPLIT_PRESETS = {
    'quito_corpus_hour': {
        'expected_length': 15356,
        'train_end': 11844,
        'valid_end': 14804,
        'test_end': 15356,
        'protocol': 'Quito global cutoff + pre-test 80/20',
    },
    'quito_corpus_min': {
        'expected_length': 5904,
        'train_end': 2074,
        'valid_end': 2592,
        'test_end': 5904,
        'protocol': 'Quito global cutoff + pre-test 80/20',
    },
    'tailedts_202401': {
        'expected_length': 744,
        'train_end': 408,
        'valid_end': 576,
        'test_end': 744,
        'protocol': 'Jan 1-17 / Jan 18-24 / Jan 25-31',
    },
    'extended_wikipedia': {
        'expected_length': 2557,
        'train_end': 2262,
        'valid_end': 2321,
        'test_end': 2557,
        'protocol': 'Frozen representative 59-day validation / 236-day rolling test protocol',
    },
}


def get_panel_split_preset(panel_dataset):
    """Return a defensive copy of one explicitly selected split protocol."""
    try:
        return dict(PANEL_SPLIT_PRESETS[panel_dataset])
    except KeyError as exc:
        choices = ', '.join(sorted(PANEL_SPLIT_PRESETS))
        raise ValueError(
            f'unsupported panel_dataset {panel_dataset!r}; choose from {choices}'
        ) from exc


def resolve_panel_root(args):
    """Select an existing canonical group without changing dataset identity.

    A group explicitly overrides panel_root, including a launcher's default.
    Without a group, preserve the existing root exactly as supplied.
    """
    group = getattr(args, 'panel_group', None)
    if not group:
        return args.panel_root
    get_panel_split_preset(args.panel_dataset)
    if Path(group).name != group or group in {'.', '..'}:
        raise ValueError('--panel_group must be a single group directory name')
    root = (
        Path(getattr(args, 'panel_grouped_root', DEFAULT_PANEL_GROUPED_ROOT))
        / args.panel_dataset / group
    ).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'panel group directory not found: {root}')
    if not (root / 'series_index.parquet').is_file():
        raise FileNotFoundError(
            f'canonical panel index not found: {root / "series_index.parquet"}'
        )
    args.panel_root = str(root)
    return args.panel_root


def validate_panel_window_feasibility(panel_dataset, preset, seq_len, pred_len):
    """Validate X availability and target-block horizon before any data load."""
    seq_len = int(seq_len)
    pred_len = int(pred_len)
    if seq_len < 1 or pred_len < 1:
        raise ValueError('panel seq_len and pred_len must be positive')
    train_end = int(preset['train_end'])
    valid_steps = int(preset['valid_end'] - train_end)
    test_steps = int(preset['test_end'] - preset['valid_end'])
    if seq_len > train_end:
        raise ValueError(
            f'{panel_dataset}: seq_len ({seq_len}) exceeds available training '
            f'history ({train_end})'
        )
    if seq_len + pred_len > train_end:
        raise ValueError(
            f'{panel_dataset}: seq_len + pred_len exceeds the training block'
        )
    if pred_len > valid_steps:
        raise ValueError(
            f'{panel_dataset}: prediction horizon exceeds validation target '
            f'block ({pred_len} > {valid_steps})'
        )
    if pred_len > test_steps:
        raise ValueError(
            f'{panel_dataset}: prediction horizon exceeds test target block '
            f'({pred_len} > {test_steps})'
        )


def build_target_split_manifests(
    series_bank, panel_dataset, preset, seq_len, pred_len,
    candidate_stride=1, query_stride=1,
):
    """Build train-only candidates and target-aligned query manifests.

    Validation/test X may use the immediately preceding history, while every Y
    interval is wholly contained in its fixed absolute target block.
    """
    validate_panel_window_feasibility(
        panel_dataset, preset, seq_len, pred_len
    )
    expected = int(preset['expected_length'])
    mismatches = np.flatnonzero(series_bank.lengths != expected)
    if len(mismatches):
        position = int(mismatches[0])
        raise ValueError(
            f'panel_dataset={panel_dataset} expects series length {expected}; '
            f'series_id={int(series_bank.series_ids[position])} has actual '
            f'length {int(series_bank.lengths[position])}'
        )

    common = dict(
        series_bank=series_bank,
        seq_len=seq_len,
        pred_len=pred_len,
    )
    candidate_manifest = WindowManifest.from_bounds(
        **common,
        stride=candidate_stride,
        start_bound=0,
        end_bound=preset['train_end'],
    )
    query_manifests = {
        'train': WindowManifest.from_bounds(
            **common,
            stride=query_stride,
            start_bound=0,
            end_bound=preset['train_end'],
        ),
        'valid': WindowManifest.from_bounds(
            **common,
            stride=query_stride,
            start_bound=preset['train_end'] - seq_len,
            end_bound=preset['valid_end'],
        ),
        'test': WindowManifest.from_bounds(
            **common,
            stride=query_stride,
            start_bound=preset['valid_end'] - seq_len,
            end_bound=preset['test_end'],
        ),
    }
    return candidate_manifest, query_manifests


def build_fixed_split_manifests(
    series_bank, panel_dataset, seq_len, pred_len,
    candidate_stride=1, query_stride=1,
):
    """Resolve a canonical preset and build its panel manifests."""
    preset = get_panel_split_preset(panel_dataset)
    candidate_manifest, query_manifests = build_target_split_manifests(
        series_bank,
        panel_dataset,
        preset,
        seq_len,
        pred_len,
        candidate_stride=candidate_stride,
        query_stride=query_stride,
    )
    return candidate_manifest, query_manifests, preset


def process_rss_bytes():
    """Return current resident memory on Linux, or ``None`` if unavailable."""
    try:
        fields = Path('/proc/self/statm').read_text(encoding='ascii').split()
        page_size = int(__import__('os').sysconf('SC_PAGE_SIZE'))
        return int(fields[1]) * page_size
    except (OSError, IndexError, ValueError):
        return None


class CanonicalParquetPanelSource:
    """Random-access reader for the preprocessing canonical panel schema."""

    def __init__(self, dataset_dir, cache_row_groups=2):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                'panel data requires pyarrow to read canonical Parquet files'
            ) from exc

        # The system allocator returns evicted row-group buffers to the OS;
        # Arrow's default mimalloc pool can otherwise retain the pilot peak.
        pa.set_memory_pool(pa.system_memory_pool())
        self._pa = pa
        self._pq = pq
        self.dataset_dir = Path(dataset_dir)
        index_path = self.dataset_dir / 'series_index.parquet'
        if not index_path.is_file():
            raise FileNotFoundError(
                f'canonical panel index not found: {index_path}'
            )
        required = [
            'series_id', 'part_file', 'row_group', 'row_in_group',
            'length', 'n_channels',
        ]
        table = pq.read_table(index_path, columns=required).combine_chunks()
        ids = table.column('series_id').to_numpy(zero_copy_only=False)
        if not np.array_equal(ids, np.arange(len(ids), dtype=np.int64)):
            raise ValueError(
                'series_index must be sorted by contiguous series_id'
            )
        part_files = table.column('part_file').combine_chunks()
        if part_files.null_count:
            raise ValueError('series_index part_file must not contain nulls')
        encoded_part_files = part_files.dictionary_encode()
        # Keep one Python string per unique part and one compact integer code
        # per series, even before a small --max_series subset is selected.
        self._part_file_names = encoded_part_files.dictionary.to_pylist()
        self._part_file_codes = encoded_part_files.indices.to_numpy(
            zero_copy_only=False
        )
        self._row_groups = table.column('row_group').to_numpy(
            zero_copy_only=False
        )
        self._rows_in_group = table.column('row_in_group').to_numpy(
            zero_copy_only=False
        )
        self._lengths = table.column('length').to_numpy(
            zero_copy_only=False
        ).astype(np.int64, copy=False)
        self._channels = table.column('n_channels').to_numpy(
            zero_copy_only=False
        ).astype(np.int64, copy=False)
        self._cache_size = max(1, int(cache_row_groups))
        self._cache = OrderedDict()
        self._files = {}
        self.series_read_count = 0
        self.row_group_disk_read_count = 0

    @property
    def num_series(self):
        return int(len(self._lengths))

    def get_length(self, series_id):
        return int(self._lengths[int(series_id)])

    def get_n_channels(self, series_id):
        return int(self._channels[int(series_id)])

    @property
    def part_file_unique_count(self):
        return int(len(self._part_file_names))

    @property
    def part_file_code_bytes(self):
        return int(self._part_file_codes.nbytes)

    def _part_file_for_series(self, series_id):
        sid = int(series_id)
        if sid < 0 or sid >= self.num_series:
            raise IndexError(series_id)
        return self._part_file_names[int(self._part_file_codes[sid])]

    def _row_group(self, part_file, row_group):
        key = (part_file, int(row_group))
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        parquet_file = self._files.get(part_file)
        if parquet_file is None:
            parquet_file = self._pq.ParquetFile(self.dataset_dir / part_file)
            self._files[part_file] = parquet_file
        table = parquet_file.read_row_group(int(row_group), columns=['values'])
        self.row_group_disk_read_count += 1
        self._cache[key] = table
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return table

    def get_series(self, series_id):
        sid = int(series_id)
        part_file = self._part_file_for_series(sid)
        group = self._row_group(part_file, self._row_groups[sid])
        value_array = group.column('values')[
            int(self._rows_in_group[sid])
        ].values
        self.series_read_count += 1
        values = value_array.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape(
            self.get_length(sid), self.get_n_channels(sid)
        )

    def release_io_cache(self):
        """Drop Parquet row groups once the selected RAM bank is complete."""
        self._cache.clear()
        for parquet_file in self._files.values():
            close = getattr(parquet_file, 'close', None)
            if close is not None:
                close()
        self._files.clear()
        try:
            self._pa.default_memory_pool().release_unused()
        except (ImportError, AttributeError):
            pass


@dataclass
class PanelDataBundle:
    source: CanonicalParquetPanelSource
    subset: SeriesSubset
    series_bank: InMemorySeriesBank
    candidate_manifest: WindowManifest
    query_manifests: dict
    datasets: dict
    memory_report: dict
    split_preset: dict


def _load_or_create_subset(args, source):
    subset_path = getattr(args, 'series_subset_path', '')
    normalized_path = None
    if subset_path:
        normalized_path = Path(subset_path)
        if normalized_path.suffix != '.npz':
            normalized_path = normalized_path.with_suffix('.npz')
    if normalized_path is not None and normalized_path.is_file():
        subset = SeriesSubset.load(normalized_path)
        if subset.total_series != source.num_series:
            raise ValueError(
                'saved subset total_series does not match the panel source'
            )
        if subset.dataset_identifier != str(Path(args.panel_root).resolve()):
            raise ValueError('saved subset dataset identity does not match panel source')
        return subset
    subset = SeriesSubset.create(
        source.num_series,
        fraction=getattr(args, 'series_fraction', None),
        max_series=getattr(args, 'max_series', None),
        seed=getattr(args, 'series_subset_seed', 0),
        dataset_identifier=str(Path(args.panel_root).resolve()),
    )
    if normalized_path is not None:
        subset.save(normalized_path)
    return subset


def build_panel_bundle(args):
    """Load one subset once and construct compact candidate/query manifests."""
    backend = getattr(args, 'retrieval_backend', 'hnsw')
    if backend not in {'exact', 'hnsw'}:
        raise ValueError('panel retrieval_backend must be exact or hnsw')
    if backend != 'hnsw' and getattr(args, 'hnsw_recall_check', False):
        raise ValueError('HNSW recall requires retrieval_backend=hnsw')
    validate_panel_crest_options(args)
    panel_dataset = getattr(args, 'panel_dataset', None)
    preset = get_panel_split_preset(panel_dataset)
    validate_panel_window_feasibility(
        panel_dataset, preset, args.seq_len, args.pred_len
    )
    resolve_panel_root(args)

    production_started_at = time.perf_counter()
    rss_before = process_rss_bytes()
    source = CanonicalParquetPanelSource(args.panel_root)
    subset = _load_or_create_subset(args, source)
    if subset.selected_count == 0:
        raise ValueError('panel subset selected no series')
    for sid in subset.selected_ids:
        actual = source.get_length(sid)
        if actual != preset['expected_length']:
            raise ValueError(
                f"panel_dataset={panel_dataset} expects series length "
                f"{preset['expected_length']}; series_id={sid} has actual length {actual}"
            )
    series_bank = InMemorySeriesBank.from_source(source, subset)
    rss_after_load = process_rss_bytes()
    source.release_io_cache()
    dataset_load_seconds = time.perf_counter() - production_started_at
    setup_started_at = time.perf_counter()
    if series_bank.n_channels != int(args.enc_in):
        raise ValueError(
            f'panel has {series_bank.n_channels} channels but enc_in={args.enc_in}'
        )

    candidate_manifest, query_manifests, preset = build_fixed_split_manifests(
        series_bank,
        panel_dataset,
        args.seq_len,
        args.pred_len,
        candidate_stride=args.candidate_stride,
        query_stride=args.query_stride,
    )
    rss_after = process_rss_bytes()
    report = {
        'selected_series_hash': hashlib.sha256(subset.selected_ids.tobytes()).hexdigest(),
        'dataset_load_seconds': dataset_load_seconds,
        **subset.summary(),
        **series_bank.summary(),
        'panel_dataset': panel_dataset,
        'split_protocol': preset['protocol'],
        'expected_length': int(preset['expected_length']),
        'train_end': int(preset['train_end']),
        'valid_end': int(preset['valid_end']),
        'test_end': int(preset['test_end']),
        'train_steps': int(preset['train_end']),
        'valid_target_steps': int(
            preset['valid_end'] - preset['train_end']
        ),
        'test_target_steps': int(
            preset['test_end'] - preset['valid_end']
        ),
        'source_series_reads': source.series_read_count,
        'source_row_group_disk_reads': source.row_group_disk_read_count,
        'part_file_unique_count': source.part_file_unique_count,
        'part_file_code_bytes': source.part_file_code_bytes,
        'candidate_windows': len(candidate_manifest),
        'train_candidate_count': len(candidate_manifest),
        'train_query_windows': len(query_manifests['train']),
        'valid_query_windows': len(query_manifests['valid']),
        'test_query_windows': len(query_manifests['test']),
        'candidate_manifest_bytes': candidate_manifest.nbytes,
        'query_manifest_bytes': int(sum(
            manifest.nbytes for manifest in query_manifests.values()
        )),
        'manifest_bytes': int(
            candidate_manifest.nbytes
            + sum(manifest.nbytes for manifest in query_manifests.values())
        ),
        'candidate_stride': int(args.candidate_stride),
        'query_stride': int(args.query_stride),
        'rss_before_bank_bytes': rss_before,
        'rss_after_series_load_bytes': rss_after_load,
        'rss_after_bank_bytes': rss_after,
        'rss_bank_delta_bytes': (
            None if rss_before is None or rss_after is None
            else rss_after - rss_before
        ),
    }
    datasets = {
        split: PanelWindowDataset(
            series_bank,
            manifest,
            candidate_manifest=candidate_manifest,
            label_len=args.label_len,
            split=split,
            memory_report=report,
        )
        for split, manifest in query_manifests.items()
    }
    bundle = PanelDataBundle(
        source, subset, series_bank, candidate_manifest,
        query_manifests, datasets, report, preset,
    )
    report['panel_index_setup_seconds'] = time.perf_counter() - setup_started_at
    bundle.production_started_at = production_started_at
    if getattr(args, 'memory_report', False):
        print_panel_memory_report(report, stage='bank-and-manifests')
    return bundle


def print_panel_memory_report(report, stage='panel'):
    """Print stable JSON so pilot logs remain easy to parse."""
    print(f'[WebscaleMemory][{stage}] {json.dumps(report, sort_keys=True)}')
