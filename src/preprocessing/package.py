"""Compact paper-series bundles retaining original logical IDs and raw values."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .io import file_hash, write_json
from .protocol import ROOT


class PackagedSource:
    """Expose bundled series through the forecasting source interface."""
    def __init__(self, root, dataset, cohort):
        directory = Path(root) / dataset
        metadata = json.loads((directory / 'package.json').read_text())
        if metadata['format'] != 'sharp-paper-series-v2' or metadata['dataset'] != dataset:
            raise ValueError('Packaged dataset format/name mismatch')
        scale = str(cohort['nominal_scale'])
        if scale not in metadata['scale_packages']:
            raise ValueError(f'No packaged dataset for scale {scale}')
        package = metadata['scale_packages'][scale]
        path = directory / f'{scale}.parquet'
        if file_hash(path) != package['parquet_sha256']:
            raise ValueError('Packaged series checksum mismatch')
        table = pq.read_table(path)
        ids = table['series_id'].to_pylist()
        if len(ids) != len(set(ids)) or len(ids) != package['series_count']:
            raise ValueError('Invalid packaged series IDs')
        if ids != cohort['ordered_source_ids']:
            raise ValueError('Packaged series order differs from the requested cohort')
        self.values = {}; self.identities = {}
        for i, sid in enumerate(ids):
            x = table['values'][i].values.to_numpy()
            if x.dtype != np.float32 or len(x) != metadata['series_length'] or not np.isfinite(x).all():
                raise ValueError('Packaged data must contain complete finite float32 series')
            self.values[sid] = x.reshape(-1, 1)
            self.identities[sid] = table['identity'][i].as_py()
        for identity in cohort['selected_identities']:
            sid = int(identity[2])
            if self.identities.get(sid) != json.dumps(identity, separators=(',', ':')):
                raise ValueError('Packaged source identity differs from cohort')
        if not set(cohort['ordered_source_ids']) <= set(ids):
            raise ValueError('Requested cohort contains series outside the bundled population')
        self.num_series = max(ids) + 1

    def get_length(self, sid):
        return len(self.values[int(sid)])

    def get_n_channels(self, sid):
        return 1

    def get_series(self, sid):
        return self.values[int(sid)]

    def release_io_cache(self):
        pass


def export_paper(canonical_root, dataset, output):
    """Write one complete Parquet file for each published scale."""
    from src.data_provider.frozen_panel import FrozenSource
    if dataset not in ('extended_wikipedia', 'quito_hour'):
        raise ValueError('Paper data packaging currently supports Wiki and Quito-Hour')
    cohorts = {n: json.loads((ROOT/'configs/cohorts'/dataset/f'{n}.json').read_text())
               for n in (100000, 200000, 400000, 800000)}
    largest = cohorts[800000]
    selector = {'identities': largest['selected_identities']} if dataset == 'quito_hour' else {'ordered_ids': largest['ordered_source_ids']}
    source = FrozenSource(canonical_root, dataset, selector)
    try:
        series = {sid: source.get_series(sid).copy()[:, 0] for sid in largest['ordered_source_ids']}
    finally:
        source.release_io_cache()
    for scale, m in cohorts.items():
        ids = m['ordered_source_ids']
        if largest['ordered_source_ids'][:len(ids)] != ids:
            raise ValueError('Published series are not nested')
        bank = np.concatenate([series[sid] for sid in sorted(ids)])[:, None]
        h = hashlib.sha256(str((bank.shape, bank.dtype)).encode()); h.update(memoryview(bank))
        if h.hexdigest() != m['selected_raw_bank_hash']:
            raise ValueError(f'Raw bank differs from published {dataset}/{scale}')
    directory = Path(output)/dataset
    directory.mkdir(parents=True, exist_ok=False)
    packages = {}
    for scale, m in cohorts.items():
        ids = m['ordered_source_ids']
        table = pa.table(dict(series_id=pa.array(ids, pa.int64()),
            identity=[json.dumps(x, separators=(',', ':')) for x in m['selected_identities']],
            values=pa.array([series[sid] for sid in ids], pa.list_(pa.float32()))))
        path = directory/f'{scale}.parquet'; pq.write_table(table, path, compression='zstd')
        packages[str(scale)] = dict(file=path.name, series_count=len(ids),
            raw_bytes=sum(series[sid].nbytes for sid in ids), parquet_bytes=path.stat().st_size,
            parquet_sha256=file_hash(path), cohort_hash=m['cohort_hash'],
            selected_raw_bank_hash=m['selected_raw_bank_hash'],
            manifest_sha256=file_hash(ROOT/'configs/cohorts'/dataset/f'{scale}.json'))
    metadata = dict(format='sharp-paper-series-v2', dataset=dataset,
        series_length=largest['split']['expected_length'], channel='ind_1' if dataset=='quito_hour' else 'page_views',
        split=largest['split'], scales=list(cohorts), includes='Full train, validation and test series; no windows expanded',
        transformations='Existing canonical float32 values; Wiki zero-fill; Quito channel ind_1 projection',
        parquet_bytes=sum(item['parquet_bytes'] for item in packages.values()), scale_packages=packages)
    write_json(directory/'package.json', metadata)
    return metadata
