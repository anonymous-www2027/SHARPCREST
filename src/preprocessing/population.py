"""Stream canonical panels into a resumable, train-only TSF table."""
import itertools
import json
import platform
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .io import file_hash, json_hash, write_json
from .protocol import regimes
from .schema import CANONICAL_SCHEMA


def check_schema(schema, expected, name):
    """Require common fields and types while permitting extra columns and nullable schemas."""
    for field in expected:
        if field.name not in schema.names or schema.field(field.name).type != field.type:
            raise ValueError(f'{name}: require {field.name} with type {field.type}')


class Population:
    """Validate canonical indexes and retain their original series identities."""
    def __init__(self, root, p):
        self.root, self.protocol = Path(root), p
        self.indexes = {}; frames = []; self.files = {}
        for ds in p['sources']:
            directory = self.root / ds
            index_path = directory / 'series_index.parquet'
            index = pd.read_parquet(index_path)
            required = {'series_id', 'source_id', 'part_file', 'row_group', 'row_in_group',
                        'length', 'n_channels', 'start_timestamp'}
            if not required <= set(index):
                raise ValueError(f'{ds}: missing index columns {required-set(index)}')
            if index[list(required)].isna().any().any():
                raise ValueError(f'{ds}: index fields cannot be null')
            for name in ('series_id', 'row_group', 'row_in_group', 'length', 'n_channels'):
                if not pd.api.types.is_integer_dtype(index[name]) or (index[name] < 0).any():
                    raise ValueError(f'{ds}: index {name} must contain nonnegative integers')
            if not len(index) or not np.array_equal(index.series_id, np.arange(len(index))):
                raise ValueError(f'{ds}: series IDs must be contiguous, sorted, and nonempty')
            if not (index.length == p['split']['expected_length']).all():
                raise ValueError(f'{ds}: series lengths must match the protocol expected_length')
            if not (index.n_channels > p['channel']).all():
                raise ValueError(f'{ds}: missing selected channel')
            meta = json.loads((directory / 'metadata.json').read_text())
            names = meta.get('channel_names')
            if (not isinstance(names, list) or not names or
                    not all(isinstance(n, str) and n for n in names) or
                    len(set(names)) != len(names) or not (index.n_channels == len(names)).all()):
                raise ValueError(f'{ds}: metadata channel_names must match the channel count')
            if p['dataset'] == 'quito_hour' and meta.get('channel_names', [None])[0] != 'ind_1':
                raise ValueError('Quito canonical channel 0 must be ind_1')
            self.indexes[ds] = index
            part_names = sorted(set(index.part_file))
            if any(Path(n).name != n or not n.endswith('.parquet') for n in part_names):
                raise ValueError('part_file must be a local Parquet filename')
            if index.duplicated(['part_file', 'row_group', 'row_in_group']).any():
                raise ValueError(f'{ds}: duplicate index row pointers')
            for name in part_names:
                part = pq.ParquetFile(directory / name)
                check_schema(part.schema_arrow, CANONICAL_SCHEMA, f'{ds}/{name}')
                entries = index.loc[index.part_file == name]
                if len(entries) != part.metadata.num_rows:
                    raise ValueError(f'{ds}/{name}: index must cover every Parquet row')
                if (entries.row_group >= part.num_row_groups).any():
                    raise ValueError(f'{ds}/{name}: index row_group is outside the Parquet file')
            for name in ['series_index.parquet', 'metadata.json'] + part_names:
                self.files[f'{ds}/{name}'] = file_hash(directory / name)
            logical = index.source_id.astype(np.int64) if p['dataset'] == 'quito_hour' else index.series_id
            frames.append(pd.DataFrame(dict(logical_id=logical, source_dataset=ds,
                canonical_series_id=index.series_id, source_id=index.source_id.astype(str))))
        self.identities = pd.concat(frames, ignore_index=True)
        if self.identities.logical_id.duplicated().any() or (self.identities.logical_id < 0).any():
            raise ValueError('Logical source IDs must be unique and nonnegative across the population')

    def rows(self):
        """Yield validated rows in source/index order without retaining full panels."""
        for ds, index in self.indexes.items():
            cached_key = None
            for row in index.itertuples():
                key = (row.part_file, int(row.row_group))
                if key != cached_key:
                    table = pq.ParquetFile(self.root / ds / key[0]).read_row_group(key[1])
                    if any(table[name].null_count for name in CANONICAL_SCHEMA.names):
                        raise ValueError('Canonical row fields cannot be null')
                    cached_key = key
                i = int(row.row_in_group)
                if not 0 <= i < len(table):
                    raise ValueError('Index row_in_group is outside its Parquet row group')
                actual_id = table['series_id'][i].as_py()
                if actual_id != row.series_id or str(table['source_id'][i].as_py()) != str(row.source_id):
                    raise ValueError('Canonical row/index identity mismatch')
                if (table['length'][i].as_py() != row.length or
                        table['n_channels'][i].as_py() != row.n_channels or
                        pd.Timestamp(table['start_timestamp'][i].as_py()) != pd.Timestamp(row.start_timestamp)):
                    raise ValueError('Canonical row/index shape or timestamp mismatch')
                scalar = table['values'][i]
                if scalar.values.null_count:
                    raise ValueError('Use explicit float NaN rather than null elements in canonical values')
                if table['frequency_seconds'][i].as_py() <= 0:
                    raise ValueError('frequency_seconds must be positive')
                x = scalar.values.to_numpy(zero_copy_only=False)
                if x.dtype != np.float32 or len(x) != row.length * row.n_channels:
                    raise ValueError('Canonical values must be float32, flattened in time-major order')
                logical = int(row.source_id) if self.protocol['dataset'] == 'quito_hour' else int(row.series_id)
                yield logical, x.reshape(row.length, row.n_channels)[:, self.protocol['channel']]

    def validate(self):
        """Check all row pointers and selected-channel values before any TSF calculation."""
        if getattr(self, '_validated', False):
            return
        for sid, values in self.rows():
            if np.isinf(values).any():
                raise ValueError(f'Infinite value in series {sid}')
        self._validated = True
        print(f'CANONICAL_VALIDATED {self.protocol["dataset"]} series={len(self.identities)}', flush=True)

    def train_rows(self, tsf_population=None):
        end = self.protocol['split']['train_end']
        alternate = iter(tsf_population.rows()) if tsf_population else None
        for sid, full in self.rows():
            x = full[:end]
            if alternate is not None:
                other_sid, other = next(alternate)
                other = other[:end]
                if sid != other_sid or not np.array_equal(np.nan_to_num(other, nan=0.0), x):
                    raise ValueError('Missing-preserving TSF input does not map to canonical zero-filled train data')
                x = other
            if np.isinf(x).any():
                raise ValueError(f'Infinite train value in series {sid}')
            yield sid, np.asarray(x, dtype=np.float64)

    def bank_hash(self, ids):
        """Match the forecasting runner's sorted, float32 univariate bank hash."""
        import hashlib
        # Hash in logical-ID order; source order is not necessarily logical order.
        wanted = set(ids); selected = {}
        for ds, index in self.indexes.items():
            logical = index.source_id.astype(np.int64) if self.protocol['dataset'] == 'quito_hour' else index.series_id
            subset = index.loc[logical.isin(wanted)]
            for (part, group), entries in subset.groupby(['part_file', 'row_group'], sort=False):
                table = pq.ParquetFile(self.root / ds / part).read_row_group(int(group))
                for row in entries.itertuples():
                    sid = int(row.source_id) if self.protocol['dataset'] == 'quito_hour' else int(row.series_id)
                    i = int(row.row_in_group)
                    if table['series_id'][i].as_py() != row.series_id or str(table['source_id'][i].as_py()) != str(row.source_id):
                        raise ValueError('Selected raw row/index identity mismatch')
                    x = np.asarray(table['values'][i].values.to_numpy(), dtype=np.float32)
                    x = np.ascontiguousarray(x.reshape(row.length, row.n_channels)[:, self.protocol['channel']])
                    if not np.isfinite(x).all():
                        raise ValueError('Forecasting requires finite selected raw series; impute explicitly upstream')
                    selected[sid] = x
        if set(selected) != wanted:
            raise ValueError('Selected series are absent from canonical inputs')
        shape = (sum(len(x) for x in selected.values()), 1)
        h = hashlib.sha256(str((shape, np.dtype('float32'))).encode())
        for sid in sorted(selected):
            h.update(memoryview(selected[sid]))
        return h.hexdigest()


def environment():
    import scipy
    import statsmodels
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                statsmodels=statsmodels.__version__, pandas=pd.__version__, pyarrow=pa.__version__)


def calculate_batch(job):
    """Use the pinned Quito formulas; held-out values never enter this function."""
    from threadpoolctl import threadpool_limits
    from ._quito_tsf import _evaluate_series
    rows, p = job; out = []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for sid, x in rows:
            q = _evaluate_series(x, period=p['period'], compute_adf=False, compute_hurst=False)
            values = [q.trend_strength, q.season_strength, q.forecastability]
            if np.isfinite(values).all():
                high = [v > p['threshold'] if p['high_operator'] == '>' else v >= p['threshold'] for v in values]
                label = ('' if p['short_labels'] else '_').join(
                    ('H' if h else 'L') if p['short_labels'] else ('HIGH' if h else 'LOW') for h in high)
            else:
                label = 'UNDEFINED'
            out.append(dict(logical_id=int(sid), T=float(values[0]), S=float(values[1]), F=float(values[2]),
                            regime=label, finite_count=int(np.isfinite(x).sum()),
                            missing_fraction=float(np.isnan(x).mean())))
    return pd.DataFrame(out)


def validate_table(frame, population):
    required = {'logical_id', 'T', 'S', 'F', 'regime'}
    if not required <= set(frame) or frame.logical_id.duplicated().any():
        raise ValueError('TSF table requires unique logical_id, T, S, F and regime columns')
    if set(frame.logical_id) != set(population.identities.logical_id):
        raise ValueError('TSF table does not cover the complete canonical population')
    p = population.protocol
    values = frame[['T', 'S', 'F']].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    if np.isinf(values).any() or ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)).any():
        raise ValueError('TSF values must be in [0,1] or undefined (NaN)')
    bits = values > p['threshold'] if p['high_operator'] == '>' else values >= p['threshold']
    codes = bits @ np.array([4, 2, 1]); codes[~finite] = 8
    expected = np.asarray(regimes(p))[codes]
    if not np.array_equal(expected, frame.regime):
        raise ValueError('Stored regime labels do not match the configured threshold')
    return frame


def analyze(population, output, workers=1, batch_size=64, tsf_population=None):
    """Resume completed TSF batches only when inputs and the protocol match."""
    population.validate()
    if tsf_population is not None:
        tsf_population.validate()
    p = population.protocol; output = Path(output); output.mkdir(parents=True, exist_ok=True)
    key = dict(protocol=p, inputs=population.files, tsf_inputs=tsf_population.files if tsf_population else None,
               environment=environment(), batch_size=batch_size,
               calculator_sha256=file_hash(Path(__file__).with_name('_quito_tsf.py')),
               pipeline_sha256=file_hash(Path(__file__)))
    manifest = output / 'population.json'
    if manifest.exists() and json.loads(manifest.read_text())['key'] != key:
        raise ValueError('TSF cache protocol, input files or environment changed; choose a new output')
    if not manifest.exists():
        write_json(manifest, dict(key=key, fingerprint=json_hash(key), completed=False))
    saved = json.loads(manifest.read_text())
    final = output / 'population_tsf.parquet'
    if saved.get('completed'):
        if file_hash(final) != saved['table_sha256']:
            raise ValueError('Completed TSF table checksum mismatch')
        return validate_table(pd.read_parquet(final), population)
    chunks = output / 'chunks'; chunks.mkdir(exist_ok=True)
    rows = iter(population.train_rows(tsf_population))
    paths = []; pending = []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    def save(item):
        path, result = item
        frame = result.result() if executor else result
        frame.to_parquet(path.with_suffix('.tmp'), index=False)
        path.with_suffix('.tmp').replace(path)
        write_json(path.with_suffix('.json'), dict(sha256=file_hash(path)))
        print(f'TSF_BATCH_COMPLETE {path.stem} rows={len(frame)}', flush=True)
    try:
        for i in itertools.count():
            batch = list(itertools.islice(rows, batch_size))
            if not batch:
                break
            path = chunks / f'{i:07d}.parquet'; paths.append(path)
            seal = path.with_suffix('.json')
            if path.exists() and seal.exists():
                if file_hash(path) != json.loads(seal.read_text())['sha256']:
                    raise ValueError('TSF chunk checksum mismatch')
                continue
            pending.append((path, executor.submit(calculate_batch, (batch, p)) if executor else calculate_batch((batch, p))))
            if len(pending) >= max(1, workers * 2):
                save(pending.pop(0))
        for item in pending:
            save(item)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    frame = validate_table(pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True), population)
    frame.to_parquet(final.with_suffix('.tmp'), index=False)
    final.with_suffix('.tmp').replace(final)
    write_json(manifest, dict(key=key, fingerprint=json_hash(key), completed=True,
                             rows=len(frame), table_sha256=file_hash(final)))
    return frame
