"""Resolve bundled datasets without importing historical experiment directories."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEB_LENGTHS = {'quito_hour': 576, 'extended_wikipedia': 576, 'tailedts_202401': 168}
HORIZONS = {'quito_hour': [48, 288, 512], 'extended_wikipedia': [7, 28, 59],
            'tailedts_202401': [24, 48, 96]}
GENERAL = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange')


def general_csv(root, dataset):
    """Accept the bundled General root or a directory containing the CSV."""
    root = Path(root).resolve()
    name = dataset + ('.csv' if dataset.startswith('ETT') else '_wide.csv')
    folder = 'ETT' if dataset.startswith('ETT') else 'experiment'
    for path in (root / name, root / folder / name):
        if path.is_file():
            return path
    raise FileNotFoundError(f'No {name} below {root}')


def cohort(dataset):
    """Validate the published nested 200K cohort and its 100K anchor."""
    from types import SimpleNamespace
    from src.reproduction.data import load_cohort
    path = ROOT / 'configs/cohorts' / dataset / '200000.json'
    manifest, _ = load_cohort(SimpleNamespace(cohort_file=path, dataset=dataset, scale=200000))
    return manifest


def load_raw_panel(root, manifest):
    """Load complete finite series and verify the published raw-bank hash."""
    from src.preprocessing.package import PackagedSource
    from src.data_provider.frozen_panel import FrozenSource
    dataset = manifest['dataset']
    root = Path(root).resolve()
    identity = ({'identities': manifest['selected_identities']} if dataset.startswith('quito_')
                else {'ordered_ids': manifest['ordered_source_ids']})
    source = (PackagedSource(root, dataset, manifest) if (root/dataset/'package.json').is_file()
              else FrozenSource(root, dataset, identity))
    try:
        raw = {int(s): source.get_series(int(s)).copy() for s in manifest['ordered_source_ids']}
    finally:
        source.release_io_cache()
    if any(x.dtype != np.float32 or x.shape != (manifest['split']['expected_length'], 1)
           or not np.isfinite(x).all() for x in raw.values()):
        raise ValueError('Expected complete finite float32 univariate series')
    bank = np.concatenate([raw[s] for s in sorted(raw)])
    h = hashlib.sha256(str((bank.shape, bank.dtype)).encode()); h.update(memoryview(bank))
    if h.hexdigest() != manifest['selected_raw_bank_hash']:
        raise ValueError('Selected raw series differ from the published cohort')
    return raw, {'selected_raw_bank_hash': h.hexdigest()}


def prepare_knn_panel(root, dataset, horizon, directory):
    """Build the portable kNN panel once, preserving series-major window IDs."""
    from src.preprocessing.io import exclusive_output, file_hash, write_json
    from src.knn_mts.panel import PanelData
    m = cohort(dataset); length = WEB_LENGTHS[dataset]
    directory = Path(directory).resolve()
    with exclusive_output(directory):
        if (directory / 'completion.json').is_file():
            # Verify the requested source as well as every cached artifact.
            load_raw_panel(root, m)
            PanelData(directory, dataset, length, horizon)
            saved = json.loads((directory/'rq1_manifest.json').read_text())
            if saved['selected_raw_bank_hash'] != m['selected_raw_bank_hash']:
                raise ValueError('Cached panel source differs from the public cohort')
            return directory
        if directory.exists():
            raise ValueError(f'Incomplete panel cache: {directory}; choose a fresh cache directory')
        raw_map, _ = load_raw_panel(root, m)
        ids = np.asarray(m['ordered_source_ids'], dtype=np.int64)
        raw = np.stack([raw_map[int(s)] for s in ids])
        te, ve, end = [m['split'][x] for x in ('train_end', 'valid_end', 'test_end')]
        if length+horizon > te or horizon > min(ve-te, end-ve):
            raise ValueError('L/H does not fit the published splits')
        train = raw[:, :te].astype(np.float64)
        mean, scale = train.mean(1), train.std(1)+1e-8
        values = ((raw.astype(np.float64)-mean[:, None])/scale[:, None]).astype(np.float32)
        directory.mkdir(parents=True)
        np.save(directory/'raw_values.npy', raw); np.save(directory/'values.npy', values)
        np.savez(directory/'scaler.npz', series_id=ids, mean=mean, scale=scale)
        payload = {}
        for split, begin, stop in [('train', 0, te), ('val', te-length, ve), ('test', ve-length, end)]:
            starts = np.arange(begin, stop-horizon-length+1, dtype=np.int64)
            payload[split+'_source_id'] = np.repeat(ids, len(starts))
            payload[split+'_past_start'] = np.tile(starts, len(ids))
        np.savez(directory/'identities.npz', candidate_source_id=payload['train_source_id'],
                 candidate_past_start=payload['train_past_start'], **payload)
        sid = payload['test_source_id']; start = payload['test_past_start']
        keep = np.flatnonzero(np.isin(sid, m['anchor_source_ids']))
        pos = {int(s): i for i, s in enumerate(ids)}
        ix = np.asarray([pos[int(s)] for s in sid[keep]])
        t = start[keep, None]+length+np.arange(horizon)
        np.savez(directory/'fixed_test.npz', query_id=keep, source_id=sid[keep],
                 query_origin=start[keep]+length, raw_target=raw[ix[:, None], t],
                 normalized_target=values[ix[:, None], t])
        write_json(directory/'rq1_manifest.json', dict(m, L=length, H=horizon))
        write_json(directory/'completion.json', dict(state='complete', hashes={
            p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}))
        PanelData(directory, dataset, length, horizon)
    return directory
