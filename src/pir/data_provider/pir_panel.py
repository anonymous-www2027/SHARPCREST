"""Load frozen whole-series cohorts with native PIR normalization and window order."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def array_hash(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    ids = data.get('ordered_source_ids', data.get('ordered_ids'))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('manifest requires nonempty unique ordered_source_ids')
    return data, np.asarray(ids, dtype=np.int64)


def load_series(root, manifest, ids, channel):
    from src.reproduction.model_data import load_raw_panel
    series, audit = load_raw_panel(root, manifest)
    if channel != 0:
        raise ValueError('RQ1 panels require channel zero')
    return series, audit


class PanelBundle:
    def __init__(self, args, raw_series=None):
        self.args = args
        self.manifest, self.ids = read_manifest(args.panel_manifest)
        anchor, self.anchor_ids = read_manifest(args.panel_anchor_manifest)
        self.input_hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for name, path in [('manifest', args.panel_manifest), ('anchor_manifest', args.panel_anchor_manifest)]}
        if anchor['dataset'] != self.manifest['dataset'] or anchor['split'] != self.manifest['split']:
            raise ValueError('anchor and cohort must share dataset and split protocol')
        if not set(self.anchor_ids).issubset(self.ids):
            raise ValueError('cohort must contain every fixed-test anchor series')
        self.split = self.manifest['split']
        te, ve, end = [int(self.split[x]) for x in ('train_end', 'valid_end', 'test_end')]
        L, H = args.seq_len, args.pred_len
        if not (0 < L+H <= te < ve < end) or H > min(ve-te, end-ve):
            raise ValueError('L/H not feasible for the frozen train/validation/test blocks')
        if not 0 <= args.label_len <= L:
            raise ValueError('label_len must be in [0,L]')
        self.raw, self.source_indexes = ((raw_series, {}) if raw_series is not None else
                                         load_series(args.panel_root, self.manifest, self.ids, args.panel_channel))
        self.data, self.mean, self.scale = {}, {}, {}
        for sid in self.ids:
            x = np.asarray(self.raw[int(sid)], dtype=np.float64)
            if x.shape != (int(self.split['expected_length']), 1) or not np.isfinite(x).all():
                raise ValueError('panel series must have expected length, one channel and finite values')
            # Existing web-scale train-only normalization, before PIR's own
            # per-window normalization. Neither fit uses validation/test data.
            self.mean[int(sid)] = x[:te].mean(0)
            self.scale[int(sid)] = x[:te].std(0)+1e-8
            self.data[int(sid)] = ((x-self.mean[int(sid)])/self.scale[int(sid)]).astype(np.float32)
        self.datasets = {}
        for flag, (begin, stop) in {'train': (0, te), 'retrieval': (0, te),
                                     'val': (te-L, ve), 'test': (ve-L, end)}.items():
            stride = args.retrieval_stride if flag == 'retrieval' else 1
            self.datasets[flag] = PanelDataset(self, flag, begin, stop, stride)

    def audit(self):
        return {'dataset': self.manifest['dataset'], 'series_count': len(self.ids),
                'anchor_series_count': len(self.anchor_ids), 'ordered_source_ids': self.ids.tolist(),
                'anchor_ids': self.anchor_ids.tolist(), 'L': self.args.seq_len, 'H': self.args.pred_len,
                'input_manifest_L': self.manifest.get('L'), 'input_manifest_H': self.manifest.get('H'),
                'split': self.split, 'normalization': 'per-series train mean / (population std + 1e-8), then native PIR instance normalization',
                'time_features': 'zeros (canonical panel has no native timestamp feature schema)',
                'source_index_sha256': self.source_indexes,
                'manifest_sha256': self.input_hashes['manifest'],
                'anchor_manifest_sha256': self.input_hashes['anchor_manifest'],
                'selected_raw_sha256': {str(sid): array_hash(self.raw[int(sid)]) for sid in self.ids},
                'query_hashes': {flag: array_hash(ds.metadata) for flag, ds in self.datasets.items()},
                'query_counts': {flag: len(ds) for flag, ds in self.datasets.items()},
                'fixed_test_queries': int(self.datasets['test'].fixed_mask.sum())}


class PanelDataset(Dataset):
    def __init__(self, bundle, flag, begin, stop, stride):
        self.bundle, self.flag = bundle, flag
        self.seq_len, self.pred_len = bundle.args.seq_len, bundle.args.pred_len
        self.label_len = bundle.args.label_len
        self.stride = stride
        starts = np.arange(begin, stop-self.seq_len-self.pred_len+1, stride, dtype=np.int64)
        self.metadata = np.column_stack([np.repeat(bundle.ids, len(starts)), np.tile(starts, len(bundle.ids))])
        if not len(self.metadata):
            raise ValueError('empty panel split')
        self.fixed_mask = np.isin(self.metadata[:, 0], bundle.anchor_ids)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sid, start = self.metadata[index]
        data = self.bundle.data[int(sid)]
        origin = int(start)+self.seq_len
        x = data[start:origin].copy()
        y = data[origin-self.label_len:origin+self.pred_len].copy()
        return x, y, np.zeros((self.seq_len, 5), np.float32), np.zeros((self.label_len+self.pred_len, 5), np.float32), index

    def get_person_similarity(self):
        return np.ones((1, 1), np.float32)

    def inverse_predictions(self, values):
        sids = self.metadata[:, 0]
        mean = np.stack([self.bundle.mean[int(s)] for s in sids])[:, None]
        scale = np.stack([self.bundle.scale[int(s)] for s in sids])[:, None]
        return values.astype(np.float64)*scale+mean


def panel_data_provider(args, flag):
    bundle = getattr(args, '_pir_panel_bundle', None)
    if bundle is None:
        if args.enc_in != 1 or args.c_out != 1 or args.dec_in != 1:
            raise ValueError('canonical web-scale protocol requires enc_in=dec_in=c_out=1')
        if args.percent != 100:
            raise ValueError('panel cohorts use complete series; percent must be 100')
        bundle = PanelBundle(args)
        args._pir_panel_bundle = bundle
    ds = bundle.datasets[flag]
    return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=flag != 'test',
                         drop_last=False, num_workers=args.num_workers)


def query_metadata(dataset, index):
    index = index.detach().cpu().numpy() if torch.is_tensor(index) else np.asarray(index)
    if hasattr(dataset, 'metadata'):
        return dataset.metadata[index]
    return np.column_stack([np.zeros(len(index), np.int64),
                            index*dataset.stride+getattr(dataset, 'past_start_offset', 0)])


def candidate_metadata(dataset):
    metadata = query_metadata(dataset, np.arange(len(dataset)))
    return {'series_id': metadata[:, 0], 'start': metadata[:, 1]}
