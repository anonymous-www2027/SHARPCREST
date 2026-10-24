"""Load published nested cohorts without historical result directories."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.reproduction.config import ROOT, WEB
from src.common.webscale_ts import InMemorySeriesBank, SeriesSubset, PanelWindowDataset
from src.data_provider.frozen_panel import FrozenSource, reorder
from src.data_provider.webscale_panel_adapter import build_target_split_manifests
from src.data_provider.panel_data_provider import PanelBatchDataset
from src.utils.panel_forecast import PanelForecastScaler
from src.utils.web_stp import WebSTPRobustScaler


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(); h.update(str((a.shape, a.dtype)).encode()); h.update(memoryview(a))
    return h.hexdigest()


def dataset_identity(dataset):
    if getattr(dataset, 'is_panel_dataset', False):
        h = hashlib.sha256()
        for start in range(0, len(dataset), 4096):
            ids = np.arange(start, min(start+4096, len(dataset)), dtype=np.int64)
            sid, pos = dataset.query_manifest.resolve_ids(ids)
            h.update(np.column_stack([ids, sid, pos]).tobytes())
        f = dataset.forecast_scaler
        return dict(count=len(dataset), query_identity_hash=h.hexdigest(),
            cohort_hash=dataset.memory_report['cohort_hash'],
            normalization_hash=hashlib.sha256((array_hash(f.mean)+array_hash(f.scale)).encode()).hexdigest())
    base = dataset.dataset; h = hashlib.sha256()
    for key in ('data_x', 'data_y', 'data_stamp'):
        h.update(key.encode()); h.update(array_hash(getattr(base,key)).encode())
    h.update(str((len(dataset),base.seq_len,base.label_len,base.pred_len)).encode())
    return dict(count=len(dataset), query_identity_hash=h.hexdigest(), cohort_hash=None,
        normalization_hash=hashlib.sha256((array_hash(base.scaler.mean_)+array_hash(base.scaler.scale_)).encode()).hexdigest())


def load_cohort(args):
    """Verify frozen series IDs; model seeds never resample the population."""
    path = Path(args.cohort_file)
    if not path.is_file():
        raise ValueError(f'No published whole-series cohort at {path}; choose an available scale or supply --cohort_file')
    raw = path.read_bytes(); m = json.loads(raw)
    if m.get('schema_version') != 1 or m.get('population_mode') != 'nested_whole_series':
        raise ValueError('Expected schema_version=1 nested_whole_series cohort')
    if m['dataset'] != args.dataset or m['nominal_scale'] != args.scale:
        raise ValueError('cohort dataset/scale differs from requested dataset/scale')
    ids=m['ordered_source_ids']; anchor=m['anchor_source_ids']
    if not ids or len(ids)!=len(set(ids)) or len(ids)!=m['selected_series_count']:
        raise ValueError('cohort IDs must be nonempty and unique')
    if not anchor or len(anchor)!=len(set(anchor)) or ids[:len(anchor)]!=anchor:
        raise ValueError('100K anchor must be the ordered prefix of the cohort')
    selected_hash=hashlib.sha256(np.array(ids,dtype='<i8').tobytes()).hexdigest()
    cohort_hash=hashlib.sha256(json.dumps(m['selected_identities'],separators=(',',':')).encode()).hexdigest() if args.dataset.startswith('quito_') else selected_hash
    if selected_hash!=m['selected_series_hash'] or cohort_hash!=m['cohort_hash']:
        raise ValueError('cohort identity hash mismatch')
    if args.dataset.startswith('quito_') and [x[2] for x in m['selected_identities']]!=ids:
        raise ValueError('Quito source identity order mismatch')
    manifest_sha=hashlib.sha256(raw).hexdigest()
    try: relative=str(path.resolve().relative_to(ROOT/'configs'))
    except ValueError: relative=None
    bundled=json.loads((ROOT/'configs/cohort_checksums.json').read_text())
    if relative in bundled and manifest_sha!=bundled[relative]:
        raise ValueError('Bundled cohort checksum mismatch; use a separate cohort_file for a new experiment')
    return m, manifest_sha


def fixed_arrays(bundle, anchor_ids):
    """Collect evaluation-only test targets from the fixed 100K anchor series."""
    dataset=bundle.datasets['test']; q=dataset.query_manifest
    sid,start=q.resolve_ids(np.arange(len(q),dtype=np.int64))
    ix=np.flatnonzero(np.isin(sid,np.asarray(anchor_ids)))
    raw=dataset.series_bank.get_windows(sid[ix],start[ix]+dataset.seq_len,dataset.pred_len)
    batched=PanelBatchDataset(dataset)
    normalized=np.concatenate([batched[ix[i:i+512]][2].numpy() for i in range(0,len(ix),512)])
    return dict(query_id=ix.astype(np.int64),source_id=sid[ix],query_origin=start[ix]+dataset.seq_len,
        raw_target=raw,normalized_target=normalized,target_mask=np.isfinite(raw))


def build_bundle(args, m):
    """Build whole-series windows and train-fitted scalers from a frozen cohort."""
    identity={'identities':m['selected_identities']} if args.dataset.startswith('quito_') else {'ordered_ids':m['ordered_source_ids']}
    if (Path(args.canonical_root)/args.dataset/'package.json').is_file():
        from src.preprocessing.package import PackagedSource
        source=PackagedSource(args.canonical_root,args.dataset,m)
    else:
        source=FrozenSource(args.canonical_root,args.dataset,identity)
    ids=np.asarray(m['ordered_source_ids'],dtype=np.int64)
    subset=SeriesSubset(source.num_series,np.sort(ids),m['selection_seed'],'published-nested-whole-series',args.dataset)
    try: bank=InMemorySeriesBank.from_source(source,subset)
    finally: source.release_io_cache()
    if bank.n_channels!=1 or not np.isfinite(bank.values).all():raise ValueError('Expected finite univariate raw data')
    if array_hash(bank.values)!=m['selected_raw_bank_hash']:raise ValueError('Selected raw data differs from frozen cohort')
    cm,qm=build_target_split_manifests(bank,WEB[args.dataset][0],m['split'],args.seq_len,args.pred_len)
    cm=reorder(cm,ids);qm={s:reorder(q,ids) for s,q in qm.items()}
    expected=m['split']['train_end']-args.seq_len-args.pred_len+1
    if not np.all(cm.window_counts==expected):raise ValueError('Partial series candidates are forbidden')
    scaler=PanelForecastScaler(bank,m['split']['train_end'],WEB[args.dataset][3])
    bank.stp_robust_scaler=WebSTPRobustScaler(bank,m['split']['train_end'])
    report=dict(cohort_hash=m['cohort_hash'],selected_series_hash=m['selected_series_hash'],
        panel_dataset=WEB[args.dataset][0],**{k:m['split'][k] for k in ('train_end','valid_end','test_end')})
    sets={s:PanelWindowDataset(bank,q,cm,label_len=0,split=s,memory_report=report) for s,q in qm.items()}
    for dataset in sets.values():dataset.forecast_scaler=scaler
    bundle=SimpleNamespace(series_bank=bank,candidate_manifest=cm,query_manifests=qm,datasets=sets,forecast_scaler=scaler,memory_report=report)
    identities={s:dataset_identity(d) for s,d in sets.items()}
    fixed=fixed_arrays(bundle,m['anchor_source_ids']); hashes={k:array_hash(v) for k,v in fixed.items()}
    reference=m.get('expected',{}).get(str(args.pred_len)) if args.seq_len==m['L'] else None
    if reference:
        if identities!=reference['dataset_identities'] or len(cm)!=reference['actual_candidate_count']:
            raise ValueError('Historical candidate/query/scaler identity mismatch')
        if 'fixed_hashes' in reference and hashes!=reference['fixed_hashes']:
            raise ValueError('Historical fixed-test identity/target mismatch')
    audit=dict(dataset_identities=identities,actual_candidate_count=len(cm),selected_series_count=len(ids),
        anchor_series_count=len(m['anchor_source_ids']),fixed_test_queries=len(fixed['query_id']),fixed_hashes=hashes,
        historical_identity_checked=bool(reference),historical_fixed_targets_checked=bool(reference and 'fixed_hashes' in reference),
        cohort_hash=m['cohort_hash'],whole_series=True,nominal_scale=args.scale,L=args.seq_len,H=args.pred_len)
    return bundle,fixed,audit
