"""Exact frozen identities, canonical channel projection, and raw panel manifests."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from src.common.webscale_ts import InMemorySeriesBank, SeriesSubset, WindowManifest, PanelWindowDataset
from src.data_provider.webscale_panel_adapter import CanonicalParquetPanelSource, PanelDataBundle, build_fixed_split_manifests
from src.utils.panel_forecast import PanelForecastScaler
from src.utils.web_stp import WebSTPRobustScaler
from src.utils.main_protocol import WEB


class ChannelView:
    def __init__(self, source, channel=0):
        self.source,self.channel=source,channel
    def get_length(self,sid):return self.source.get_length(sid)
    def get_n_channels(self,sid):return 1
    def get_series(self,sid):return self.source.get_series(sid)[:,self.channel:self.channel+1]


class FrozenSource:
    def __init__(self, root, name, cohort):
        self.sources={};self.mapping={}
        if name.startswith('quito_'):
            import pandas as pd
            for _,dataset,sid,canonical in cohort['identities']:
                if dataset not in self.sources:
                    source=CanonicalParquetPanelSource(Path(root)/dataset)
                    meta=json.loads((Path(root)/dataset/'metadata.json').read_text())
                    if meta['channel_names'][0]!='ind_1':raise ValueError('Quito channel0 must be ind_1')
                    self.sources[dataset]=ChannelView(source)
                self.mapping[sid]=(dataset,canonical)
            for dataset in self.sources:
                index=pd.read_parquet(Path(root)/dataset/'series_index.parquet').set_index('series_id')
                for sid,(ds,canonical) in self.mapping.items():
                    if ds==dataset and int(index.loc[canonical,'source_id'])!=sid:
                        raise ValueError('Quito canonical/source identity mismatch')
            self.num_series=max(self.mapping)+1  # logical source-ID domain, never used to sample
        else:
            self.sources[name]=CanonicalParquetPanelSource(Path(root)/name)
            self.mapping={sid:(name,sid) for sid in cohort['ordered_ids']}
            self.num_series=self.sources[name].num_series
    def get_length(self,sid):
        ds,cid=self.mapping[int(sid)];return self.sources[ds].get_length(cid)
    def get_n_channels(self,sid):
        ds,cid=self.mapping[int(sid)];return self.sources[ds].get_n_channels(cid)
    def get_series(self,sid):
        ds,cid=self.mapping[int(sid)];return self.sources[ds].get_series(cid)
    def release_io_cache(self):
        for source in self.sources.values():
            getattr(source,'source',source).release_io_cache()


class OrderedManifest(WindowManifest):
    """Existing compact manifest, retaining the frozen cohort's label order."""
    def encode(self,series_ids,start_indices):
        # WindowManifest.encode assumes sorted series. Resolve via a sorted view
        # while retaining original prefix offsets for externally frozen labels.
        order=np.argsort(self.series_ids)
        ids=np.asarray(series_ids,dtype=np.int64);starts=np.asarray(start_indices,dtype=np.int64)
        ix=np.searchsorted(self.series_ids[order],ids)
        if np.any(ix>=len(order)):raise KeyError('unknown series')
        pos=order[ix]
        if not np.array_equal(self.series_ids[pos],ids):raise KeyError('unknown series')
        delta=starts-self.first_start[pos];local=delta//self.stride[pos]
        if ((delta<0)|(delta%self.stride[pos]!=0)|(local>=self.window_counts[pos])).any():raise KeyError('invalid start')
        return self.prefix_offsets[pos]+local


def reorder(manifest,ids):
    ix=np.searchsorted(manifest.series_ids,ids)
    if not np.array_equal(manifest.series_ids[ix],ids):raise ValueError('frozen manifest identity mismatch')
    counts=manifest.window_counts[ix]
    return OrderedManifest(ids,manifest.first_start[ix],counts,np.r_[0,np.cumsum(counts)],
                           manifest.stride[ix],manifest.seq_len,manifest.pred_len)


def frozen_cohort(name):
    path=Path(__file__).resolve().parents[2]/'configs/legacy_frozen_cohorts.json'
    cohort=json.loads(path.read_text())[name]
    if name.startswith('quito_'):
        serialized=json.dumps(cohort['identities'],separators=(',',':')).encode()
        ids=np.array([r[2] for r in cohort['identities']],dtype=np.int64)
    else:
        ids=np.array(cohort['ordered_ids'],dtype=np.int64);serialized=ids.tobytes()
    if hashlib.sha256(serialized).hexdigest()!=WEB[name]['hash'] or cohort['cohort_hash']!=WEB[name]['hash']:
        raise ValueError('frozen cohort hash mismatch')
    if len(ids)!=WEB[name]['N'] or len(np.unique(ids))!=len(ids):raise ValueError('frozen cohort count mismatch')
    return cohort,ids


def build_frozen_bundle(args):
    started=time.perf_counter();name=args.main_dataset;spec=WEB[name]
    cohort,ids=frozen_cohort(name)
    source=FrozenSource(args.canonical_root,name,cohort)
    subset=SeriesSubset(source.num_series,np.sort(ids),0,'frozen',name)
    bank=InMemorySeriesBank.from_source(source,subset);source.release_io_cache()
    if bank.n_channels!=1 or not np.isfinite(bank.values).all():raise ValueError('finite univariate raw bank required')
    cm,qm,preset=build_fixed_split_manifests(bank,spec['panel'],args.seq_len,args.pred_len)
    cm=reorder(cm,ids);qm={s:reorder(m,ids) for s,m in qm.items()}
    scaler=PanelForecastScaler(bank,preset['train_end'],spec['m'])
    # STP coordinates are independent of the model z-score and raw search/value bank.
    bank.stp_robust_scaler=WebSTPRobustScaler(bank,preset['train_end'])
    report=dict(bank.summary(),cohort_hash=spec['hash'],selected_series_hash=spec['hash'],
        panel_dataset=spec['panel'],dataset_load_seconds=time.perf_counter()-started,
        train_end=preset['train_end'],valid_end=preset['valid_end'],test_end=preset['test_end'],
        candidate_stride=1,query_stride=1,model_space='train-only zscore',retrieval_space='raw',
        value_transfer='QUERY_SCALE_ORIGINAL',forecast_channels=1,
        stp_space='historical train-only robust WN-R')
    datasets={s:PanelWindowDataset(bank,q,candidate_manifest=cm,label_len=args.label_len,split=s,memory_report=report) for s,q in qm.items()}
    for dataset in datasets.values():dataset.forecast_scaler=scaler
    bundle=PanelDataBundle(source,subset,bank,cm,qm,datasets,report,preset)
    bundle.forecast_scaler=scaler;bundle.stp_robust_scaler=bank.stp_robust_scaler
    bundle.production_started_at=started
    return bundle
