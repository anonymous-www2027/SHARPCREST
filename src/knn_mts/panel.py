"""Frozen 200K panel data: no resampling, no windows across series boundaries."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import file_hash, array_hash

WEB_DATASETS=('quito_hour','extended_wikipedia','tailedts_202401')

class PanelWindows(Dataset):
    def __init__(self,values,positions,starts,length,horizon):
        self.values,self.positions,self.starts=values,positions,starts
        self.length,self.horizon=length,horizon
    def __len__(self):return len(self.starts)
    def __getitem__(self,index):
        x=self.values[int(self.positions[index])]; start=int(self.starts[index]); edge=start+self.length
        return x[start:edge],x[edge:edge+self.horizon],index

class PanelData:
    def __init__(self,path,dataset,length,horizon):
        self.path=Path(path).resolve()
        complete=json.loads((self.path/'completion.json').read_text())
        if complete['state']!='complete':raise ValueError('incomplete panel export')
        for name,digest in complete['hashes'].items():
            if file_hash(self.path/name)!=digest:raise ValueError('panel artifact hash mismatch: '+name)
        m=json.loads((self.path/'rq1_manifest.json').read_text());self.manifest=m
        if (m['dataset'],m['L'],m['H'],m['nominal_scale'])!=(dataset,length,horizon,200000):raise ValueError('panel dataset/L/H/cohort mismatch')
        self.dataset,self.length,self.horizon,self.channels=dataset,length,horizon,1
        self.columns=['value'];self.ends={k:m['split'][k+'_end'] for k in ['train','valid','test']}
        self.ends['val']=self.ends.pop('valid')
        self.series_ids=np.asarray(m['ordered_source_ids'],dtype=np.int64)
        anchor=np.asarray(m['anchor_source_ids'],dtype=np.int64)
        if not np.array_equal(self.series_ids[:len(anchor)],anchor):raise ValueError('missing anchor prefix')
        with np.load(self.path/'scaler.npz') as s:
            np.testing.assert_array_equal(s['series_id'],self.series_ids)
            self.mean,self.scale=s['mean'].copy(),s['scale'].copy()
        values=np.load(self.path/'values.npy'); raw=np.load(self.path/'raw_values.npy')
        if values.shape!=(len(self.series_ids),self.ends['test'],1) or not np.isfinite(values).all():raise ValueError('invalid panel values')
        train_raw=raw[:,:self.ends['train']].astype(np.float64)
        np.testing.assert_array_equal(self.mean,train_raw.mean(1))
        np.testing.assert_array_equal(self.scale,train_raw.std(1)+1e-8)
        np.testing.assert_array_equal(values,((raw.astype(np.float64)-self.mean[:,None,:])/self.scale[:,None,:]).astype(np.float32))
        self.values=torch.from_numpy(values)
        positions={int(s):i for i,s in enumerate(self.series_ids)}
        self.splits={};self.identities={}
        with np.load(self.path/'identities.npz') as a:
            for split in ['train','val','test']:
                sid=a[split+'_source_id'].copy();starts=a[split+'_past_start'].copy()
                ix=np.asarray([positions[int(s)] for s in sid],dtype=np.int64)
                begin=0 if split=='train' else self.ends['train' if split=='val' else 'val']-length
                end=self.ends[split]-horizon-length
                expected_sid=np.repeat(self.series_ids,end-begin+1); expected_starts=np.tile(np.arange(begin,end+1),len(self.series_ids))
                np.testing.assert_array_equal(sid,expected_sid);np.testing.assert_array_equal(starts,expected_starts)
                self.identities[split]=dict(query_id=np.arange(len(sid)),source_id=sid,past_start=starts,forecast_start=starts+length)
                self.splits[split]=PanelWindows(self.values,ix,starts,length,horizon)
            cs,ct=a['candidate_source_id'],a['candidate_past_start']
            np.testing.assert_array_equal(cs,self.identities['train']['source_id']);np.testing.assert_array_equal(ct,self.identities['train']['past_start'])
            self.candidate_starts=torch.from_numpy(self.splits['train'].positions*self.ends['train']+ct)
            self.candidate_identity=dict(source_id=cs.copy(),past_start=ct.copy())
        self.bank_values=self.values[:,:self.ends['train']].contiguous().view(-1,1)
        self.fixed_mask=np.isin(self.identities['test']['source_id'],anchor)
        with np.load(self.path/'fixed_test.npz') as f:
            keep=np.flatnonzero(self.fixed_mask)
            np.testing.assert_array_equal(keep,f['query_id'])
            np.testing.assert_array_equal(self.identities['test']['source_id'][keep],f['source_id'])
            np.testing.assert_array_equal(self.identities['test']['forecast_start'][keep],f['query_origin'])
            for begin in range(0,len(keep),1024):
                q=keep[begin:begin+1024];s=self.splits['test'];ix=s.positions[q];t=s.starts[q,None]+length+np.arange(horizon)
                np.testing.assert_array_equal(values[ix[:,None],t],f['normalized_target'][begin:begin+len(q)])
                np.testing.assert_array_equal(raw[ix[:,None],t],f['raw_target'][begin:begin+len(q)])
        self.audit=dict(dataset=dataset,length=length,horizon=horizon,channels=1,columns=self.columns,
            panel_artifact_sha256=file_hash(self.path/'completion.json'),nominal_scale=200000,
            cohort_hash=m['cohort_hash'],series_count=len(self.series_ids),anchor_series_count=len(anchor),
            fixed_test_queries=int(self.fixed_mask.sum()),bank_candidates=len(self.candidate_starts),
            split_windows={k:len(v) for k,v in self.splits.items()},boundaries=self.ends,
            normalization='RAFT train-only per-series mean/std+1e-8',scaler_hash=array_hash(np.stack([self.mean,self.scale])),
            query_start_hashes={k:array_hash(np.stack([v['source_id'],v['past_start']],1)) for k,v in self.identities.items()},
            bank_id_rule='candidate row index -> saved (source_id, local past_start); train-only complete series')
    def evaluation_scale(self,split,begin,end):
        return self.scale[self.splits[split].positions[begin:end],None,:]
