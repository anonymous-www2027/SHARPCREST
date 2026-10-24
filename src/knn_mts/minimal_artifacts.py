"""Four-array output; streaming metrics and byte-identical, immutable array sharing."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .panel import PanelData
from .data import file_hash
from .retrieval import synchronize
from .model import forecast

ARRAY_NAMES = ('backbone_pred', 'knn_pred', 'target', 'neighbor_ids')


def deduplicate(out, index_path):
    from src.knn_mts.runner import write_json
    index_path = Path(index_path)
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    records = {}
    for p in sorted(Path(out).glob('*/*.npy')):
        if p.stem not in ARRAY_NAMES:
            raise ValueError('unexpected minimal artifact: '+str(p))
        digest = file_hash(p)
        old = Path(index[digest]) if digest in index else p
        if old != p:
            if not old.is_file() or file_hash(old) != digest:
                raise ValueError('shared array changed: '+str(old))
            tmp = p.with_suffix('.link.tmp')
            os.link(old, tmp)
            os.replace(tmp, p)
        else:
            index[digest] = str(p.resolve())
        records[str(p.relative_to(out))] = dict(sha256=digest, canonical=str(old.resolve()))
    write_json(index_path, index)
    write_json(Path(out)/'array_sharing.json', records)


class Stats:
    def __init__(self):
        self.total=0.;self.count=0;self.low=float('inf');self.high=-float('inf');self.below=0
    def add(self, x):
        self.total+=x.sum(dtype=np.float64);self.count+=x.size
        self.low=min(self.low,float(x.min()));self.high=max(self.high,float(x.max()))
        self.below+=int((x<1.01).sum())
    def result(self, effective=False):
        r=dict(mean=float(self.total/self.count),min=self.low,max=self.high)
        if effective:r['fraction_below_1_01']=self.below/self.count
        return r


@torch.no_grad()
def evaluate_minimal(model, bank, data, split, args, out):
    from src.knn_mts.runner import write_json, score_arrays, summarize
    directory=out/split;directory.mkdir();dataset=data.splits[split]
    shape=(len(dataset),args.pred_len,data.channels)
    arrays={n:np.lib.format.open_memmap(directory/(n+'.npy'),mode='w+',dtype='float32',shape=shape)
            for n in ARRAY_NAMES[:3]}
    ids=np.lib.format.open_memmap(directory/'neighbor_ids.npy',mode='w+',dtype='int64',
                                 shape=(len(dataset),data.channels,args.k))
    totals={n:np.zeros(5) for n in ('backbone','knn','combined')}
    fixed={n:np.zeros(5) for n in totals}
    stats={n:Stats() for n in ('mixing_lambda','effective_candidates','cr_q','lambda_q','sigma_q','A_q','V_q','N_q_size')}
    model.eval();begin=0;model_seconds=0.;old=dict(bank.timings);started=time.perf_counter()
    for past,target,past_start in DataLoader(dataset,batch_size=args.retrieval_batch_size,shuffle=False,num_workers=args.num_workers):
        synchronize(args.device);tick=time.perf_counter()
        backbone=torch.cat([forecast(model, part.to(args.device)) for part in past.split(args.eval_batch_size)], 0).cpu().numpy()
        synchronize(args.device);model_seconds+=time.perf_counter()-tick
        queries=past.permute(0,2,1).contiguous().view(-1,args.seq_len)
        knn=np.empty((len(queries),args.pred_len),dtype=np.float32)
        lam=np.empty(len(queries),dtype=np.float32);found=np.empty((len(queries),args.k),dtype=np.int64)
        qids=(past_start[:,None]*data.channels+torch.arange(data.channels)[None,:]).reshape(-1)
        for first in range(0,len(queries),args.retrieval_batch_size):
            last=min(first+args.retrieval_batch_size,len(queries))
            value,mixing,labels,dist=bank.predict(queries[first:last],qids[first:last],split)
            if not torch.isfinite(dist).all():raise RuntimeError('nonfinite distance')
            knn[first:last]=value.cpu().numpy();lam[first:last]=mixing.cpu().numpy();found[first:last]=labels.cpu().numpy()
            stats['effective_candidates'].add(bank.last['effective_candidates'].cpu().numpy())
            if bank.crest:
                ctl=bank.last['controller'].cpu().numpy()
                for j,n in enumerate(('cr_q','lambda_q','sigma_q')):stats[n].add(ctl[:,j])
                for n in ('A_q','V_q','N_q_size'):stats[n].add(bank.last['reliability_components'][n].cpu().numpy())
        B=len(past);end=begin+B
        knn=knn.reshape(B,data.channels,args.pred_len).transpose(0,2,1);lam=lam.reshape(B,data.channels)
        combined=(1-lam[:,None,:])*backbone+lam[:,None,:]*knn;truth=target.numpy()
        if not all(np.isfinite(x).all() for x in (backbone,knn,combined,lam)):raise RuntimeError('nonfinite output')
        arrays['backbone_pred'][begin:end]=backbone;arrays['knn_pred'][begin:end]=knn;arrays['target'][begin:end]=truth
        ids[begin:end]=found.reshape(B,data.channels,args.k);stats['mixing_lambda'].add(lam)
        scale=data.evaluation_scale(split,begin,end) if isinstance(data,PanelData) else data.scale
        for name,pred in [('backbone',backbone),('knn',knn),('combined',combined)]:
            totals[name]+=score_arrays(pred,truth,data.mean,scale)
            if isinstance(data,PanelData) and split=='test':
                keep=data.fixed_mask[begin:end]
                if keep.any():fixed[name]+=score_arrays(pred[keep],truth[keep],data.mean,scale[keep])
        begin=end
    for arr in [*arrays.values(),ids]:arr.flush()
    if isinstance(data,PanelData):np.savez(directory/'query_identity.npz',**data.identities[split])
    else:np.savez(directory/'query_identity.npz',past_start=dataset.starts,forecast_start=dataset.starts+args.seq_len,variable_names=np.asarray(data.columns))
    metrics={n:summarize(t) for n,t in totals.items()}
    if isinstance(data,PanelData) and split=='test':
        f={n:summarize(t) for n,t in fixed.items()};metrics.update(fixed=f,fixed_test_MSE=f['combined']['MSE'],
            fixed_test_MAE=f['combined']['MAE'],fixed_test_queries=int(data.fixed_mask.sum()),
            fixed_scalar_count=f['combined']['scalar_count'],full_test_MSE=metrics['combined']['MSE'],full_test_MAE=metrics['combined']['MAE'])
    for n in ('mixing_lambda','effective_candidates'):metrics[n]=stats[n].result(n=='effective_candidates')
    if bank.crest:metrics['reliability_components']={n:stats[n].result() for n in ('A_q','V_q','N_q_size')}
    if bank.crest:metrics['crest_controller']={n:stats[n].result() for n in ('cr_q','lambda_q','sigma_q')}
    timing=dict(wall_seconds=time.perf_counter()-started,backbone_forward_seconds=model_seconds,
                scalar_series_queries=len(dataset)*data.channels,**{k:bank.timings[k]-old[k] for k in bank.timings})
    write_json(directory/'metrics.json',metrics);write_json(directory/'timing.json',timing)
    print('EVALUATED',split,json.dumps(metrics),flush=True)
    return metrics,timing
