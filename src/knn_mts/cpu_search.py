"""Optional CPU engines over already window-centered, unit-normalized keys."""
import math
import time
from collections import Counter
import numpy as np
import torch


class ExactCPUIndex:
    def __init__(self, keys, threads=32):
        self.keys=torch.from_numpy(np.ascontiguousarray(keys,dtype=np.float32))
        self.threads=threads;self.seconds=0.;self.calls=0;self.batches=Counter()
        self.info=dict(backend='cpu_exact',threads=threads,count=len(keys),dimension=keys.shape[1])

    def knn_query(self, queries, k, num_threads):
        if num_threads!=self.threads or torch.get_num_threads()!=self.threads:
            raise ValueError('CPU exact Torch thread count differs from requested threads')
        start=time.perf_counter();q=torch.from_numpy(np.ascontiguousarray(queries,dtype=np.float32))
        scores=q@self.keys.T;scores.clamp_(-1,1).neg_().add_(1)
        distances,labels=scores.topk(k,largest=False,sorted=True)
        result=labels.numpy(),distances.numpy()
        self.seconds+=time.perf_counter()-start;self.calls+=1;self.batches[len(q)]+=1
        return result


class FaissCPUIndex:
    """Original auto thresholds/OPQ32+IVF+PQ32, with zero padding for PQ32.

    Inputs are Pearson keys. Approximate scores never become final distances:
    RawWindowBank refines selected candidates in the original representation.
    """
    def __init__(self, keys, threads=32, nprobe=32):
        import faiss
        keys=np.ascontiguousarray(keys,dtype=np.float32)
        if keys.ndim!=2 or len(keys)<1 or threads<1 or nprobe<1 or not np.isfinite(keys).all():
            raise ValueError('invalid FAISS keys or parameters')
        self.faiss=faiss;self.threads=threads;self.dimension=keys.shape[1]
        self.seconds=0.;self.calls=0;self.batches=Counter()
        faiss.omp_set_num_threads(threads)
        self.index_dimension=math.ceil(self.dimension/32)*32 if len(keys)>=30000 else self.dimension
        encoded=self.pad(keys)
        clusters=min(int(4*math.sqrt(len(keys))),len(keys)//30,131072)
        if len(keys)<30000:
            # Flat already uses row IDs; original IDMap,,Flat is malformed.
            factory='Flat'
        elif len(keys)<1000000:
            factory=f'OPQ32_{self.index_dimension},IVF{clusters},PQ32'
        else:
            factory=f'OPQ32_{self.index_dimension},IVF{clusters}_HNSW32,PQ32'
        self.index=faiss.index_factory(self.index_dimension,factory,faiss.METRIC_INNER_PRODUCT)
        tick=time.perf_counter();self.index.train(encoded[-1000000:]);train_seconds=time.perf_counter()-tick
        tick=time.perf_counter();self.index.add(encoded);add_seconds=time.perf_counter()-tick
        if len(keys)>=30000:
            faiss.ParameterSpace().set_index_parameter(self.index,'nprobe',nprobe)
            assert faiss.extract_index_ivf(self.index).nprobe==nprobe
        if len(keys)>=1000000:
            faiss.ParameterSpace().set_index_parameter(self.index,'quantizer_efSearch',8)
        assert self.index.ntotal==len(keys)
        self.info=dict(backend='faiss_cpu',factory=factory,threads=threads,nprobe=nprobe if len(keys)>=30000 else None,
            count=len(keys),dimension=self.dimension,index_dimension=self.index_dimension,
            zero_padding=self.index_dimension-self.dimension,metric='inner product of Pearson keys',
            train_seconds=train_seconds,add_seconds=add_seconds,faiss_version=faiss.__version__,
            training_policy='up to last1000000 train candidates; unchanged FAISS clustering defaults')

    def pad(self,queries):
        q=np.ascontiguousarray(queries,dtype=np.float32)
        if q.ndim!=2 or q.shape[1]!=self.dimension:raise ValueError('FAISS query dimension mismatch')
        if self.index_dimension==self.dimension:return q
        return np.pad(q,((0,0),(0,self.index_dimension-self.dimension)))

    def knn_query(self,queries,k,num_threads):
        if num_threads!=self.threads or self.faiss.omp_get_max_threads()!=self.threads:
            raise ValueError('FAISS thread setting changed')
        tick=time.perf_counter();scores,labels=self.index.search(self.pad(queries),k)
        self.seconds+=time.perf_counter()-tick;self.calls+=1;self.batches[len(queries)]+=1
        if np.any(labels<0) or not np.isfinite(scores).all():
            raise RuntimeError('FAISS returned fewer than K candidates; no silent exact fallback')
        return labels,1-scores
