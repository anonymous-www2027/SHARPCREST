"""One independent learned or retrieval-only run using production semantics."""
import json
import os
from pathlib import Path
import platform
import random
import resource
import time
import traceback
from datetime import datetime, timezone
from unittest.mock import patch
import numpy as np
import torch
from src.reproduction.config import WEB
from src.reproduction.data import load_cohort, build_bundle, dataset_identity, array_hash
from src.reproduction.threads import retrieval_threads, thread_snapshot
from src.models.RAFT import Model
from src.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from src.utils.webscale_metrics import WebscaleMetrics


def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')


def synchronize(args):
    if args.device.type=='cuda':torch.cuda.synchronize(args.device)


def evaluate(experiment, anchor_ids=None, fixed=None, save_predictions=False):
    data,loader=experiment._get_data('test');experiment.model.eval()
    scaler=getattr(data,'forecast_scaler',None)
    full=WebscaleMetrics() if scaler else None;anchor=WebscaleMetrics() if scaler else None
    square=absolute=0.;count=0;fixed_count=0;seen=[];ps=[];ys=[]
    with torch.no_grad():
        for batch in loader:
            ids,x,y=experiment._unpack_batch(batch)
            pred=experiment.model(x.float().to(experiment.device),ids,mode='test').cpu().numpy()
            target=y[:,-experiment.args.pred_len:].numpy();qi=ids.numpy();seen.extend(qi.tolist())
            error=pred.astype(np.float64)-target.astype(np.float64)
            if not np.isfinite(error).all():raise ValueError('Nonfinite test prediction/target')
            square+=float((error**2).sum());absolute+=float(np.abs(error).sum());count+=error.size
            if scaler:
                sid,start=data.query_manifest.resolve_ids(qi)
                raw=data.series_bank.get_windows(sid,start+data.seq_len,data.pred_len)
                full.update(pred,target,raw,sid,scaler)
                keep=np.isin(sid,anchor_ids)
                if np.any(keep):
                    n=int(keep.sum());position=slice(fixed_count,fixed_count+n)
                    if not np.array_equal(qi[keep],fixed['query_id'][position]) or not np.array_equal(target[keep],fixed['normalized_target'][position]):
                        raise ValueError('Fixed evaluation query/target drift')
                    anchor.update(pred[keep],target[keep],raw[keep],sid[keep],scaler);fixed_count+=n
            if save_predictions:ps.append(pred);ys.append(target)
    if seen!=list(range(len(data))):raise ValueError('Incomplete or reordered test traversal')
    result=dict(MSE=square/count,MAE=absolute/count,scalar_count=count)
    if scaler:
        if fixed_count!=len(fixed['query_id']):raise ValueError('Missing anchor queries')
        result.update(full.result());a=anchor.result()
        result.update(full_test_MSE=result['MSE'],full_test_MAE=result['MAE'],
            fixed_test_MSE=a['normalized_MSE'],fixed_test_MAE=a['normalized_MAE'],
            fixed_test_queries=fixed_count,fixed_scalar_count=int(fixed['normalized_target'].size),
            fixed_raw_sMAPE=a['raw_sMAPE'],fixed_raw_MASE=a['raw_MASE'],fixed_mase_coverage=a['mase_coverage'])
    predictions=dict(query_id=np.asarray(seen),prediction=np.concatenate(ps),target=np.concatenate(ys)) if save_predictions else None
    return result,predictions


def execute(args):
    """Prepare retrieval once, train when requested, and evaluate the best model."""
    cohort,cohort_sha=load_cohort(args) if args.dataset in WEB else (None,None)
    if args.dry_run:
        print(json.dumps(dict(arguments=vars(args),cohort_sha256=cohort_sha,
            selected_series_count=cohort['selected_series_count'] if cohort else None),indent=2,default=str));return
    out=args.output or Path('outputs')/(args.dataset+'-'+args.method.replace('&','_')+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    out=out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError(f'Output directory is not empty: {out}')
    out.mkdir(parents=True,exist_ok=True)
    args.checkpoints=str(out/'checkpoints');args.cache_root=str(out/'cache');args.log_dir=str(out/'logs')
    torch.set_num_threads(args.cpu_threads)
    if args.device.type=='cuda':torch.cuda.set_device(args.device);torch.cuda.reset_peak_memory_stats(args.device)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    dump(out/'arguments.json',vars(args))
    import importlib.metadata
    dump(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        hnswlib=importlib.metadata.version('hnswlib'),cuda=torch.version.cuda,device=str(args.device),cpu_threads=torch.get_num_threads(),
        retrieval_cpu_threads=args.retrieval_cpu_threads,training_cpu_threads=args.cpu_threads,
        thread_policy='CPU/BLAS retrieval_cpu_threads within complete Model.prepare_dataset; restore cpu_threads afterwards. HNSW build/search have separate thread settings.',
        gpu=torch.cuda.get_device_name(args.device) if args.device.type=='cuda' else None,
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    started=time.perf_counter();timing={};fixed=None
    try:
        if cohort:
            t=time.perf_counter();bundle,fixed,audit=build_bundle(args,cohort);timing['data_preparation_seconds']=time.perf_counter()-t
            args._panel_data_bundle=bundle
            dump(out/'cohort.json',cohort);audit['cohort_manifest_sha256']=cohort_sha
            np.savez_compressed(out/'fixed_test_identity.npz',**{k:fixed[k] for k in ('query_id','source_id','query_origin')})
        else:
            from src.data_provider.legacy_data_provider import legacy_data_provider
            t=time.perf_counter();datasets={s:legacy_data_provider(args,'val' if s=='valid' else s)[0] for s in ('train','valid','test')}
            if any(len(d)==0 for d in datasets.values()):raise ValueError('Empty dataset split')
            audit=dict(dataset_identities={s:dataset_identity(d) for s,d in datasets.items()},actual_candidate_count=len(datasets['train']))
            timing['data_preparation_seconds']=time.perf_counter()-t
        dump(out/'data_identity.json',audit)
        if args.prepare_only:
            dump(out/'completion.json',dict(status='PREPARED',retrieval=False,training=False));return
        prepare=Model.prepare_dataset
        def measured(model,*datasets):
            tool=model.rt
            with retrieval_threads(args.retrieval_cpu_threads) as threads:
                timing['retrieval_threads']=threads
                synchronize(args);t=time.perf_counter()
                result=prepare(model,*datasets);synchronize(args)
                timing['retrieval_prepare_wall_seconds']=time.perf_counter()-t
            if tool.crest_enabled:
                timing['reliability_configuration']=model.reliability_configuration
            timing['threads_after_retrieval']=thread_snapshot()
            timing['production_prepare_runtime']=dict(tool.prepare_runtime)
            timing['production_split_runtime']={s:dict(v) for s,v in tool.split_runtime.items()}
            timing['retrieval_definition']='Complete Model.prepare_dataset including candidate/key/index, all three split retrievals, panel query-scale/cache serialization/cleanup; excludes data loading and learning. Native timers may overlap.'
            return result
        with patch.object(Model,'prepare_dataset',measured):
            ex=Exp_Long_Term_Forecast(args)
        dump(out/'initial_model_hashes.json',{k:array_hash(v.detach().cpu().numpy()) for k,v in ex.model.state_dict().items()})
        if not args.retrieval_only:
            # Training selects the best validation checkpoint. Fixed-anchor
            # test errors are reported afterwards and never guide selection.
            synchronize(args);t=time.perf_counter();ex.train('model');synchronize(args)
            timing['training_wall_seconds']=time.perf_counter()-t
            t=time.perf_counter();metrics,predictions=evaluate(ex,cohort['anchor_source_ids'] if cohort else None,fixed,args.save_predictions)
            synchronize(args);timing['test_evaluation_wall_seconds']=time.perf_counter()-t
            dump(out/'metrics.json',metrics)
            if predictions is not None:np.savez_compressed(out/'test_predictions.npz',**predictions)
        timing['worker_wall_seconds']=time.perf_counter()-started
        timing['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        timing['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated(args.device) if args.device.type=='cuda' else 0
        dump(out/'timing.json',timing)
        dump(out/'completion.json',dict(status='COMPLETE',training=not args.retrieval_only,method=args.method,seed=args.seed))
        print('REPRODUCTION_COMPLETE',out,flush=True)
    except BaseException:
        dump(out/'failure.json',dict(traceback=traceback.format_exc()));raise
