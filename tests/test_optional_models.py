"""Portable model contracts, anchor identities and complete CPU smoke runs."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from src.reproduction.model_data import ROOT, cohort, prepare_knn_panel
from src.reproduction.models import parser, resolve, run_knn


def isolated_runtime(function):
    """Keep native OpenMP/FAISS state out of subsequent DataLoader fork tests."""
    import functools
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        import os
        import subprocess
        import sys
        if os.environ.get('SHARP_OPTIONAL_TEST_CHILD') == function.__name__:
            return function(*args, **kwargs)
        environment = dict(os.environ, SHARP_OPTIONAL_TEST_CHILD=function.__name__)
        result = subprocess.run([sys.executable, '-m', 'pytest',
            str(Path(__file__).resolve())+'::'+function.__name__, '-q', '--disable-warnings'],
            env=environment, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout+'\n'+result.stderr
    return wrapped


@pytest.mark.parametrize('framework', ['pir', 'knn_mts'])
@pytest.mark.parametrize('dataset', ['ETTh1','ETTh2','ETTm1','ETTm2','Exchange','Weather',
                                     'quito_hour','extended_wikipedia','tailedts_202401'])
@pytest.mark.parametrize('method', ['BASE','SHARP','CREST','S&C'])
def test_public_protocol(framework, dataset, method):
    web = dataset in ('quito_hour','extended_wikipedia','tailedts_202401')
    a = parser().parse_args(['--framework',framework,'--data-type','web' if web else 'general',
        '--data-dir',str(ROOT/'data'/('web' if web else 'general')),'--dataset',dataset,'--method',method])
    result = resolve(a)
    assert result.learning_rate == .0001 and result.stp_weights == [.5,.25,.25]
    assert result.batch_size == 32
    assert result.seq_len == ((168 if dataset=='tailedts_202401' else 576) if web else 96 if framework=='pir' else 720)
    if framework == 'knn_mts':
        assert result.value_transfer == 'query_last' and result.mmr_chunk == result.retrieval_batch_size == 1024
        assert (result.epochs,result.patience) == ((100,10) if web else (10,3))
    else:
        assert result.refine_lr == .0001 and result.retrieval_num == 50
        assert result.sharp_c == (5 if web else 2)


@pytest.mark.parametrize('framework,extra', [('pir',['--learning_rate','.001']),
    ('knn_mts',['--stp-weights','1','0','0'])])
def test_excluded_settings_rejected(framework, extra):
    a = parser().parse_args(['--framework',framework,'--data-type','general','--data-dir',str(ROOT/'data/general'),
                            '--dataset','ETTh1','--method','CREST','--']+extra)
    with pytest.raises(ValueError,match='Public RQ1'):
        resolve(a)


@pytest.mark.parametrize('dataset,h', [('extended_wikipedia',7),('quito_hour',48)])
def test_public_panel_identity(tmp_path, dataset, h):
    from src.knn_mts.panel import PanelData
    from src.pir.data_provider.pir_panel import PanelBundle
    length=576; m=cohort(dataset)
    path=prepare_knn_panel(ROOT/'data/web',dataset,h,tmp_path/'panel')
    data=PanelData(path,dataset,length,h)
    args=SimpleNamespace(panel_manifest=ROOT/'configs/cohorts'/dataset/'200000.json',
        panel_anchor_manifest=ROOT/'configs/cohorts'/dataset/'100000.json',panel_root=ROOT/'data/web',
        panel_channel=0,seq_len=length,pred_len=h,label_len=0,retrieval_stride=1)
    bundle=PanelBundle(args)
    for i,sid in enumerate(data.series_ids):
        np.testing.assert_array_equal(data.values[i].numpy(),bundle.data[int(sid)])
    np.testing.assert_array_equal(data.fixed_mask,bundle.datasets['test'].fixed_mask)
    for split in ['train','val','test']:
        np.testing.assert_array_equal(bundle.datasets[split].metadata,
            np.column_stack([data.identities[split]['source_id'],data.identities[split]['past_start']]))
    assert not np.any(data.candidate_identity['past_start']+length+h > m['split']['train_end'])
    # Corrupting any cached array must prevent its reuse.
    with (path/'values.npy').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        prepare_knn_panel(ROOT/'data/web',dataset,h,path)


@pytest.fixture
def tiny_csv(tmp_path):
    rng=np.random.default_rng(12)
    f=pd.DataFrame(rng.normal(size=(800,2)),columns=['x','OT'])
    f.insert(0,'date',pd.date_range('2001-01-01',periods=len(f),freq='h'))
    path=tmp_path/'Exchange_wide.csv';f.to_csv(path,index=False)
    return path


@isolated_runtime
def test_knn_fourway_training_and_shared_model(tmp_path, tiny_csv):
    from src.knn_mts.runner import parser as native_parser
    from src.knn_mts.model import make_model, forecast
    from src.knn_mts.retrieval import RawWindowBank
    from src.knn_mts.data import GeneralData
    checkpoints=[]
    for method in ['KNN','KNN_SHARP','KNN_CREST','KNN_SHARP_CREST']:
        out=tmp_path/method
        args=native_parser().parse_args(['--dataset','Exchange','--csv',str(tiny_csv),'--output',str(out),
            '--seq-len','24','--pred-len','7','--epochs','1','--patience','1','--device','cpu',
            '--num-workers','0','--threads','1','--method',method,'--artifact-mode','minimal',
            '--hnsw-build-threads','1','--hnsw-search-threads','1'])
        run_knn(args,tmp_path/'shared')
        m=json.loads((out/'metrics.json').read_text())
        assert np.isfinite(m['test']['combined']['MSE'])
        t=json.loads((out/'timing.json').read_text())
        assert t['train_retrieval_seconds'] is None and t['train_retrieval_status']=='not_performed'
        protocol=json.loads((out/'training_protocol.json').read_text())
        assert protocol['checkpoint_reused'] == (method!='KNN')
        checkpoints.append(protocol['checkpoint_sha256'])
        assert {p.name for p in (out/'test').glob('*.npy')} == {'backbone_pred.npy','knn_pred.npy','target.npy','neighbor_ids.npy'}
    assert len(set(checkpoints))==1
    # A different training configuration must not accept the old checkpoint.
    args.output=tmp_path/'bad';args.checkpoint=Path(protocol['checkpoint']);args.patience=2
    from src.knn_mts.runner import run
    with pytest.raises(ValueError,match='training settings'):
        run(args)


@isolated_runtime
def test_knn_pearson_transfer_and_batch_invariance():
    from src.knn_mts.retrieval import RawWindowBank
    torch.set_num_threads(1);torch.manual_seed(71)
    train=torch.randn(400,2); query=torch.randn(23,24); ids=torch.arange(23)
    bank=RawWindowBank(train,24,7,'cpu',method='KNN_CREST',value_transfer='query_last',seed=4)
    whole=bank.predict(query,ids); weights=bank.last['weights'].clone()
    labels=whole[2]; starts=bank.candidate_past_starts(labels)
    expected=((bank.futures(labels)-train[starts+23,labels%2][...,None]+query[:,-1,None,None])*weights[...,None]).sum(1)
    torch.testing.assert_close(whole[0],expected,rtol=0,atol=0)
    pieces=[bank.predict(query[i:i+5],ids[i:i+5]) for i in range(0,len(query),5)]
    for i in range(4):torch.testing.assert_close(whole[i],torch.cat([p[i] for p in pieces]),rtol=1e-5,atol=1e-6)


@isolated_runtime
def test_pir_fourway_training_and_rng_boundary(tmp_path, tiny_csv):
    from src.pir.runner import build_parser, run
    torch.set_num_threads(1)
    boundaries=[]
    for method in ['PIR','PIR_SHARP','PIR_CREST','PIR_SHARP_CREST']:
        out=tmp_path/method
        args=build_parser().parse_args(['--result_dir',str(out),'--cpu','--root_path',str(tiny_csv.parent),
            '--data_path',tiny_csv.name,'--data','custom','--enc_in','2','--dec_in','2','--c_out','2',
            '--pred_len','7','--train_epochs','1','--refine_epochs','1','--patience','1',
            '--retrieval_method',method,'--torch_threads','1','--hnsw_threads','1','--hnsw_build_threads','1',
            '--shared_pretraining_cache',str(tmp_path/'shared')])
        run(args)
        m=json.loads((out/'metrics.json').read_text());assert np.isfinite(m['MSE'])
        audit=json.loads((out/'shared_pretraining.json').read_text())
        boundaries.append((audit['boundary_model_hash'],audit['boundary_rng_hash']))
        assert json.loads((out/'completion.json').read_text())['state']=='complete'
    assert len(set(boundaries))==1


@isolated_runtime
def test_faiss_flat_uses_pearson_keys():
    pytest.importorskip('faiss')
    from src.knn_mts.retrieval import RawWindowBank
    torch.set_num_threads(1);torch.manual_seed(517)
    train=torch.randn(500,2);q=torch.randn(11,24)
    exact=RawWindowBank(train,24,7,'cpu',value_transfer='query_last')
    flat=RawWindowBank(train,24,7,'cpu',value_transfer='query_last',search_backend='faiss_cpu',cpu_search_threads=2)
    assert flat.index.info['factory']=='Flat'
    for a,b in zip(exact.predict(q),flat.predict(q)):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


@isolated_runtime
def test_pir_panel_fourway_training(tmp_path, monkeypatch):
    from src.pir.runner import build_parser, run
    from src.pir.data_provider import pir_panel
    ids=[7,13]; length=800
    rng=np.random.default_rng(81)
    raw={sid:rng.normal(size=(length,1)).astype(np.float32) for sid in ids}
    m=dict(dataset='smoke_panel', ordered_source_ids=ids,
           split=dict(train_end=560,valid_end=640,test_end=800,expected_length=800))
    cohort_path=tmp_path/'cohort.json'; cohort_path.write_text(json.dumps(m))
    anchor=tmp_path/'anchor.json';anchor.write_text(json.dumps(dict(m,ordered_source_ids=ids[:1])))
    monkeypatch.setattr(pir_panel,'load_series',lambda *args:(raw,{}))
    for method in ['PIR','PIR_SHARP','PIR_CREST','PIR_SHARP_CREST']:
        out=tmp_path/method
        args=build_parser().parse_args(['--result_dir',str(out),'--cpu','--panel_root',str(tmp_path),
            '--panel_manifest',str(cohort_path),'--panel_anchor_manifest',str(anchor),
            '--seq_len','96','--pred_len','7','--enc_in','1','--dec_in','1','--c_out','1',
            '--train_epochs','1','--refine_epochs','1','--patience','1','--torch_threads','1',
            '--hnsw_threads','1','--hnsw_build_threads','1','--retrieval_method',method,
            '--shared_pretraining_cache',str(tmp_path/'shared')])
        run(args)
        metric=json.loads((out/'metrics.json').read_text())
        pred=np.load(out/'pred.npy');true=np.load(out/'true.npy');mask=np.load(out/'fixed_test_mask.npy')
        assert metric['fixed_test_queries']==154 and mask.sum()*2==len(mask)
        assert metric['fixed_test_MSE']==np.mean((pred[mask].astype('float64')-true[mask])**2)


@isolated_runtime
def test_knn_panel_fourway_training(tmp_path):
    from src.knn_mts.runner import parser as native_parser
    from src.preprocessing.io import write_json, file_hash
    from src.knn_mts.panel import PanelData
    # A small valid artifact exercises the same train/anchor path without a large bank.
    directory=tmp_path/'panel';directory.mkdir()
    ids=np.array([7,13],dtype=np.int64);length=24;h=7;te,ve,end=200,240,280
    raw=np.random.default_rng(37).normal(size=(2,end,1)).astype('float32')
    mean=raw[:,:te].astype('float64').mean(1);scale=raw[:,:te].astype('float64').std(1)+1e-8
    values=((raw.astype('float64')-mean[:,None])/scale[:,None]).astype('float32')
    np.save(directory/'raw_values.npy',raw);np.save(directory/'values.npy',values)
    np.savez(directory/'scaler.npz',mean=mean,scale=scale,series_id=ids)
    payload={}
    for split,start,stop in [('train',0,te),('val',te-length,ve),('test',ve-length,end)]:
        starts=np.arange(start,stop-length-h+1,dtype=np.int64)
        payload[split+'_source_id']=np.repeat(ids,len(starts));payload[split+'_past_start']=np.tile(starts,2)
    np.savez(directory/'identities.npz',candidate_source_id=payload['train_source_id'],candidate_past_start=payload['train_past_start'],**payload)
    keep=np.flatnonzero(payload['test_source_id']==7);origins=payload['test_past_start'][keep]+length
    np.savez(directory/'fixed_test.npz',query_id=keep,source_id=payload['test_source_id'][keep],query_origin=origins,
        raw_target=raw[0,origins[:,None]+np.arange(h)],normalized_target=values[0,origins[:,None]+np.arange(h)])
    write_json(directory/'rq1_manifest.json',dict(dataset='extended_wikipedia',L=length,H=h,nominal_scale=200000,
        ordered_source_ids=ids.tolist(),anchor_source_ids=[7],cohort_hash='synthetic-smoke',
        split=dict(train_end=te,valid_end=ve,test_end=end)))
    write_json(directory/'completion.json',dict(state='complete',hashes={p.name:file_hash(p) for p in directory.iterdir()}))
    for method in ['KNN','KNN_SHARP','KNN_CREST','KNN_SHARP_CREST']:
        out=tmp_path/method
        args=native_parser().parse_args(['--dataset','extended_wikipedia','--panel-artifact',str(directory),
            '--output',str(out),'--seq-len',str(length),'--pred-len',str(h),'--epochs','1','--patience','1',
            '--num-workers','0','--threads','1','--device','cpu','--method',method,
            '--hnsw-build-threads','1','--hnsw-search-threads','1'])
        run_knn(args,tmp_path/'cache')
        metric=json.loads((out/'metrics.json').read_text())['test']
        assert metric['fixed_test_queries']==len(keep) and np.isfinite(metric['fixed_test_MSE'])
        identity=np.load(out/'test/query_identity.npz')
        assert np.array_equal(identity['source_id'],payload['test_source_id'])


@isolated_runtime
def test_knn_full_artifact_reliability_components(tmp_path, tiny_csv):
    from src.knn_mts.runner import parser as native_parser, run
    out=tmp_path/'full'
    args=native_parser().parse_args(['--dataset','Exchange','--csv',str(tiny_csv),'--output',str(out),
        '--seq-len','24','--pred-len','7','--epochs','1','--patience','1','--device','cpu',
        '--num-workers','0','--threads','1','--method','KNN_CREST','--artifact-mode','full'])
    run(args)
    for split in ('val','test'):
        metrics=json.loads((out/split/'metrics.json').read_text())
        assert set(metrics['reliability_components'])=={'A_q','V_q','N_q_size'}
        assert set(metrics['crest_controller'])=={'cr_q','lambda_q','sigma_q'}
        controller=np.load(out/split/'crest_controller.npy')
        np.testing.assert_allclose(controller[...,1],.3+.6*controller[...,0],rtol=1e-6)
        np.testing.assert_allclose(controller[...,2],.05+.25*(1-controller[...,0]),rtol=1e-6)
