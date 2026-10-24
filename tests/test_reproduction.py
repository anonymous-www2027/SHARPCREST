"""Portable entrypoint tests: cohort integrity and complete four-way execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from src.reproduction.config import parse_args, ROOT
from src.reproduction.data import array_hash, load_cohort, build_bundle
from src.reproduction.runner import execute


def test_config_overrides_and_method_mapping(tmp_path):
    p=tmp_path/'config.json';p.write_text(json.dumps(dict(dataset='extended_wikipedia',method='CREST',seed=42,topm=7,mmr_candidate_pool=18,profile_reliability_k=11)))
    a=parse_args(['--config',str(p),'--seed','123','--hnsw_seed','456','--stp_state_weight','.5','--stp_dispersion_weight','.25','--stp_transition_weight','.25'])
    assert (a.seed,a.hnsw_seed,a.topm,a.mmr_candidate_pool,a.profile_reliability_k)==(123,456,7,18,11)
    assert (a.retrieval_backend,a.retrieval_controller)==('exact','crest')
    assert parse_args(['--method','S&C']).retrieval_backend=='hnsw'
    with pytest.raises(SystemExit):parse_args(['--method','CREST','--topm','41','--mmr_candidate_pool','40'])
    with pytest.raises(SystemExit):parse_args(['--method','RAFT','--retrieval_backend','hnsw'])
    with pytest.raises(SystemExit):parse_args(['--seed',str(2**32)])
    assert (a.cpu_threads,a.retrieval_cpu_threads)==(4,32)
    with pytest.raises(SystemExit):parse_args(['--retrieval_cpu_threads','0'])
    assert parse_args(['--retrieval_cpu_threads','8']).retrieval_cpu_threads==8
    p.write_text('{"unknown_parameter": 1}')
    with pytest.raises(SystemExit):parse_args(['--config',str(p)])


def test_published_nested_cohorts():
    for ds in ['quito_hour','extended_wikipedia','tailedts_202401']:
        previous=[]
        for p in sorted((ROOT/'configs/cohorts'/ds).glob('*.json'),key=lambda p:int(p.stem)):
            m,_=load_cohort(parse_args(['--dataset',ds,'--scale',p.stem]))
            assert m['ordered_source_ids'][:len(previous)]==previous
            previous=m['ordered_source_ids']


def test_stp_defaults_and_explicit_weight_override(tmp_path):
    from types import SimpleNamespace
    from src.run import build_parser
    from src.models.RAFT import Model
    from src.layers.Retrieval import RetrievalTool
    from src.utils.profile_reliability import stp_weights_from_args
    expected = (0.5, 0.25, 0.25)
    raw = build_parser().parse_args(['--data', 'ETTh1', '--model_id', 'test', '--enc_in', '7'])
    assert stp_weights_from_args(raw) == expected
    for config in (None, 'rq1_web200k.json', 'rq1_general.json'):
        argv = ['--method', 'CREST', '--no_use_gpu']
        if config:
            argv += ['--config', str(ROOT/'configs'/config)]
        args = parse_args(argv)
        assert stp_weights_from_args(args) == expected
        assert Model(args).rt.stp_weights == expected
    for name in ('state', 'dispersion', 'transition'):
        delattr(args, f'stp_{name}_weight')
    assert Model(args).rt.stp_weights == expected
    assert stp_weights_from_args(SimpleNamespace()) == expected
    assert RetrievalTool(seq_len=24, pred_len=4, channels=1, n_period=1).stp_weights == expected
    config = tmp_path/'weights.json'
    config.write_text(json.dumps(dict(stp_state_weight=.2, stp_dispersion_weight=.3, stp_transition_weight=.5)))
    args = parse_args(['--config', str(config), '--method', 'CREST', '--no_use_gpu'])
    assert Model(args).rt.stp_weights == (.2, .3, .5)
    args = parse_args(['--config', str(config), '--method', 'CREST', '--no_use_gpu',
                      '--stp_state_weight', '.6', '--stp_dispersion_weight', '.1', '--stp_transition_weight', '.3'])
    assert Model(args).rt.stp_weights == (.6, .1, .3)


@pytest.fixture
def tiny_data(tmp_path):
    directory=tmp_path/'data/web/processed/extended_wikipedia';directory.mkdir(parents=True)
    values=np.random.default_rng(711).normal(size=(3,120,1)).astype(np.float32)
    pq.write_table(pa.table({'values':[s.ravel().tolist() for s in values]}),directory/'part.parquet')
    pq.write_table(pa.table(dict(series_id=[0,1,2],part_file=['part.parquet']*3,row_group=[0]*3,row_in_group=[0,1,2],length=[120]*3,n_channels=[1]*3)),directory/'series_index.parquet')
    ids=[2,0,1];h=hashlib.sha256(np.array(ids,dtype='<i8').tobytes()).hexdigest()
    m=dict(schema_version=1,population_mode='nested_whole_series',dataset='extended_wikipedia',nominal_scale=200000,
        L=24,H=4,split=dict(expected_length=120,train_end=80,valid_end=100,test_end=120),ordered_source_ids=ids,
        selected_identities=[],anchor_source_ids=[2],selected_series_count=3,selection_seed=0,cohort_hash=h,
        selected_series_hash=h,selected_raw_bank_hash=array_hash(values.reshape(-1,1)),expected={})
    p=tmp_path/'cohort.json';p.write_text(json.dumps(m))
    general=tmp_path/'data/general/experiment';general.mkdir(parents=True)
    df=pd.DataFrame(np.random.default_rng(71).normal(size=(360,8)),columns=['x'+str(i) for i in range(7)]+['OT'])
    df.insert(0,'date',pd.date_range('2000-01-01',periods=360,freq='h'));df.to_csv(general/'Exchange_wide.csv',index=False)
    return tmp_path/'data',p


@pytest.mark.parametrize('dataset',['extended_wikipedia','Exchange'])
@pytest.mark.parametrize('method',['RAFT','SHARP','CREST','S&C'])
def test_four_methods_train_and_evaluate(tmp_path,tiny_data,dataset,method,monkeypatch):
    from src.models.RAFT import Model
    from src.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    prepare=Model.prepare_dataset;train=Exp_Long_Term_Forecast.train
    def checked_prepare(self,*datasets):
        assert torch.get_num_threads()==2
        return prepare(self,*datasets)
    def checked_train(self,*args,**kwargs):
        assert torch.get_num_threads()==1
        return train(self,*args,**kwargs)
    monkeypatch.setattr(Model,'prepare_dataset',checked_prepare)
    monkeypatch.setattr(Exp_Long_Term_Forecast,'train',checked_train)
    root,cohort=tiny_data;out=tmp_path/'output'
    argv=['--dataset',dataset,'--method',method,'--data_root',str(root),'--no_use_gpu',
        '--seq_len','24','--pred_len','4','--label_len','0','--train_epochs','1','--patience','0',
        '--topm','4','--mmr_candidate_pool','8','--profile_reliability_k','6',
        '--retrieval_batch_size','32','--value_mmr_chunk_size','16','--batch_size','32',
        '--hnsw_build_num_threads','1','--hnsw_search_num_threads','1','--cpu_threads','1',
        '--retrieval_cpu_threads','2','--output',str(out),'--save_predictions']
    if dataset=='extended_wikipedia':argv+=['--cohort_file',str(cohort)]
    a=parse_args(argv);execute(a)
    metrics=json.loads((out/'metrics.json').read_text());preds=np.load(out/'test_predictions.npz')
    err=preds['prediction'].astype(np.float64)-preds['target'].astype(np.float64)
    assert metrics['MSE']==pytest.approx((err**2).mean())
    if dataset=='extended_wikipedia':
        ids=np.load(out/'fixed_test_identity.npz')['query_id'];err=err[ids]
        assert metrics['fixed_test_MSE']==pytest.approx((err**2).mean())
        assert metrics['fixed_test_MAE']==pytest.approx(np.abs(err).mean())
        assert metrics['fixed_test_queries']==17
    assert json.loads((out/'checkpoints/model/training_protocol.json').read_text())['best_checkpoint_reloaded']
    assert json.loads((out/'completion.json').read_text())['status']=='COMPLETE'
    timing=json.loads((out/'timing.json').read_text())
    if method in ('CREST', 'S&C'):
        reliability = timing['reliability_configuration']
        assert reliability['formula'] == 'cr_q=A_q*V_q'
        assert reliability['neighborhood'] == 'stp_ranked_spaced_up_to_n'
        assert reliability['min_spacing'] == a.pred_len
        assert reliability['shortfall_policy'] == 'use_available'
        assert reliability['tau_v'] == 1.0
        assert all(v > 0 for v in reliability['value_scale'])
    else:
        assert 'reliability_configuration' not in timing
    assert timing['retrieval_threads']['torch_threads']==2
    assert timing['threads_after_retrieval']['torch_threads']==1
    assert all(p['num_threads']==2 for p in timing['retrieval_threads']['pools'])
    with pytest.raises(ValueError,match='not empty'):execute(a)


@pytest.mark.parametrize('fail',[False,True])
def test_retrieval_thread_limits_restore_on_success_and_failure(fail):
    from threadpoolctl import threadpool_limits
    from src.reproduction.threads import retrieval_threads,thread_snapshot
    previous=torch.get_num_threads()
    try:
        with threadpool_limits(limits=4):
            torch.set_num_threads(4)
            before=thread_snapshot()
            def run():
                with retrieval_threads(32) as observed:
                    assert observed['torch_threads']==32
                    assert all(p['num_threads']==32 for p in observed['pools'])
                    if fail:raise RuntimeError('retrieval failed')
            if fail:
                with pytest.raises(RuntimeError,match='retrieval failed'):run()
            else:run()
            assert thread_snapshot()==before
    finally:
        torch.set_num_threads(previous)


def test_crest_defaults_keep_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    a=parse_args(['--method','CREST'])
    assert a.mmr_device==a.retrieval_postprocess_device=='cuda'
    assert (a.cpu_threads,a.retrieval_cpu_threads)==(4,32)


def test_cohort_tampering_and_horizon_identity(tmp_path,tiny_data):
    root,path=tiny_data
    a=parse_args(['--data_root',str(root),'--cohort_file',str(path),'--seq_len','24','--pred_len','4'])
    m,_=load_cohort(a);b,f,audit=build_bundle(a,m)
    a.pred_len=8;b2,f2,audit2=build_bundle(a,m)
    assert np.array_equal(b.candidate_manifest.series_ids,b2.candidate_manifest.series_ids)
    assert audit['actual_candidate_count']-audit2['actual_candidate_count']==3*4
    m['anchor_source_ids']=[0];path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='anchor'):load_cohort(a)


def test_general_hnsw_threads_are_applied(tmp_path,tiny_data):
    from src.data_provider.legacy_data_provider import legacy_data_provider
    from src.models.RAFT import Model
    root,_=tiny_data
    a=parse_args(['--dataset','Exchange','--method','SHARP','--data_root',str(root),
        '--seq_len','24','--pred_len','4','--label_len','0','--topm','4','--no_use_gpu',
        '--hnsw_build_num_threads','1','--hnsw_search_num_threads','2'])
    data,_=legacy_data_provider(a,'train');tool=Model(a).rt;tool.prepare_dataset(data)
    assert all(index.num_threads==2 for index in tool.hnsw_indices)


def test_cli_from_other_directory_without_historical_campaigns(tmp_path):
    import subprocess,sys
    root=str(ROOT)
    code='''
import sys, importlib.abc
class RejectCampaigns(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'experiments' or fullname.startswith('experiments.'):
            raise AssertionError('Historical campaign import: '+fullname)
sys.meta_path.insert(0, RejectCampaigns())
sys.path.insert(0, ROOT)
sys.argv=['reproduce.py','--dataset','extended_wikipedia','--method','S&C','--seed','123','--cpu_threads','2','--dry_run']
import reproduce
reproduce.main()
'''.replace('ROOT',repr(root))
    out=subprocess.check_output([sys.executable,'-c',code],cwd=tmp_path,text=True)
    value=json.loads(out)
    assert value['arguments']['seed']==value['arguments']['hnsw_seed']==123
    assert value['arguments']['cpu_threads']==2
