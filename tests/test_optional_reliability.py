"""Independent checks of CREST reliability in PIR and kNN-MTS."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.knn_mts.retrieval import RawWindowBank
from src.pir.retrieval import PIRRetrieval


def reference(similarity, futures, series, starts, n, horizon, scale, tau=1.):
    selected = []
    for i in np.argsort(-similarity, kind='stable'):
        if all(series[i] != series[j] or abs(starts[i]-starts[j]) >= horizon for j in selected):
            selected.append(i)
            if len(selected) == n:
                break
    A_q = similarity[selected].mean()
    V_q = (np.mean([np.exp(-np.mean(np.abs(futures[i]-futures[j])/scale)/tau)
                    for p, i in enumerate(selected) for j in selected[p+1:]]) if len(selected)>1 else 1.)
    return A_q, V_q, A_q*V_q, len(selected)


def pir_bank(raw=None, method='PIR_CREST'):
    generator=torch.Generator().manual_seed(22)
    keys=torch.randn(12,12,2,generator=generator)
    values=torch.randn(12,3,2,generator=generator)
    if raw is None:raw=torch.randn(12,3,2,generator=generator)*torch.tensor([2.,9.])
    args=SimpleNamespace(retrieval_num=2,retrieval_method=method,crest_pool=8,stp_weights=(.5,.25,.25),
        retrieval_stride=1,reliability_r=4,profile_sections=3,seed=0,hnsw_build_threads=1,hnsw_threads=1)
    return PIRRetrieval(keys,values,{'series_id':np.repeat([71,902],6),'start':np.tile(np.arange(6),2)},args,raw_values=raw)


@pytest.mark.parametrize('channel',[0,1])
@pytest.mark.parametrize('n',[1,4,8])
def test_pir_reference_spacing_raw_future_and_mapping(monkeypatch,channel,n):
    import src.pir.retrieval as module
    bank=pir_bank();bank.args.reliability_r=n
    sim=torch.tensor([[.95,.9,.85,.8,.75,.7,.65,.6]])
    monkeypatch.setattr(module,'robust_stp_distance',lambda *args: torch.zeros_like(sim))
    monkeypatch.setattr(module,'stp_distance_to_similarity',lambda *args:sim)
    labels=torch.tensor([[0,1,6,3,7,9,4,10]])
    scores=torch.linspace(.9,.2,8)[None]
    bank.select(torch.zeros(1,9,2),scores,labels,channel,torch.tensor([71]),torch.tensor([100]))
    expected=reference(sim[0].numpy(),bank.raw_values[labels[0],:,channel:channel+1].numpy(),
        bank.sid[labels[0]].numpy(),bank.starts[labels[0]].numpy(),n,3,bank.value_scale[channel].item())
    components=bank.last_reliability_components
    np.testing.assert_allclose([components[x].item() for x in ('A_q','V_q','cr_q','N_q_size')],expected,rtol=2e-6)
    _,_,ctl=bank.select(torch.zeros(1,9,2),scores,labels,channel,torch.tensor([71]),torch.tensor([100]))
    np.testing.assert_allclose(ctl[0].numpy(),[expected[2],.3+.6*expected[2],.05+.25*(1-expected[2])],rtol=2e-6)
    assert expected[3] <= 4  # overlapping same-series windows never refill the neighborhood


def test_pir_native_aggregation_and_components():
    bank=pir_bank();queries=bank.keys[:3];metadata=np.column_stack([np.repeat(71,3),np.arange(100,103)])
    output,_,_=bank.retrieve(queries,metadata,'test')
    for c in range(2):
        expected=(bank.values[:,:,c][bank.last['labels'][:,c]]*torch.softmax(bank.last['scores'][:,c],-1)[...,None]).sum(1)
        torch.testing.assert_close(output[:,:,c],expected,rtol=0,atol=0)
    components=bank.last['reliability_components']
    torch.testing.assert_close(bank.last['controller'][...,0],components[...,0]*components[...,1])
    assert bank.report()['controller_columns']==['cr_q','lambda_q','sigma_q']
    assert bank.report()['sigma_used_for_aggregation'] is False
    scale=np.quantile(bank.raw_values.numpy(),.75,axis=(0,1))-np.quantile(bank.raw_values.numpy(),.25,axis=(0,1))+1e-8
    np.testing.assert_allclose(bank.value_scale.numpy(),scale,rtol=1e-6)
    with pytest.raises(ValueError,match='training futures'):
        PIRRetrieval(bank.keys,bank.values,{'series_id':bank.sid.numpy(),'start':bank.starts.numpy()},bank.args)


@pytest.mark.parametrize('panel',[False,True])
def test_knn_scale_and_coordinates(panel):
    torch.manual_seed(81)
    values=torch.randn(60,1 if panel else 3)
    extra={}
    if panel:
        extra=dict(candidate_starts=np.r_[np.arange(10),30+np.arange(10)],
            candidate_series_ids=np.repeat([71,902],10),candidate_local_starts=np.tile(np.arange(10),2))
    bank=RawWindowBank(values,12,4,'cpu',k=2,crest_pool=8,reliability_k=5,profile_sections=3,method='KNN_CREST',**extra)
    labels=torch.arange(len(bank.keys))[None]
    expected=np.quantile(bank.futures(labels).numpy(),[.25,.75])
    np.testing.assert_allclose(bank.controller.value_scale,[expected[1]-expected[0]+1e-8],rtol=1e-6)
    series,starts=bank.candidate_coordinates(labels)
    np.testing.assert_array_equal(series, np.asarray(extra['candidate_series_ids'])[None] if panel else labels%3)
    np.testing.assert_array_equal(starts, np.asarray(extra['candidate_local_starts'])[None] if panel else labels//3)


@pytest.mark.parametrize('n',[1,4,8])
def test_knn_reference_and_channel_identity(monkeypatch,n):
    import src.knn_mts.controller as module
    torch.manual_seed(3)
    bank=RawWindowBank(torch.randn(60,2),12,3,'cpu',k=2,crest_pool=8,reliability_k=n,profile_sections=3,method='KNN_CREST')
    sim=torch.tensor([[.95,.9,.85,.8,.75,.7,.65,.6]])
    monkeypatch.setattr(module,'robust_stp_distance',lambda *args: torch.zeros_like(sim))
    monkeypatch.setattr(module,'stp_distance_to_similarity',lambda *args:sim)
    labels=torch.tensor([[0,2,1,6,3,7,8,9]])
    series,starts=bank.candidate_coordinates(labels)
    expected=reference(sim[0].numpy(),bank.futures(labels)[0].numpy(),series[0].numpy(),starts[0].numpy(),n,3,bank.controller.value_scale.item())
    cr_q,lambda_q,sigma_q=bank.controller.reliability(torch.randn(1,12),labels)
    np.testing.assert_allclose([bank.controller.last_reliability_components[x].item() for x in ('A_q','V_q','cr_q','N_q_size')],expected,rtol=2e-6)
    np.testing.assert_allclose([cr_q.item(),lambda_q.item(),sigma_q.item()],[expected[2],.3+.6*expected[2],.05+.25*(1-expected[2])],rtol=2e-6)


def test_panel_requires_true_coordinates():
    with pytest.raises(ValueError,match='true series IDs'):
        RawWindowBank(torch.randn(60,1),12,3,'cpu',k=2,crest_pool=8,profile_sections=3,
                      method='KNN_CREST',candidate_starts=np.arange(15))


@pytest.mark.parametrize('tau',[0.,-1.,float('nan')])
def test_invalid_future_temperature(tau):
    with pytest.raises(ValueError,match='tau_v'):
        RawWindowBank(torch.randn(60,1),12,3,'cpu',k=2,crest_pool=8,profile_sections=3,method='KNN_CREST',tau_v=tau)


def test_pir_prepared_cache_ignores_targets():
    class Dataset(torch.utils.data.Dataset):
        flag='test'
        metadata=np.array([[71,100],[902,100],[71,101]])
        def __len__(self):return 3
        def __getitem__(self,index):
            return bank.keys[index],torch.full((3,2),float('nan')),torch.zeros(12,5),torch.zeros(3,5),index
    bank=pir_bank();bank.args.retrieval_batch_size=2
    past=bank.keys[:3];query=past-past.mean(1,keepdim=True)
    query=query/torch.sqrt(query.var(1,keepdim=True,unbiased=False)+1e-5)
    output,scores,_=bank.retrieve(query,Dataset.metadata,'test')
    controller=bank.last['controller'].clone();components=bank.last['reliability_components'].clone()
    bank.prepare_queries([Dataset()])
    cached,cached_scores,_=bank.lookup([0,1,2],'test')
    for actual,expected in [(cached,output),(cached_scores,scores),(bank.last['controller'],controller),
                            (bank.last['reliability_components'],components)]:
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)


def test_pir_index_keeps_pre_normalization_train_futures():
    from src.pir.exp.exp_long_term_forecasting_pir import Exp_Long_Term_Forecast_PIR
    prototype=pir_bank()
    args=prototype.args;args.pred_len=3
    raw_past=prototype.keys*5+10;raw_future=prototype.raw_values*3+20
    class Dataset(torch.utils.data.Dataset):
        metadata=np.column_stack([prototype.sid.numpy(),prototype.starts.numpy()])
        def __len__(self):return len(raw_past)
        def __getitem__(self,index):return raw_past[index],raw_future[index],torch.zeros(12,5),torch.zeros(3,5),index
    class Model:
        def construct_index(self,count):
            self.keys=torch.empty(count,12,2);self.values=torch.empty(count,3,2)
        def add_key_value(self,past,future,index):
            mean=past.mean(1,keepdim=True);std=torch.sqrt(past.var(1,keepdim=True,unbiased=False)+1e-5)
            self.keys[index]=(past-mean)/std;self.values[index]=(future-mean)/std
    exp=SimpleNamespace(device=torch.device('cpu'),args=args,model=Model())
    dataset=Dataset()
    Exp_Long_Term_Forecast_PIR.prepare_retrieval_index(exp,dataset,
        torch.utils.data.DataLoader(dataset,batch_size=5,shuffle=True,generator=torch.Generator().manual_seed(6)))
    adapter=exp.model.retrieval_adapter
    torch.testing.assert_close(adapter.raw_values,raw_future,rtol=0,atol=0)
    assert not torch.allclose(adapter.raw_values,adapter.values)
