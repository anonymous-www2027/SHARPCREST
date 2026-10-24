"""Verify distributed data against the published cohorts and evaluation targets."""
import json
from pathlib import Path
import pytest
from src.reproduction.config import ROOT, parse_args
from src.reproduction.data import load_cohort, build_bundle
from src.preprocessing.package import PackagedSource


@pytest.mark.parametrize('dataset', ['extended_wikipedia', 'quito_hour'])
@pytest.mark.parametrize('scale', [100000, 200000, 400000, 800000])
def test_packaged_data_reproduces_published_inputs(dataset, scale):
    root = ROOT/'data/web'
    a = parse_args(['--dataset',dataset,'--scale',str(scale),'--canonical_root',str(root), '--no_use_gpu'])
    m,_ = load_cohort(a)
    source=PackagedSource(root,dataset,m)
    assert list(source.values)==m['ordered_source_ids']
    bundle, fixed, audit = build_bundle(a,m)
    assert audit['historical_identity_checked']
    # The 200K manifest stores the fixed-anchor hashes shared by all four scales.
    reference=json.loads((ROOT/'configs/cohorts'/dataset/'200000.json').read_text())
    assert audit['fixed_hashes']==reference['expected'][str(a.pred_len)]['fixed_hashes']
    assert audit['selected_series_count'] == len(m['ordered_source_ids'])
    assert set(fixed['source_id']) == set(m['anchor_source_ids'])
    assert bundle.series_bank.total_time_steps == len(m['ordered_source_ids'])*m['split']['expected_length']


def test_bundled_default_and_explicit_data_root(tmp_path):
    a=parse_args(['--dataset','quito_hour'])
    if not (Path('./data')/'web/processed').exists():
        assert Path(a.canonical_root)==ROOT/'data/web'
    explicit=parse_args(['--dataset','quito_hour','--data_root',str(tmp_path/'missing')])
    assert Path(explicit.canonical_root)==tmp_path/'missing/web/processed'


def test_package_checksum_and_identity_guard(tmp_path):
    import shutil
    ds='extended_wikipedia'
    shutil.copytree(ROOT/'data/web'/ds,tmp_path/ds)
    m=json.loads((ROOT/'configs/cohorts'/ds/'100000.json').read_text())
    modified=json.loads(json.dumps(m));modified['selected_identities'][0][3]=-1
    with pytest.raises(ValueError,match='identity'):
        PackagedSource(tmp_path,ds,modified)
    with (tmp_path/ds/'100000.parquet').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        PackagedSource(tmp_path,ds,m)


def test_each_scale_is_read_from_its_own_file(tmp_path):
    import shutil
    ds='extended_wikipedia'
    shutil.copytree(ROOT/'data/web'/ds,tmp_path/ds)
    (tmp_path/ds/'800000.parquet').unlink()
    for scale in [100000,200000,400000]:
        m=json.loads((ROOT/'configs/cohorts'/ds/f'{scale}.json').read_text())
        assert len(PackagedSource(tmp_path,ds,m).values)==m['selected_series_count']
    m=json.loads((ROOT/'configs/cohorts'/ds/'800000.json').read_text())
    with pytest.raises(FileNotFoundError):
        PackagedSource(tmp_path,ds,m)
