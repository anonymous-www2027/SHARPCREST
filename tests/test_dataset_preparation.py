"""Data conversion, leakage guards, nested selection and runner interoperability."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from src.preprocessing.io import file_hash
from src.preprocessing.io import exclusive_output
from src.preprocessing.protocol import protocol, regimes
from src.preprocessing.schema import CanonicalWriter
from src.preprocessing.selection import extend, select_scales, scale_chain
from src.preprocessing.population import Population, analyze, calculate_batch, validate_table
from src.preprocessing.prepare import prepare, load_table
from src.preprocessing.quito import convert_quito, _regularize_group


def small_protocol():
    return dict(L=16, H=8, period=7, anchor_series=2,
        split=dict(expected_length=96, train_end=64, valid_end=80, test_end=96,
                   protocol='synthetic test'))


def write_panel(root, values, name='extended_wikipedia', source_ids=None):
    directory = root / name
    with CanonicalWriter(directory, dataset_name=name, source_files=[], frequency_seconds=86400,
                         channel_names=['page_views'], source_format='test', source_value_dtypes=['float32'],
                         target_values_per_part=300, row_group_max_rows=2) as writer:
        for i, x in enumerate(values):
            writer.add_series(series_id=i, source_id=str(i if source_ids is None else source_ids[i]),
                start_timestamp=np.datetime64('2020-01-01'), frequency_seconds=86400, values=x)
    return directory


@pytest.fixture
def panel(tmp_path):
    root = tmp_path / 'input'
    values = [np.random.default_rng(i).normal(size=96).astype(np.float32) for i in range(6)]
    write_panel(root, values)
    return root, values


def test_quito_conversion_sorts_and_regularizes(tmp_path):
    if pa.__version__ == '19.0.0':
        pytest.skip('Raw Quito conversion intentionally rejects Arrow 19.0.0')
    raw = tmp_path / 'raw.parquet'
    pq.write_table(pa.table(dict(item_id=[9, 3, 9, 3],
        date_time=pa.array(np.array(['2020-01-01T02','2020-01-01T01','2020-01-01T00','2020-01-01T00'], dtype='datetime64[ns]')),
        ind_2=[92.,31.,90.,30.], ind_1=[9.2,3.1,9.,3.])), raw)
    out = tmp_path / 'quitobench_hour'
    metadata = convert_quito(source=raw, dataset_name='quitobench_hour', output_dir=out,
                             max_partition_rows=2)
    assert metadata['channel_names'] == ['ind_1', 'ind_2']
    assert metadata['source_sha256'] == file_hash(raw)
    assert metadata['inserted_missing_time_steps'] == 1
    table = pq.read_table(out / 'part-00000.parquet').to_pandas()
    assert table.source_id.tolist() == ['3', '9']
    np.testing.assert_array_equal(table.iloc[0]['values'], np.array([3.,30.,3.1,31.], dtype=np.float32))
    np.testing.assert_array_equal(table.iloc[1]['values'], np.array([9.,90.,np.nan,np.nan,9.2,92.], dtype=np.float32))
    for name, expected in json.loads((out / 'checksums.json').read_text()).items():
        assert file_hash(out / name) == expected
    with pytest.raises(FileExistsError):
        convert_quito(source=raw, dataset_name='quitobench_hour', output_dir=out)


@pytest.mark.parametrize('hours', [[0,0], [0,0.5]])
def test_quito_rejects_duplicate_or_off_grid_timestamps(hours):
    timestamps = np.datetime64('2020-01-01','ns') + np.asarray(np.array(hours)*3600e9, dtype='timedelta64[ns]')
    table = pa.table(dict(date_time=timestamps, ind_1=[1.,2.]))
    with pytest.raises(ValueError):
        _regularize_group(table, 0, 2, ['ind_1'], 3600)


def test_nested_quota_selection_and_independent_scale_invocation():
    p = protocol('extended_wikipedia'); p.update(anchor_series=10, L=10, H=1)
    p['split']['train_end'] = 10010
    frame = pd.DataFrame(dict(logical_id=np.arange(100), regime=['LOW_LOW_LOW']*60+['HIGH_HIGH_HIGH']*40))
    together = select_scales(frame, [100000,200000,400000,800000], p)
    alone = select_scales(frame, [800000], p)
    assert alone == together
    previous = []
    for scale, row in together.items():
        assert row['ids'][:len(previous)] == previous
        assert len(row['ids']) == scale//10000
        labels = frame.set_index('logical_id').loc[row['ids']].regime.value_counts()
        for quota in row['quotas']:
            assert labels.get(quota['regime'], 0) == quota['integer_quota']
        previous = row['ids']
    assert scale_chain([400000]) == [100000,200000,400000]


def test_no_silent_quota_violation_or_population_truncation():
    p = protocol('extended_wikipedia')
    frame = pd.DataFrame(dict(logical_id=[0,1,2,3,4], regime=['LOW_LOW_LOW']+['HIGH_HIGH_HIGH']*4))
    with pytest.raises(ValueError, match='infeasible'):
        extend(frame, [0], 2, p)
    with pytest.raises(ValueError, match='exceeds'):
        extend(frame, [], 6, p)


def test_train_only_tsf_and_resume(panel, tmp_path):
    pytest.importorskip('statsmodels')
    root, values = panel
    p = protocol('extended_wikipedia', small_protocol())
    pop = Population(root, p)
    rows = list(pop.train_rows())
    assert all(len(x) == 64 for _, x in rows)
    altered = [x.copy() for x in values]
    for x in altered:
        x[64:] = 1e20
    second = tmp_path / 'changed'
    write_panel(second, altered)
    changed = Population(second, p)
    a = analyze(pop, tmp_path / 'cache', batch_size=2)
    b = calculate_batch((list(changed.train_rows()), p))
    pd.testing.assert_frame_equal(a, b)
    cached = analyze(pop, tmp_path / 'cache', batch_size=2)
    pd.testing.assert_frame_equal(a, cached)
    with pytest.raises(ValueError, match='changed'):
        analyze(changed, tmp_path / 'cache', batch_size=2)
    with (tmp_path / 'cache/population_tsf.parquet').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        analyze(pop, tmp_path / 'cache', batch_size=2)


def test_missing_preserving_wiki_is_checked(tmp_path):
    raw = np.arange(96, dtype=np.float32); raw[10] = np.nan
    write_panel(tmp_path/'canonical', [np.nan_to_num(raw)])
    write_panel(tmp_path/'missing', [raw])
    p = protocol('extended_wikipedia', small_protocol())
    a, b = Population(tmp_path/'canonical', p), Population(tmp_path/'missing', p)
    assert np.isnan(next(a.train_rows(b))[1][10])
    different = raw.copy(); different[5] = 999
    write_panel(tmp_path/'different', [different])
    with pytest.raises(ValueError, match='does not map'):
        list(a.train_rows(Population(tmp_path/'different', p)))


def test_invalid_table_labels_and_coverage(panel):
    root, _ = panel; p = protocol('extended_wikipedia', small_protocol()); pop = Population(root, p)
    good = pd.DataFrame(dict(logical_id=np.arange(6), T=.1, S=.1, F=.1, regime='LOW_LOW_LOW'))
    validate_table(good, pop)
    with pytest.raises(ValueError, match='complete'):
        validate_table(good.iloc[:5], pop)
    bad = good.copy(); bad.loc[0,'regime'] = 'HIGH_HIGH_HIGH'
    with pytest.raises(ValueError, match='labels'):
        validate_table(bad, pop)


def test_generated_cohort_consumable_by_public_runner(panel, tmp_path):
    pytest.importorskip('statsmodels')
    root, _ = panel; out = tmp_path / 'prepared'
    result = prepare(root, 'extended_wikipedia', [100000], out, overrides=small_protocol(), batch_size=2)
    path = out / result['cohorts'][0]['cohort']
    m = json.loads(path.read_text())
    assert result['status'] == 'PASS' and not result['cohorts'][0]['published_match']
    from src.reproduction.data import load_cohort, build_bundle
    args = SimpleNamespace(dataset='extended_wikipedia', cohort_file=path, scale=100000,
                           canonical_root=str(root), seq_len=16, pred_len=8)
    loaded, _ = load_cohort(args)
    bundle, fixed, audit = build_bundle(args, loaded)
    assert audit['actual_candidate_count'] == 2*(64-16-8+1)
    assert audit['fixed_test_queries'] == 2*(96-80-8+1)
    assert bundle.series_bank.n_channels == 1
    assert set(fixed['source_id']) == set(m['anchor_source_ids'])
    assert prepare(root, 'extended_wikipedia', [100000], out, overrides=small_protocol(), batch_size=2) == result
    source = out/'population/population_tsf.parquet'
    again = prepare(root, 'extended_wikipedia', [100000], tmp_path/'reuse', overrides=small_protocol(), tsf_table=source)
    reused = json.loads((tmp_path/'reuse'/again['cohorts'][0]['cohort']).read_text())
    assert reused['ordered_source_ids'] == m['ordered_source_ids']
    assert reused['selected_raw_bank_hash'] == m['selected_raw_bank_hash']


def test_unsealed_table_rejected(panel, tmp_path):
    root, _ = panel
    path = tmp_path / 'foreign.parquet'
    pd.DataFrame(dict(logical_id=[0], T=[0.], S=[0.], F=[0.], regime=['LOW_LOW_LOW'])).to_parquet(path)
    with pytest.raises(ValueError, match='Unrecognized'):
        load_table(path, Population(root, protocol('extended_wikipedia', small_protocol())))


def test_source_index_mismatch(panel):
    root, _ = panel
    index_path = root/'extended_wikipedia/series_index.parquet'
    index = pd.read_parquet(index_path); index.loc[0, 'source_id'] = 'not-the-source-id'; index.to_parquet(index_path)
    pop = Population(root, protocol('extended_wikipedia', small_protocol()))
    with pytest.raises(ValueError, match='identity mismatch'):
        list(pop.train_rows())


def test_parallel_tsf_matches_serial(panel, tmp_path):
    pytest.importorskip('statsmodels')
    root, _ = panel; pop = Population(root, protocol('extended_wikipedia', small_protocol()))
    a = analyze(pop, tmp_path/'serial', workers=1, batch_size=2)
    b = analyze(pop, tmp_path/'parallel', workers=2, batch_size=2)
    pd.testing.assert_frame_equal(a, b)


def test_interrupted_population_resumes_only_missing_batches(panel, tmp_path, monkeypatch):
    pytest.importorskip('statsmodels')
    import src.preprocessing.population as module
    root, _ = panel; pop = Population(root, protocol('extended_wikipedia', small_protocol()))
    calls = []
    def interrupted(job):
        calls.append(job)
        if len(calls) == 3:
            raise RuntimeError('simulated interruption')
        return calculate_batch(job)
    monkeypatch.setattr(module, 'calculate_batch', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        analyze(pop, tmp_path/'cache', batch_size=2)
    assert (tmp_path/'cache/chunks/0000000.json').exists()
    calls.clear()
    def resumed(job):
        calls.append(job)
        return calculate_batch(job)
    monkeypatch.setattr(module, 'calculate_batch', resumed)
    recovered = analyze(pop, tmp_path/'cache', batch_size=2)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(recovered, calculate_batch((list(pop.train_rows()), pop.protocol)))


def test_concurrent_output_and_invalid_paper_wiki_input(panel, tmp_path):
    root, _ = panel; out = tmp_path/'locked'
    with exclusive_output(out):
        with pytest.raises(RuntimeError, match='Another process'):
            prepare(root, 'extended_wikipedia', [100000], out, overrides=small_protocol())
    with pytest.raises(ValueError, match='Paper Wiki TSF requires'):
        prepare(root, 'extended_wikipedia', [100000], tmp_path/'paper', require_published=True)


def test_raw_checksum_rejects_different_quito_input(tmp_path):
    if pa.__version__ == '19.0.0':
        pytest.skip('Raw conversion rejects Arrow 19.0.0')
    raw = tmp_path/'raw.parquet'; raw.write_bytes(b'wrong version')
    with pytest.raises(ValueError, match='checksum'):
        convert_quito(source=raw, dataset_name='quitobench_hour', output_dir=tmp_path/'out',
                      expected_source_sha256='0'*64)
    assert not (tmp_path/'out').exists()


def test_custom_cli_auto_anchor_nested_and_verified_table_reuse(tmp_path):
    pytest.importorskip('statsmodels')
    from src.preprocessing.__main__ import main
    root, out = tmp_path/'canonical', tmp_path/'prepared'
    values = [np.random.default_rng(i).normal(size=25064).astype(np.float32) for i in range(9)]
    write_panel(root, values, name='custom_sensor')
    overrides = tmp_path/'protocol.json'
    overrides.write_text(json.dumps(dict(L=20, H=10, period=7)))
    main(['prepare', '--canonical-root', str(root), '--dataset', 'custom_sensor',
          '--protocol', str(overrides), '--L', '16', '--H', '8', '--train-end', '25024',
          '--valid-end', '25044', '--test-end', '25064', '--scales', '100000', '200000',
          '--output', str(out), '--batch-size', '3'])
    request = json.loads((out/'request.json').read_text())
    p = request['protocol']
    assert (p['L'], p['H'], p['period'], p['anchor_series'], p['selection_seed']) == (16, 8, 7, None, 0)
    first = json.loads((out/'cohorts/custom_sensor/100000.json').read_text())
    second = json.loads((out/'cohorts/custom_sensor/200000.json').read_text())
    assert first['selected_series_count'] == 4 and first['actual_candidate_count'] == 100004
    assert second['selected_series_count'] == 8 and second['actual_candidate_count'] == 200008
    assert second['ordered_source_ids'][:4] == first['ordered_source_ids'] == second['anchor_source_ids']
    frame = pd.read_parquet(out/'population/population_tsf.parquet')
    assert len(frame) == 9
    assert frame.finite_count.eq(25024).all()
    for scale in (100000, 200000):
        rows = pd.read_csv(out/f'cohorts/custom_sensor/{scale}_quotas.csv')
        assert rows.existing_count.add(rows.added_count).equals(rows.integer_quota)
    result = prepare(root, 'custom_sensor', [200000], tmp_path/'reused', overrides=p,
                     tsf_table=out/'population/population_tsf.parquet')
    reused = json.loads((tmp_path/'reused'/result['cohorts'][-1]['cohort']).read_text())
    assert reused['ordered_source_ids'] == second['ordered_source_ids']
    assert reused['selected_raw_bank_hash'] == second['selected_raw_bank_hash']


def test_custom_protocol_required_settings_and_channel(tmp_path):
    with pytest.raises(ValueError, match='explicit settings.*L.*H.*period.*train_end'):
        protocol('new_panel')
    with pytest.raises(ValueError, match='directory name'):
        protocol('../escape', small_protocol())
    p = protocol('sensors', dict(small_protocol(), channel=1))
    with CanonicalWriter(tmp_path/'sensors', dataset_name='sensors', source_files=[],
                         frequency_seconds=3600, channel_names=['first', 'second'],
                         source_format='test', source_value_dtypes=['float32']) as writer:
        x = np.column_stack([np.zeros(96), np.arange(96)]).astype(np.float32)
        writer.add_series(series_id=0, source_id='sensor-a', start_timestamp=np.datetime64('2020-01-01'),
                          frequency_seconds=3600, values=x)
    pop = Population(tmp_path, p)
    pop.validate()
    np.testing.assert_array_equal(next(pop.train_rows())[1], np.arange(64))
    with pytest.raises(ValueError, match='No published cohort'):
        prepare(tmp_path, 'sensors', [100000], tmp_path/'bad', overrides=p, require_published=True)


@pytest.mark.parametrize('corruption', ['schema', 'pointer', 'values'])
def test_complete_validation_before_first_tsf_batch(panel, tmp_path, monkeypatch, corruption):
    import src.preprocessing.population as module
    root, _ = panel
    index_path = root/'extended_wikipedia/series_index.parquet'
    idx = pd.read_parquet(index_path)
    row = idx.iloc[-1]
    part_path = root/'extended_wikipedia'/row.part_file
    if corruption == 'schema':
        table = pq.read_table(part_path).drop(['frequency_seconds'])
        pq.write_table(table, part_path)
    elif corruption == 'pointer':
        idx.loc[idx.index[-1], 'row_in_group'] = 999
        idx.to_parquet(index_path, index=False)
    else:
        table = pq.read_table(part_path)
        rows = table.to_pylist()
        rows[-1]['values'] = rows[-1]['values'][:-1]
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), part_path)
    def should_not_run(*args, **kwargs):
        pytest.fail('TSF started before full canonical validation')
    monkeypatch.setattr(module, 'calculate_batch', should_not_run)
    with pytest.raises(ValueError):
        prepare(root, 'extended_wikipedia', [100000], tmp_path/'bad',
                overrides=small_protocol(), batch_size=1)
    assert not (tmp_path/'bad/population').exists()
