"""Exact batch and sampler parity with the pre-refactor DataLoader path."""

import copy
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import (
    BatchSampler, DataLoader, RandomSampler, SequentialSampler,
)
from torch.utils.data._utils.collate import default_collate

from src.common.webscale_ts import InMemorySeriesBank, PanelWindowDataset, WindowManifest
from src.data_provider.data_factory import DATASETS, IndexedDataset, data_provider
from src.data_provider.panel_data_provider import PanelBatchDataset
from src.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from src.models.RAFT import Model


def panel_bundle(channels=2, stride=1, label_len=0):
    # Non-contiguous series IDs, unequal lengths, and nonzero split origins.
    lengths = np.array([32, 35, 38], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    values = np.random.default_rng(43).normal(
        size=(int(offsets[-1]), channels)
    ).astype(np.float32)
    bank = InMemorySeriesBank(values, np.array([2, 8, 31]), offsets, lengths)
    manifests = {
        split: WindowManifest.from_bounds(
            bank, seq_len=6, pred_len=2, stride=stride,
            start_bound=start, end_bound=end,
        )
        for split, start, end in (
            ('train', 0, 19), ('valid', 13, 28), ('test', 22, lengths),
        )
    }
    return SimpleNamespace(datasets={
        split: PanelWindowDataset(
            bank, manifest, manifests['train'], label_len=label_len, split=split,
        )
        for split, manifest in manifests.items()
    })


def panel_args(bundle, batch_size=5, num_workers=0):
    return SimpleNamespace(
        data='panel', _panel_data_bundle=bundle,
        batch_size=batch_size, num_workers=num_workers,
    )


def assert_equal_batch(actual, expected):
    assert type(actual) is type(expected) is list
    assert len(actual) == len(expected) == 5
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype
        assert result.shape == reference.shape
        assert result.is_contiguous() == reference.is_contiguous()
        assert torch.equal(result, reference)


@pytest.mark.parametrize('split', ['train', 'valid', 'test'])
@pytest.mark.parametrize('channels,stride,label_len', [(1, 1, 0), (3, 2, 4)])
def test_arbitrary_batch_exactly_matches_collated_samples(split, channels, stride, label_len):
    dataset = panel_bundle(channels, stride, label_len).datasets[split]
    # Include repeats, both endpoints, and jumps across series boundaries.
    ids = [7, 2, 9, 1, len(dataset) - 1, 0, 7]
    expected = default_collate([dataset[index] for index in ids])
    actual = PanelBatchDataset(dataset)[ids]
    assert_equal_batch(actual, expected)
    assert actual[0].tolist() == ids
    assert actual[0].dtype == torch.int64
    assert all(field.dtype == torch.float32 for field in actual[1:])
    assert actual[1].shape == (len(ids), 6, channels)
    assert actual[2].shape == (len(ids), 2, channels)
    assert actual[3].shape == (len(ids), 6, 1)
    assert actual[4].shape == (len(ids), 2, 1)


@pytest.mark.parametrize('flag', ['train', 'val', 'valid', 'test'])
@pytest.mark.parametrize('batch_size', [1, 5, 100])
@pytest.mark.parametrize('num_workers', [0, 2])
def test_loader_content_order_boundaries_and_rng_match_over_epochs(flag, batch_size, num_workers):
    bundle = panel_bundle()
    args = panel_args(bundle, batch_size, num_workers)
    dataset = bundle.datasets['valid' if flag == 'val' else flag]

    def collect(reference):
        torch.manual_seed(812)
        if reference:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=flag != 'test',
                num_workers=num_workers, drop_last=False,
            )
        else:
            returned_dataset, loader = data_provider(args, flag)
            assert returned_dataset is dataset
            assert isinstance(loader.dataset, PanelBatchDataset)
            assert loader.batch_size is None
            assert isinstance(loader.sampler, BatchSampler)
            sampler_type = SequentialSampler if flag == 'test' else RandomSampler
            assert isinstance(loader.sampler.sampler, sampler_type)
            assert loader.sampler.batch_size == batch_size
            assert loader.sampler.drop_last is False
        epochs = []
        for _ in range(2):
            batches = list(loader)
            assert len(loader) == len(batches) == (len(dataset) + batch_size - 1) // batch_size
            indices = torch.cat([batch[0] for batch in batches]).tolist()
            assert sorted(indices) == list(range(len(dataset)))
            if flag == 'test':
                assert indices == list(range(len(dataset)))
            # Match global RNG state as well as outputs, including between epochs.
            epochs.append((batches, torch.random.get_rng_state(), torch.rand(3)))
        return epochs

    expected = collect(reference=True)
    actual = collect(reference=False)
    for (batches, state, draws), (references, old_state, old_draws) in zip(actual, expected):
        assert len(batches) == len(references)
        for batch, reference in zip(batches, references):
            assert_equal_batch(batch, reference)
        assert torch.equal(state, old_state)
        assert torch.equal(draws, old_draws)


@pytest.mark.parametrize('flag', ['train', 'val', 'test'])
def test_main_loader_uses_one_resolve_and_two_gathers_per_batch(flag):
    bundle = panel_bundle()
    dataset, loader = data_provider(panel_args(bundle), flag)
    bank = dataset.series_bank
    manifest = dataset.query_manifest
    with mock.patch.object(PanelWindowDataset, '__getitem__', side_effect=AssertionError('scalar access')), \
            mock.patch.object(bank, 'get_window', side_effect=AssertionError('scalar window')), \
            mock.patch.object(manifest, 'resolve_ids', wraps=manifest.resolve_ids) as resolve, \
            mock.patch.object(bank, 'get_windows', wraps=bank.get_windows) as gather:
        for batch in loader:
            assert resolve.call_count == 1
            assert gather.call_count == 2
            np.testing.assert_array_equal(resolve.call_args[0][0], batch[0].numpy())
            assert gather.call_args_list[0][0][2] == dataset.seq_len
            assert gather.call_args_list[1][0][2] == dataset.pred_len
            resolve.reset_mock()
            gather.reset_mock()


def test_production_window_shapes_with_small_bank():
    values = np.arange(744, dtype=np.float32).reshape(744, 1)
    bank = InMemorySeriesBank(values, [0], [0, 744], [744])
    manifest = WindowManifest.from_bounds(bank, 168, 1, end_bound=408)
    dataset = PanelWindowDataset(bank, manifest)
    bundle = SimpleNamespace(datasets={'test': dataset})
    _, loader = data_provider(panel_args(bundle, batch_size=32), 'test')
    batch = next(iter(loader))
    assert_equal_batch(batch, default_collate([dataset[i] for i in range(32)]))
    assert batch[1].shape == (32, 168, 1)
    assert batch[2].shape == (32, 1, 1)
    index, x, y = Exp_Long_Term_Forecast._unpack_batch(batch)
    assert index is batch[0] and x is batch[1] and y is batch[2]


@pytest.fixture(scope='module')
def legacy_csv(tmp_path_factory):
    root = tmp_path_factory.mktemp('legacy-batch-parity')
    # Enough rows for the unchanged ETT-minute split, shared by all cases.
    count = 57600
    pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=count, freq='h'),
        'first': np.sin(np.arange(count) / 5),
        'OT': np.cos(np.arange(count) / 7),
    }).to_csv(root / 'series.csv', index=False)
    return root


@pytest.mark.parametrize('name', ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom'])
@pytest.mark.parametrize('flag', ['train', 'val', 'test'])
def test_legacy_provider_keeps_datasets_and_automatic_collation(legacy_csv, name, flag):
    args = SimpleNamespace(
        data=name, root_path=str(legacy_csv), data_path='series.csv',
        seq_len=6, label_len=3, pred_len=2, features='M', target='OT',
        batch_size=5, num_workers=0,
    )
    with mock.patch('src.data_provider.panel_data_provider.panel_data_provider', side_effect=AssertionError('panel dispatch')), \
            mock.patch.object(PanelBatchDataset, '__init__', side_effect=AssertionError('panel loader')):
        dataset, loader = data_provider(args, flag)
        assert type(loader) is DataLoader
        assert type(dataset) is IndexedDataset
        assert type(dataset.dataset) is DATASETS[name]
        assert loader.dataset is dataset
        assert loader.batch_size == args.batch_size
        assert loader.num_workers == args.num_workers
        assert loader.drop_last is False
        assert loader.collate_fn is default_collate
        sampler_type = SequentialSampler if flag == 'test' else RandomSampler
        assert isinstance(loader.sampler, sampler_type)
        assert not hasattr(args, '_panel_data_bundle')
        torch.manual_seed(913)
        actual = next(iter(loader))
        torch.manual_seed(913)
        reference = next(iter(DataLoader(
            dataset, batch_size=5, shuffle=flag != 'test',
            num_workers=0, drop_last=False,
        )))
        assert_equal_batch(actual, reference)
        assert actual[2].shape == (5, 5, 2)  # Legacy Y retains label context.


def test_tiny_raft_adam_steps_are_bitwise_identical():
    bundle = panel_bundle(channels=1)
    dataset = bundle.datasets['train']
    config = SimpleNamespace(
        task_name='long_term_forecast', device=torch.device('cpu'),
        seq_len=6, pred_len=2, enc_in=1, n_period=1, topm=2,
        hnsw_m=8, hnsw_ef_construction=20, hnsw_ef_search=20,
        hnsw_seed=3, retrieval_batch_size=5, retrieval_controller='baseline',
    )
    torch.manual_seed(23)
    initial = Model(config)
    initial.rt.prepare_dataset(dataset)
    initial.retrieval_dict = {
        'train': initial.rt.retrieve_all(dataset, train=True, split='train'),
    }
    del initial.rt

    def train(reference):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        torch.manual_seed(62)
        if reference:
            loader = DataLoader(dataset, batch_size=5, shuffle=True, drop_last=False)
        else:
            _, loader = data_provider(panel_args(bundle), 'train')
        losses = []
        for batch in loader:
            index, x, y = Exp_Long_Term_Forecast._unpack_batch(batch)
            optimizer.zero_grad()
            output = model(x, index, mode='train')
            loss = torch.nn.functional.mse_loss(output, y)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        return torch.stack(losses), model.state_dict()

    expected_losses, expected_params = train(reference=True)
    actual_losses, actual_params = train(reference=False)
    assert torch.equal(actual_losses, expected_losses)
    assert actual_params.keys() == expected_params.keys()
    for name in actual_params:
        assert torch.equal(actual_params[name], expected_params[name]), name
