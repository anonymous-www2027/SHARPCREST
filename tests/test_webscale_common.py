import numpy as np

from src.common.webscale_ts import (
    InMemorySeriesBank,
    PanelWindowDataset,
    SeriesSubset,
    WindowManifest,
)


class CountingSource:
    def __init__(self, series):
        self.series = [np.asarray(item, dtype=np.float32) for item in series]
        self.reads = []

    @property
    def num_series(self):
        return len(self.series)

    def get_length(self, series_id):
        return len(self.series[series_id])

    def get_n_channels(self, series_id):
        return self.series[series_id].shape[1]

    def get_series(self, series_id):
        self.reads.append(series_id)
        return self.series[series_id]


def test_subset_is_deterministic_and_round_trips(tmp_path):
    left = SeriesSubset.create(20, max_series=6, seed=17, dataset_identifier='x')
    right = SeriesSubset.create(20, max_series=6, seed=17, dataset_identifier='x')
    assert np.array_equal(left.selected_ids, right.selected_ids)
    path = left.save(tmp_path / 'subset.npz')
    loaded = SeriesSubset.load(path)
    assert loaded.summary() == left.summary()
    assert np.array_equal(loaded.selected_ids, left.selected_ids)
    assert path.with_suffix('.json').is_file()
    different = SeriesSubset.create(20, max_series=6, seed=18)
    assert not np.array_equal(left.selected_ids, different.selected_ids)
    assert SeriesSubset.create(20, fraction=0.25, seed=1).selected_count == 5
    assert SeriesSubset.create(20, max_series=7, seed=1).selected_count == 7


def test_bank_reads_each_selected_series_once_and_is_contiguous():
    source = CountingSource([
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.arange(20, dtype=np.float32).reshape(10, 2),
        np.arange(16, dtype=np.float32).reshape(8, 2),
    ])
    subset = SeriesSubset.create(3, max_series=2, seed=3)
    bank = InMemorySeriesBank.from_source(source, subset)
    assert source.reads == subset.selected_ids.tolist()
    assert bank.values.flags.c_contiguous
    assert bank.values.dtype == np.float32
    assert bank.offsets.tolist() == [0, *np.cumsum(bank.lengths).tolist()]
    selected_id = int(subset.selected_ids[0])
    assert np.array_equal(bank.get_series(selected_id), source.series[selected_id])
    assert np.array_equal(
        bank.get_window(selected_id, 1, 2), source.series[selected_id][1:3]
    )
    ids = np.full(2, selected_id)
    starts = np.array([0, 1])
    assert np.array_equal(
        bank.get_windows(ids, starts, 2),
        np.stack([source.series[selected_id][0:2], source.series[selected_id][1:3]]),
    )
    assert bank.nbytes == sum(
        item.nbytes for item in (
            bank.values, bank.series_ids, bank.offsets, bank.lengths
        )
    )
    assert source.reads == subset.selected_ids.tolist()


def test_manifest_mapping_is_o_series_and_invertible():
    values = np.arange(2 * 20, dtype=np.float32).reshape(20, 2)
    bank = InMemorySeriesBank(
        values, np.array([3, 8]), np.array([0, 9, 20]), np.array([9, 11])
    )
    manifest = WindowManifest.from_bounds(
        bank, seq_len=3, pred_len=2, stride=2
    )
    ids = np.arange(len(manifest), dtype=np.int64)
    series, starts = manifest.resolve_ids(ids)
    assert np.array_equal(manifest.encode(series, starts), ids)
    assert manifest.resolve_ids(np.int64(0)) == (3, 0)
    assert manifest.resolve_ids(np.int64(len(manifest) - 1)) == (8, 6)
    assert series.dtype == np.int64 and starts.dtype == np.int64
    assert manifest.nbytes < len(manifest) * 16


def test_panel_dataset_produces_five_fields_on_demand():
    series = np.arange(40, dtype=np.float32).reshape(20, 2)
    bank = InMemorySeriesBank(
        series, np.array([0]), np.array([0, 20]), np.array([20])
    )
    manifest = WindowManifest.from_bounds(
        bank, seq_len=4, pred_len=3, stride=2
    )
    dataset = PanelWindowDataset(bank, manifest, label_len=2, split='train')
    index, x, y, x_mark, y_mark = dataset[2]
    assert index == 2
    assert np.shares_memory(x, bank.values)
    assert np.shares_memory(y, bank.values)
    assert np.array_equal(x, series[4:8])
    assert np.array_equal(y, series[8:11])
    assert x_mark.shape == (4, 1)
    assert y_mark.shape == (3, 1)
