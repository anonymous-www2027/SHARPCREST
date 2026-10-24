import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class ForecastDataset(Dataset):
    """Local CSV loader for the fixed-window datasets used by RAFT/SHARP."""

    split_borders = None

    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='S', target='OT', scale=True):
        if flag not in {'train', 'val', 'test'}:
            raise ValueError(f'unsupported split: {flag}')
        if size is None:
            size = [96, 48, 96]

        self.seq_len, self.label_len, self.pred_len = size
        if self.label_len > self.seq_len:
            raise ValueError(
                f'label_len ({self.label_len}) cannot exceed seq_len '
                f'({self.seq_len})'
            )
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.scaler = StandardScaler()
        self._read_data()

    def _get_borders(self, data_length):
        if self.split_borders is None:
            train_size = int(data_length * 0.7)
            test_size = int(data_length * 0.2)
            val_size = data_length - train_size - test_size
            border1s = [0, train_size - self.seq_len,
                        train_size + val_size - self.seq_len]
            border2s = [train_size, train_size + val_size, data_length]
        else:
            train_end, val_end, test_end = self.split_borders
            if data_length < test_end:
                raise ValueError(
                    f'{self.data_path} contains {data_length} rows; '
                    f'at least {test_end} are required for the ETT split'
                )
            border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
            border2s = [train_end, val_end, test_end]
        return border1s, border2s

    def _read_data(self):
        path = os.path.join(self.root_path, self.data_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'dataset not found: {path}. SHARP expects a local CSV file.'
            )
        frame = pd.read_csv(path)
        if 'date' not in frame.columns:
            raise ValueError(f'{path} must contain a date column')

        border1s, border2s = self._get_borders(len(frame))
        split_index = {'train': 0, 'val': 1, 'test': 2}[self.flag]
        border1 = border1s[split_index]
        border2 = border2s[split_index]

        if self.features in {'M', 'MS'}:
            values = frame.drop(columns=['date'])
        elif self.features == 'S':
            values = frame[[self.target]]
        else:
            raise ValueError('features must be one of M, MS, or S')

        if self.scale:
            self.scaler.fit(values.iloc[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(values.values)
        else:
            data = values.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # RAFT does not consume timestamp embeddings. A one-column placeholder
        # preserves the standard forecasting batch structure without retaining
        # TSLib's embedding stack.
        self.data_stamp = np.zeros((border2 - border1, 1), dtype=np.float32)

    def __getitem__(self, index):
        x_start = index
        x_end = x_start + self.seq_len
        y_start = x_end - self.label_len
        y_end = y_start + self.label_len + self.pred_len
        return (
            self.data_x[x_start:x_end],
            self.data_y[y_start:y_end],
            self.data_stamp[x_start:x_end],
            self.data_stamp[y_start:y_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(ForecastDataset):
    split_borders = (12 * 30 * 24,
                     12 * 30 * 24 + 4 * 30 * 24,
                     12 * 30 * 24 + 8 * 30 * 24)


class Dataset_ETT_minute(ForecastDataset):
    split_borders = (12 * 30 * 24 * 4,
                     12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                     12 * 30 * 24 * 4 + 8 * 30 * 24 * 4)


class Dataset_Custom(ForecastDataset):
    pass
