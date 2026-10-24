"""Historical Dataset and automatically collated DataLoader path."""

from torch.utils.data import DataLoader, Dataset

from src.data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
)


DATASETS = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


class IndexedDataset(Dataset):
    """Add stable indices used to address precomputed retrieval features."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (index, *self.dataset[index])

    def __getattr__(self, name):
        dataset = self.__dict__.get('dataset')
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)


def legacy_data_provider(args, flag):
    if args.data not in DATASETS:
        raise ValueError(
            f'unsupported dataset {args.data}; choose from {sorted(DATASETS)}'
        )
    dataset = DATASETS[args.data](
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
    )
    dataset = IndexedDataset(dataset)
    print(flag, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # Match RAFT/Time-Series-Library: validation is shuffled as well,
        # while test order remains stable.
        shuffle=flag != 'test',
        num_workers=args.num_workers,
        drop_last=False,
    )
    return dataset, loader
