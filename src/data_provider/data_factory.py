"""Dispatch general time series and web-scale panels to separate loaders."""

# Retain these imports for callers of the original factory module.
from src.data_provider.legacy_data_provider import (
    DATASETS,
    IndexedDataset,
    legacy_data_provider,
)


def data_provider(args, flag):
    if args.data == 'panel':
        from src.data_provider.panel_data_provider import panel_data_provider

        return panel_data_provider(args, flag)
    return legacy_data_provider(args, flag)
