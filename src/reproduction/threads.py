"""Temporarily set CPU numerical threads for complete retrieval preparation."""
from contextlib import contextmanager

import torch
from threadpoolctl import threadpool_info, threadpool_limits


def thread_snapshot():
    return dict(torch_threads=torch.get_num_threads(), pools=threadpool_info())


@contextmanager
def retrieval_threads(count):
    """Restore both PyTorch and BLAS/OpenMP limits, including on failure."""
    previous = torch.get_num_threads()
    with threadpool_limits(limits=count):
        try:
            torch.set_num_threads(count)
            yield thread_snapshot()
        finally:
            torch.set_num_threads(previous)
