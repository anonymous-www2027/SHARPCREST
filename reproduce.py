"""Run a configured experiment without historical campaign files."""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    # BLAS libraries read these variables at import time, before importing torch.
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--config', type=Path)
    early.add_argument('--cpu_threads', type=int)
    known, _ = early.parse_known_args()
    config = json.loads(known.config.read_text()) if known.config else {}
    threads = known.cpu_threads if known.cpu_threads is not None else config.get('cpu_threads', 4)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        early.error('cpu_threads must be a positive integer')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(threads)
    sys.dont_write_bytecode = True
    from src.reproduction.config import parse_args
    from src.reproduction.runner import execute
    execute(parse_args())


if __name__ == '__main__':
    main()
