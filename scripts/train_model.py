"""Bootstrap optional RQ1 models with isolated imports and deterministic runtime."""
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--framework' in sys.argv and sys.argv[sys.argv.index('--framework')+1] == 'knn_mts':
    os.environ['MKL_THREADING_LAYER'] = 'INTEL'
    import numpy as np
    _probe = np.ones((4, 4), dtype=np.float32)
    _probe @ _probe
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reproduction.models import main

if __name__ == '__main__':
    main()
