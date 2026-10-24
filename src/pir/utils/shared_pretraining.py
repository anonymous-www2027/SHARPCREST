"""Share the best pretrain model and its RNG boundary. Refinement and retrieval remain method-specific."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import torch


SCHEMA = 1
IGNORED_ARGUMENTS = {
    'result_dir', 'checkpoints', 'bakcbone_checkpoints', 'shared_pretraining_cache',
    'retrieval_method', 'retrieval_execution', 'retrieval_batch_size',
    'hnsw_threads', 'hnsw_build_threads', 'hnsw_m', 'hnsw_ef_construction',
    'hnsw_ef_search', 'sharp_c', 'crest_pool', 'reliability_r', 'profile_sections',
    'stp_weights', 'preflight_only',
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    """Hash nested states without pickle's storage-identity dependent encoding."""
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().contiguous().numpy()
        if isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                visit(key); visit(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(str((type(x).__name__, len(x))).encode())
            for item in x:
                visit(item)
        else:
            h.update((type(x).__name__ + ':' + repr(x) + '\n').encode())
    visit(value)
    return h.hexdigest()


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise RuntimeError('SHARED_PRETRAIN_CUDA_ENVIRONMENT_CHANGED')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


class SharedPretraining:
    def __init__(self, args, model, setting):
        self.args, self.model, self.setting = args, model, setting
        self.enabled = bool(getattr(args, 'shared_pretraining_cache', ''))
        self.hit = False
        self.lock = None

    def __enter__(self):
        if not self.enabled:
            return self
        if self.args.load_pretrained_backbone != 0:
            raise ValueError('shared pretraining requires native pretraining mode 0')
        self.started = time.perf_counter()
        self.output = Path(self.args.result_dir)
        read = lambda name: json.loads((self.output / name).read_text())
        data = read('data_audit.json')
        # Pool-dependent eligibility fields describe retrieval, not pretraining.
        data = {split: {k: data[split][k] for k in
                       ('count', 'series_id_hash', 'past_start_hash')}
                for split in ('train', 'retrieval', 'val', 'test')}
        if (self.output / 'panel_manifest.json').exists():
            data['panel'] = read('panel_manifest.json')
        else:
            data['source_csv_sha256'] = read('data_audit.json')['source_csv_sha256']
        config = {k: v for k, v in vars(self.args).items()
                  if not k.startswith('_') and k not in IGNORED_ARGUMENTS}
        device = next(self.model.parameters()).device
        self.identity = {
            'schema': SCHEMA, 'arguments': config, 'data': data,
            'source_hashes': read('source_hashes.json'),
            'initial_model_hash': digest(self.model.state_dict()),
            'entry_rng_hash': digest(capture_rng()),
            'environment': {
                'torch': str(torch.__version__), 'numpy': np.__version__,
                'cuda': torch.version.cuda, 'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'torch_threads': torch.get_num_threads(),
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'thread_environment': {k: os.environ.get(k) for k in
                                       ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                        'CUBLAS_WORKSPACE_CONFIG')},
            },
        }
        encoded = json.dumps(self.identity, sort_keys=True, allow_nan=False).encode()
        self.key = hashlib.sha256(encoded).hexdigest()
        root = Path(self.args.shared_pretraining_cache).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.entry = root / self.key
        self.lock = (root / (self.key + '.lock')).open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX)
        except BaseException:
            self.lock.close()
            raise
        self.wait_finished = time.perf_counter()
        return self

    def restore(self):
        if not self.enabled or not self.entry.exists():
            return False
        receipt = json.loads((self.entry / 'receipt.json').read_text())
        if receipt['key'] != self.key or receipt['identity'] != self.identity:
            raise RuntimeError('SHARED_PRETRAIN_IDENTITY_MISMATCH')
        checkpoint = self.entry / 'boundary.pth'
        if file_hash(checkpoint) != receipt['checkpoint_sha256']:
            raise RuntimeError('SHARED_PRETRAIN_CHECKPOINT_HASH_MISMATCH')
        payload = torch.load(checkpoint, map_location='cpu')
        if payload['schema'] != SCHEMA or digest(payload['model']) != receipt['model_hash']:
            raise RuntimeError('SHARED_PRETRAIN_MODEL_HASH_MISMATCH')
        if digest(payload['rng']) != receipt['boundary_rng_hash']:
            raise RuntimeError('SHARED_PRETRAIN_RNG_HASH_MISMATCH')
        self.model.load_state_dict(payload['model'], strict=True)
        modules = dict(self.model.named_modules())
        for name, training in payload['module_training'].items():
            modules[name].training = training
        # Retain the same backbone checkpoint artifact as an independent run.
        destination = Path(self.args.bakcbone_checkpoints) / self.setting
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(payload['model'], destination / 'checkpoint.pth')
        self.hit = True
        self._record(receipt, 'reused')
        restore_rng(payload['rng'])
        print('SHARED_PRETRAIN_HIT', self.key, flush=True)
        return True

    def publish(self):
        if not self.enabled or self.hit:
            return
        # Best weights have already been reloaded, but RNG is the END of the
        # pretraining loop (including validation/test), not the best epoch's RNG.
        rng = capture_rng()
        state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        payload = {'schema': SCHEMA, 'model': state, 'rng': rng,
                   'module_training': {k: v.training for k, v in self.model.named_modules()}}
        temporary = Path(tempfile.mkdtemp(prefix=self.key + '.partial-', dir=self.entry.parent))
        try:
            checkpoint = temporary / 'boundary.pth'
            torch.save(payload, checkpoint)
            history = self.output / 'training_history.jsonl'
            epochs = [json.loads(line) for line in history.read_text().splitlines()
                      if json.loads(line)['stage'] == 'pretrain'] if history.exists() else []
            receipt = {'key': self.key, 'identity': self.identity,
                       'producer_result_dir': str(self.output),
                       'producer_method': self.args.retrieval_method,
                       'pretrain_epochs': epochs,
                       'checkpoint_sha256': file_hash(checkpoint),
                       'model_hash': digest(state), 'boundary_rng_hash': digest(rng)}
            (temporary / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            temporary.rename(self.entry)
            self._record(receipt, 'created')
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            restore_rng(rng)
        print('SHARED_PRETRAIN_CREATED', self.key, flush=True)

    def _record(self, receipt, action):
        from src.pir.utils.pir_results import write_json
        write_json(self.output / 'shared_pretraining.json', {
            'action': action, 'cache_key': self.key, 'cache_entry': str(self.entry),
            'producer_result_dir': receipt['producer_result_dir'],
            'checkpoint_sha256': receipt['checkpoint_sha256'],
            'boundary_model_hash': receipt['model_hash'],
            'boundary_rng_hash': receipt['boundary_rng_hash'],
            'pretrain_epochs_completed_by_producer': len(receipt['pretrain_epochs']),
            'pretrain_epochs_executed_here': len(receipt['pretrain_epochs']) if action == 'created' else 0,
            'lookup_and_lock_seconds': self.wait_finished - self.started,
            'shared_pretrain_stage_wall_seconds': time.perf_counter() - self.started,
            'timing_scope': 'outside retrieval timers; do not add to retrieval_prepare_total_seconds',
            'boundary': 'best full model loaded; end-of-pretrain RNG; before retrieval preparation',
        })
        write_json(self.output / 'shared_pretraining_history.json', {
            'executed_here': action == 'created',
            'producer_result_dir': receipt['producer_result_dir'],
            'epochs': receipt['pretrain_epochs'],
        })

    def __exit__(self, *exc):
        if self.lock is not None:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
