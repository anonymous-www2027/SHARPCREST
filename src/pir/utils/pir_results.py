"""Persist query identities, predictions, fixed-anchor metrics and provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def errors(pred, true):
    diff = pred.astype(np.float64)-true.astype(np.float64)
    if not diff.size or not np.isfinite(diff).all():
        raise ValueError('empty/nonfinite evaluation output')
    return {'MSE': float(np.mean(diff**2)), 'MAE': float(np.mean(np.abs(diff))),
            'scalar_count': int(diff.size)}


def save_results(args, model, dataset, pred, true, controllers):
    from src.pir.data_provider.pir_panel import query_metadata
    root = Path(args.result_dir)
    result = errors(pred, true)
    result.update(normalized_MSE=result['MSE'], normalized_MAE=result['MAE'],
                  full_test_MSE=result['MSE'], full_test_MAE=result['MAE'])
    metadata = query_metadata(dataset, np.arange(len(dataset)))
    if len(pred) != len(metadata):
        raise ValueError('prediction/query count mismatch')
    np.save(root/'pred.npy', pred)
    np.save(root/'true.npy', true)
    np.savez(root/'test_queries.npz', series_id=metadata[:, 0], past_start=metadata[:, 1],
             forecast_origin=metadata[:, 1]+args.seq_len)
    if hasattr(dataset, 'bundle'):
        mask = dataset.fixed_mask
        fixed = errors(pred[mask], true[mask])
        result.update(fixed_test_MSE=fixed['MSE'], fixed_test_MAE=fixed['MAE'],
                      fixed_scalar_count=fixed['scalar_count'], fixed_test_queries=int(mask.sum()))
        np.save(root/'fixed_test_mask.npy', mask)
        np.save(root/'raw_pred.npy', dataset.inverse_predictions(pred))
        np.save(root/'raw_true.npy', dataset.inverse_predictions(true))
        bundle = dataset.bundle
        np.savez(root/'train_scaler.npz', series_id=bundle.ids,
                 mean=np.stack([bundle.mean[int(s)] for s in bundle.ids]),
                 scale=np.stack([bundle.scale[int(s)] for s in bundle.ids]))
        write_json(root/'panel_manifest.json', bundle.audit())
    else:
        np.savez(root/'train_scaler.npz', mean=dataset.scaler.mean_, scale=dataset.scaler.scale_)
    if controllers:
        controller = np.concatenate(controllers)
        np.save(root/'test_reliability.npy', controller)
        result['test_reliability'] = dict(zip(('cr_q', 'lambda_q', 'sigma_q'), controller.mean((0, 1)).astype(float).tolist()))
        result['reliability_layout'] = '[test_query,channel,(cr_q,lambda_q,sigma_q)]'
        result['sigma_used_for_aggregation'] = False
    write_json(root/'metrics.json', result)
    write_json(root/'retrieval_timing.json', model.retrieval_adapter.report())


def file_hashes(root):
    hashes = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and path.name not in ('completion.json', 'status.json'):
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    h.update(chunk)
            hashes[str(path.relative_to(root))] = h.hexdigest()
    return hashes
