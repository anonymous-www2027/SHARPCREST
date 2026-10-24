"""Small best-validation policy and scalar-pooled main evaluator."""
import json
from pathlib import Path
import numpy as np
import torch
from src.utils.webscale_metrics import WebscaleMetrics


class BestValidation:
    def __init__(self, patience=0, min_delta=0.):
        self.patience,self.min_delta=patience,min_delta
        self.best=float('inf');self.bad_epochs=0;self.best_epoch=None
    def update(self, loss, epoch):
        if not np.isfinite(loss):raise ValueError('nonfinite validation MSE')
        improved=loss < self.best-self.min_delta
        if improved:self.best=loss;self.best_epoch=epoch;self.bad_epochs=0
        else:self.bad_epochs+=1
        return improved,bool(self.patience and self.bad_epochs>=self.patience)


def evaluate_main(experiment, dataset, loader, setting):
    model=experiment.model;model.eval();panel=getattr(dataset,'forecast_scaler',None)
    evaluator=WebscaleMetrics() if panel is not None else None
    square=absolute=0.;count=0
    with torch.no_grad():
        for batch in loader:
            index,x,y=experiment._unpack_batch(batch)
            pred=model(x.float().to(experiment.device),index,mode='test').cpu().numpy()
            target=y[:,-experiment.args.pred_len:].numpy()
            if evaluator is not None:
                sids,starts=dataset.query_manifest.resolve_ids(index.numpy())
                raw=dataset.series_bank.get_windows(sids,starts+dataset.seq_len,dataset.pred_len)
                evaluator.update(pred,target,raw,sids,panel)
            else:
                error=pred.astype(np.float64)-target.astype(np.float64)
                if not np.isfinite(error).all():raise ValueError('nonfinite test error')
                square+=float((error**2).sum(dtype=np.float64));absolute+=float(np.abs(error).sum(dtype=np.float64));count+=error.size
    result=evaluator.result() if evaluator is not None else dict(MSE=square/count,MAE=absolute/count)
    destination=Path('results')/setting;destination.mkdir(parents=True,exist_ok=True)
    (destination/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
    return (result['normalized_MSE'],result['normalized_MAE']) if evaluator else (result['MSE'],result['MAE'])
