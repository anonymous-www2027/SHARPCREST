"""Load the repository's unchanged DLinear, without importing legacy optional models."""
import importlib.util
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().with_name('dlinear.py')


def make_model(length, horizon, channels, individual=False):
    spec = importlib.util.spec_from_file_location('_original_dlinear', MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DLinear(seq_len=length, pred_len=horizon, enc_in=channels, individual=individual)


def forecast(model, past):
    return model(history_data=past.unsqueeze(-1), future_data=None,
                 batch_seen=None, epoch=None, train=model.training).squeeze(-1)
