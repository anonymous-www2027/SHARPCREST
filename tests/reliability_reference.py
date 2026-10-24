"""Small NumPy reference for controller tests."""
import numpy as np
import torch


def cr_reference(similarity, futures, k, training_futures, tau_v=1.0,
                 tool=None, candidate_labels=None):
    order = np.argsort(-similarity.detach().cpu().numpy(), kind='stable')
    if tool is None:
        positions = order[:k].tolist()
    else:
        labels = candidate_labels.detach().cpu().numpy()
        if tool.panel_mode:
            series, starts = tool.panel_candidate_manifest.resolve_ids(labels)
        else:
            series, starts = np.zeros_like(labels), labels
        positions = []
        for pos in order:
            if all(series[pos] != series[other] or abs(starts[pos] - starts[other]) >= tool.pred_len
                   for other in positions):
                positions.append(int(pos))
                if len(positions) == k:
                    break
    k = len(positions)
    A_q = similarity[positions].mean(dim=-1)
    selected = futures[positions].detach().cpu().numpy().astype(np.float64)
    train = training_futures.detach().cpu().numpy().astype(np.float64)
    scale = np.quantile(train, .75, axis=(0, 1)) - np.quantile(train, .25, axis=(0, 1)) + 1e-8
    if k == 1:
        V_q = 1.0
    else:
        V_q = np.mean([np.exp(-np.mean(np.abs(selected[i] - selected[j]) / scale) / tau_v)
                       for i in range(k) for j in range(i + 1, k)])
    return A_q * A_q.new_tensor(V_q)


def use_past_only_controller(tool):
    """Isolate unchanged downstream behavior in historical output-parity tests."""
    from types import MethodType
    from src.utils.profile_reliability import topk_reliability, map_reliability
    def old_controller(self, similarity, candidate_labels):
        A_q = topk_reliability(similarity, self.profile_reliability_k)
        sigma_q, lambda_q = map_reliability(A_q, self.sigma_min, self.sigma_max,
                                          self.lambda_min, self.lambda_max)
        return A_q, sigma_q, lambda_q
    tool._candidate_reliability = MethodType(old_controller, tool)
