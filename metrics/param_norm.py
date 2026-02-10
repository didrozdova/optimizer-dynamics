import torch
from typing import Dict, Iterable, Optional


def param_l2_norm(param: torch.Tensor) -> float:
    return float(param.detach().norm(p=2).item())


def param_global_l2_norm(params: Iterable[torch.Tensor]) -> float:
    sq_sum = 0.0
    for p in params:
        w = p.detach()
        sq_sum += float(w.pow(2).sum().item())
    return float(sq_sum ** 0.5)


def param_norms_by_module(
    model: torch.nn.Module,
    module_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:

    if module_names is None:
        module_names = []

    out: Dict[str, float] = {}

    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            continue

        out[f"param.{name}"] = param_global_l2_norm(module.parameters())

    return out
