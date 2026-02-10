import torch
from typing import Dict, Iterable, Optional


def grad_l2_norm(param: torch.Tensor) -> float:
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().norm(p=2).item())


def grad_global_l2_norm(params: Iterable[torch.Tensor]) -> float:
    sq_sum = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq_sum += float(g.pow(2).sum().item())
    return float(sq_sum ** 0.5)


def grad_norms_by_module(
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

        out[f"grad.{name}"] = grad_global_l2_norm(module.parameters())

    return out
