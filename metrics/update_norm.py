import torch
from typing import Dict


@torch.no_grad()
def snapshot_params(model) -> Dict[str, torch.Tensor]:

    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


@torch.no_grad()
def update_norm(model, prev_params: Dict[str, torch.Tensor]) -> float:
    
    sq_sum = 0.0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        delta = p.detach() - prev_params[name]
        sq_sum += delta.norm(p=2).item() ** 2

    return sq_sum ** 0.5
