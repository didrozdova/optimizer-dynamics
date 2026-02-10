from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


@dataclass
class OptimConfig:
    name: str = "sgd"  
    lr: float = 1e-2
    weight_decay: float = 0.0
 
    momentum: float = 0.9
    nesterov: bool = False
 
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


def make_optimizer(
    params: Iterable[torch.nn.Parameter],
    cfg: OptimConfig,
) -> torch.optim.Optimizer:
    name = cfg.name.lower()

    if name in {"sgd", "vanilla_sgd"}:
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=0.0,
            weight_decay=cfg.weight_decay,
            nesterov=False,
        )

    if name in {"sgd_momentum", "momentum", "sgdm"}:
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )

    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )

    raise ValueError(
        f"Unknown optimizer: {cfg.name}. Supported: sgd, sgd_momentum, adam, adamw"
    )
