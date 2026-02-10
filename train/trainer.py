from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from metrics.grad_norm import grad_global_l2_norm, grad_norms_by_module
from metrics.param_norm import param_global_l2_norm, param_norms_by_module
from metrics.update_norm import snapshot_params, update_norm


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 256
    log_every: int = 50
    clip_norm: Optional[float] = None
    device: str = "cpu"
    sat_logit_thr: float = 8.0  


@torch.no_grad()
def acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (logits > 0).to(dtype=torch.long)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def sat_rate_from_logits(logits: torch.Tensor, thr: float) -> float:
    return float((logits.abs() > thr).float().mean().item())


def train_classifier(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    cfg: TrainConfig = TrainConfig(),
    module_names_for_norms: Sequence[str] = ("fc1", "fc2"),
) -> Dict[str, list]:
   
    device = torch.device(cfg.device)
    model.to(device)

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    if X_val is not None:
        X_val = X_val.to(device)
    if y_val is not None:
        y_val = y_val.to(device)

    n = X_train.shape[0]
    model.train()

    history: Dict[str, list] = {
        "step": [],
        "train.loss": [],
        "train.acc": [],
        "val.loss": [],
        "val.acc": [],
        "logits.mean": [],
        "logits.std": [],
        "sat_rate": [],
        "z1.mean": [],
        "z1.std": [],
        "active_frac": [],
        "grad.global": [],
        "param.global": [],
        "update.global": [],
    }

    for name in module_names_for_norms:
        history[f"grad.{name}"] = []
        history[f"param.{name}"] = []

    for step in range(cfg.steps):
        idx = torch.randint(0, n, (cfg.batch_size,), device=device)
        xb, yb = X_train[idx], y_train[idx]

        prev = snapshot_params(model)

        logits, z1, a1 = model(xb)
        logits = logits.view(-1)
        y_float = yb.to(dtype=torch.float32)

        loss = F.binary_cross_entropy_with_logits(logits, y_float)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_norm)

        optimizer.step()

        if step % cfg.log_every == 0 or step < 10:
            with torch.no_grad():
                tr_acc = acc_from_logits(logits, yb)

                if X_val is not None and y_val is not None:
                    model.eval()
                    val_logits, _, _ = model(X_val)
                    val_logits = val_logits.view(-1)
                    val_loss = float(
                        F.binary_cross_entropy_with_logits(val_logits, y_val.float()).item()
                    )
                    val_acc = acc_from_logits(val_logits, y_val)
                    model.train()
                else:
                    val_loss, val_acc = float("nan"), float("nan")

                history["step"].append(step)
                history["train.loss"].append(float(loss.item()))
                history["train.acc"].append(tr_acc)
                history["val.loss"].append(val_loss)
                history["val.acc"].append(val_acc)

                history["logits.mean"].append(float(logits.mean().item()))
                history["logits.std"].append(float(logits.std().item()))
                history["sat_rate"].append(sat_rate_from_logits(logits, cfg.sat_logit_thr))

                history["z1.mean"].append(float(z1.mean().item()))
                history["z1.std"].append(float(z1.std().item()))
                history["active_frac"].append(float((z1 > 0).float().mean().item()))

                history["grad.global"].append(grad_global_l2_norm(model.parameters()))
                history["param.global"].append(param_global_l2_norm(model.parameters()))
                history["update.global"].append(update_norm(model, prev))

                gmods = grad_norms_by_module(model, module_names_for_norms)
                pmods = param_norms_by_module(model, module_names_for_norms)
                for name in module_names_for_norms:
                    history[f"grad.{name}"].append(gmods.get(f"grad.{name}", 0.0))
                    history[f"param.{name}"].append(pmods.get(f"param.{name}", 0.0))

    return history

