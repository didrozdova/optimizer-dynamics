import torch
from typing import Tuple
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_moons_dataset(
    n_samples: int = 2000,
    noise: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:

    X_np, y_np = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=seed,
    )

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)
    return X, y


def split_and_standardize(
    X: torch.Tensor,
    y: torch.Tensor,
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Train / val split + standardization (fit on train only).
    """
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_np,
        y_np,
        test_size=test_size,
        random_state=seed,
        stratify=y_np,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    device = X.device

    return (
        torch.tensor(X_tr, dtype=torch.float32, device=device),
        torch.tensor(y_tr, dtype=torch.long, device=device),
        torch.tensor(X_val, dtype=torch.float32, device=device),
        torch.tensor(y_val, dtype=torch.long, device=device),
        scaler,
    )
