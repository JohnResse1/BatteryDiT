"""Pre-registered prediction heads, sharing one split and one preprocessing."""
from __future__ import annotations

import numpy as np

from . import _certified as C


def fit_scaler(x: np.ndarray) -> dict:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return {"mu": mu, "sd": sd}


def apply_scaler(scaler: dict, x: np.ndarray) -> np.ndarray:
    return (x - scaler["mu"]) / scaler["sd"]


def ridge(x_tr, y_tr, x_te, alpha: float = C.RIDGE_ALPHA):
    scaler = fit_scaler(x_tr)
    xtr, xte = apply_scaler(scaler, x_tr), apply_scaler(scaler, x_te)
    ymu = y_tr.mean(axis=0, keepdims=True)
    n, d = xtr.shape
    w = np.linalg.solve(xtr.T @ xtr + alpha * np.eye(d), xtr.T @ (y_tr - ymu))
    return xte @ w + ymu


def knn(x_tr, y_tr, x_te, k: int = C.KNN_NEIGHBOURS):
    from sklearn.neighbors import KNeighborsRegressor
    scaler = fit_scaler(x_tr)
    model = KNeighborsRegressor(n_neighbors=min(k, len(x_tr)), n_jobs=-1)
    model.fit(apply_scaler(scaler, x_tr), y_tr)
    return model.predict(apply_scaler(scaler, x_te))


def mlp(x_tr, y_tr, x_te, seed: int = 20260926):
    """Very small 2-layer MLP (default 32 -> 16); never tuned on evaluation data."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    y2d = y_tr if y_tr.ndim == 2 else y_tr.reshape(-1, 1)
    scaler = fit_scaler(x_tr)
    xtr, xte = apply_scaler(scaler, x_tr), apply_scaler(scaler, x_te)
    ymu, ysd = y2d.mean(0, keepdims=True), y2d.std(0, keepdims=True)
    ysd[ysd == 0] = 1.0
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], C.MLP_HIDDEN[0]), torch.nn.ReLU(),
        torch.nn.Linear(C.MLP_HIDDEN[0], C.MLP_HIDDEN[1]), torch.nn.ReLU(),
        torch.nn.Linear(C.MLP_HIDDEN[1], y2d.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=C.MLP_LEARNING_RATE)
    lossfn = torch.nn.MSELoss()
    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(((y2d - ymu) / ysd), dtype=torch.float32)
    net.train()
    for _ in range(C.MLP_EPOCHS):
        opt.zero_grad(); lossfn(net(xt), yt).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(xte, dtype=torch.float32)).numpy()
    out = out * ysd + ymu
    return out if y_tr.ndim == 2 else out.reshape(-1)


HEADS = {"ridge": ridge, "knn": knn, "mlp": mlp}
