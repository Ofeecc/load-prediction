"""
Probabilistic and deterministic evaluation metrics for GEFCom2014-L.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

QUANTILES_99 = np.round(np.arange(0.01, 1.00, 0.01), 2)


def pinball_loss_matrix(
    y_true: np.ndarray,
    y_quantiles: np.ndarray,
    quantiles: np.ndarray = QUANTILES_99,
) -> np.ndarray:
    """
    Compute the pinball loss matrix of shape (N, Q).
    y_true: shape (N,)
    y_quantiles: shape (N, Q)
    quantiles: shape (Q,) in (0, 1)
    """
    y = np.asarray(y_true, dtype=float).reshape(-1, 1)
    yq = np.asarray(y_quantiles, dtype=float)
    q = np.asarray(quantiles, dtype=float).reshape(1, -1)
    if yq.shape != (len(y), q.size) or len(y) == 0:
        raise ValueError("Forecast shape must be (N, Q), matching nonempty truth and quantile levels")
    if not all(np.isfinite(a).all() for a in (y, yq, q)):
        raise ValueError("Truth, forecasts and levels must be finite")
    if np.any((q <= 0) | (q >= 1)) or np.any(np.diff(q.ravel()) <= 0):
        raise ValueError("Quantile levels must be strictly increasing in (0, 1)")
    if np.any(np.diff(yq, axis=1) < -1e-10):
        raise ValueError("Crossing quantiles must be fixed explicitly before evaluation")
    diff = y - yq
    return np.maximum(q * diff, (q - 1.0) * diff)


def pinball_loss(
    y_true: np.ndarray,
    y_quantiles: np.ndarray,
    quantiles: np.ndarray = QUANTILES_99,
) -> float:
    """
    Mean Pinball Loss across all hours and all 99 quantiles (official GEFCom2014 metric).
    """
    return float(np.mean(pinball_loss_matrix(y_true, y_quantiles, quantiles)))


def evaluate_all_metrics(
    y_true: np.ndarray,
    y_quantiles: np.ndarray,
    y_point: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute comprehensive probabilistic and deterministic metrics for a 99-quantile forecast.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    yq = np.asarray(y_quantiles, dtype=float)
    # Validate before any indexing; evaluation must not silently repair predictions.
    pinball_loss_matrix(y, yq)
    yq_sorted = yq

    if y_point is None:
        # 50th percentile (index 49 corresponds to q=0.50)
        y_pt = yq_sorted[:, 49]
    else:
        y_pt = np.asarray(y_point, dtype=float).ravel()

    if y_pt.shape != y.shape or not np.isfinite(y_pt).all():
        raise ValueError("Point predictions must match finite truth shape")
    pb_mat = pinball_loss_matrix(y, yq_sorted)
    pb_mean = float(np.mean(pb_mat))
    # Tail pinball loss (q <= 0.10 or q >= 0.90)
    tail_mask = (QUANTILES_99 <= 0.10) | (QUANTILES_99 >= 0.90)
    pb_tail = float(np.mean(pb_mat[:, tail_mask]))
    # Central pinball loss (0.25 <= q <= 0.75)
    central_mask = (QUANTILES_99 >= 0.25) & (QUANTILES_99 <= 0.75)
    pb_central = float(np.mean(pb_mat[:, central_mask]))

    err = y - y_pt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    nonzero = y != 0
    mape = float(np.mean(np.abs(err[nonzero] / y[nonzero])) * 100.0) if nonzero.any() else float("nan")

    # 80% interval: q=0.10 (idx 9) to q=0.90 (idx 89)
    q10, q90 = yq_sorted[:, 9], yq_sorted[:, 89]
    picp_80 = float(np.mean((y >= q10) & (y <= q90)) * 100.0)

    # 90% interval: q=0.05 (idx 4) to q=0.95 (idx 94)
    q05, q95 = yq_sorted[:, 4], yq_sorted[:, 94]
    picp_90 = float(np.mean((y >= q05) & (y <= q95)) * 100.0)
    miw_90 = float(np.mean(q95 - q05))

    return {
        "Pinball": pb_mean,
        "Pinball_Tail": pb_tail,
        "Pinball_Central": pb_central,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "PICP_80(%)": picp_80,
        "PICP_90(%)": picp_90,
        "MIW_90(MW)": miw_90,
    }

