import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, precision_score,
    recall_score, f1_score, auc, accuracy_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, median_absolute_error,
    mean_absolute_percentage_error, max_error, r2_score
)
from scipy.stats import pearsonr


def get_cindex(y, p):
    summ, pair = 0, 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if y[i] > y[j]:
                pair += 1
                summ += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    return summ / pair if pair != 0 else 0


def _r_squared_error(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = np.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
    y_obs_sq = np.sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = np.sum((y_pred - y_pred_mean) ** 2)
    return mult / float(y_obs_sq * y_pred_sq)


def _get_k(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    return np.sum(y_obs * y_pred) / float(np.sum(y_pred * y_pred))


def _squared_error_zero(y_obs, y_pred):
    k = _get_k(y_obs, y_pred)
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    return 1 - np.sum((y_obs - k * y_pred) ** 2) / float(
        np.sum((y_obs - y_obs_mean) ** 2))


def get_rm2(ys_orig, ys_line):
    r2 = _r_squared_error(ys_orig, ys_line)
    r02 = _squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute(r2 * r2 - r02 * r02)))


def get_metrics_reg(y_true, y_pred, title, with_rm2=False, with_ci=False):
    metrics = {
        f"{title}_mse": float(mean_squared_error(y_true, y_pred)),
        f"{title}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{title}_medae": float(median_absolute_error(y_true, y_pred)),
        f"{title}_mape": float(
            mean_absolute_percentage_error(y_true, y_pred)),
        f"{title}_maxe": float(max_error(y_true, y_pred)),
        f"{title}_r2": float(r2_score(y_true, y_pred)),
        f"{title}_pearsonr": pearsonr(
            y_true.flatten(), y_pred.flatten())[0],
    }
    if with_rm2:
        metrics[f"{title}_rm2"] = get_rm2(
            y_true.flatten().tolist(), y_pred.flatten().tolist())
    if with_ci:
        metrics[f"{title}_ci"] = get_cindex(
            y_true.flatten().tolist(), y_pred.flatten().tolist())
    return metrics


def get_metrics_cls(y_true, y_pred, title,
                    transform=torch.sigmoid, threshold=0.5):
    if transform is not None:
        y_pred = transform(y_pred)
    y_pred_lbl = (y_pred >= threshold).float()
    metrics = {
        "f1": float(f1_score(y_true, y_pred_lbl)),
        "precision": float(
            precision_score(y_true, y_pred_lbl, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_lbl)),
        "accuracy": float(accuracy_score(y_true, y_pred_lbl)),
        "mcc": float(matthews_corrcoef(y_true, y_pred_lbl)),
    }
    try:
        metrics["rocauc"] = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        metrics["rocauc"] = np.nan
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        metrics["prauc"] = float(auc(rec, prec))
    except ValueError:
        metrics["prauc"] = np.nan
    return metrics
