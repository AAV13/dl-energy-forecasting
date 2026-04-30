import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def compute_all_metrics(y_true, y_pred, target_names):
    """
    y_true, y_pred: numpy arrays of shape (n_samples, n_targets, horizon)
    target_names: list of target variable names
    Returns a dict of {target: {mae, rmse, mape}}
    """
    results = {}
    for i, name in enumerate(target_names):
        t = y_true[:, i, :]
        p = y_pred[:, i, :]
        results[name] = {
            "MAE":  round(mae(t, p), 4),
            "RMSE": round(rmse(t, p), 4),
            "MAPE": round(mape(t, p), 4)
        }
    return results