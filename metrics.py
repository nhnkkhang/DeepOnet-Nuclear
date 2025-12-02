import numpy as np
from sklearn import metrics

from . import config


def accuracy(y_true, y_pred):
    return np.mean(np.equal(np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)))


def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def nanl2_relative_error(y_true, y_pred):
    """Return the L2 relative error treating Not a Numbers (NaNs) as zero."""
    err = y_true - y_pred
    err = np.nan_to_num(err)
    y_true = np.nan_to_num(y_true)
    return np.linalg.norm(err) / np.linalg.norm(y_true)


def mean_l2_relative_error(y_true, y_pred):
    """Compute the average of L2 relative error along the first axis."""
    return np.mean(
        np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)
    )


def _absolute_percentage_error(y_true, y_pred):
    return 100 * np.abs(
        (y_true - y_pred) / np.clip(np.abs(y_true), np.finfo(config.real(np)).eps, None)
    )

def _absolute_error(y_true, y_pred):
    return np.abs(
        (y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(_absolute_percentage_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(_absolute_error(y_true, y_pred))

def max_absolute_percentage_error(y_true, y_pred):
    return np.amax(_absolute_percentage_error(y_true, y_pred))


def absolute_percentage_error_std(y_true, y_pred):
    return np.std(_absolute_percentage_error(y_true, y_pred))


def mean_squared_error(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)


def mod_mse_max(y_true, y_pred):
    """
    Computes Composite Loss (MSE + Peak) using NumPy.
    """
    # 1. Create Mask: (y_true != 0) returns bool, .astype converts to float
    mask = (y_true != 0).astype(np.float32)

    # --- MSE Calculation ---
    squared_error = np.square(y_true - y_pred)
    masked_squared_error = squared_error * mask
    
    # np.sum with no axis sums the entire array
    mse_loss = np.sum(masked_squared_error) / (np.sum(mask) + 1e-7)

    # --- Peak Calculation ---
    # axis=1 computes max across the feature dimension (1, 1296) -> (1,)
    max_true = np.max(y_true, axis=1)
    
    # Ensure masked values (0s) don't interfere if data is negative
    # (Assuming data is generally positive or 0 is the floor)
    max_pred = np.max(y_pred * mask, axis=1) 
    
    peak_l = np.mean(np.square(max_true - max_pred))

    # --- Composite ---
    alpha = 0.1
    return (1.0 - alpha) * mse_loss + alpha * peak_l


def mod_mae_max(y_true, y_pred):
    """
    Computes Composite Loss (MAE + Peak Abs) using NumPy.
    """
    # 1. Create Mask
    mask = (y_true != 0).astype(np.float32)

    # --- MAE Calculation ---
    absolute_error = np.abs(y_true - y_pred)
    masked_absolute_error = absolute_error * mask
    
    mae_loss = np.sum(masked_absolute_error) / (np.sum(mask) + 1e-7)

    # --- Peak Calculation ---
    max_true = np.max(y_true, axis=1)
    max_pred = np.max(y_pred * mask, axis=1)
    
    # Using Absolute difference for peak to match MAE logic
    peak_l = np.mean(np.abs(max_true - max_pred))

    # --- Composite ---
    alpha = 0.1
    return (1.0 - alpha) * mae_loss + alpha * peak_l

def get(identifier):
    metric_identifier = {
        "accuracy": accuracy,
        "l2 relative error": l2_relative_error,
        "nanl2 relative error": nanl2_relative_error,
        "mean l2 relative error": mean_l2_relative_error,
        "mean squared error": mean_squared_error,
        "customMSE": mod_mse_max,
        "customMAE": mod_mae_max,
        "mae": mean_absolute_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
        "MAPE": mean_absolute_percentage_error,
        "max APE": max_absolute_percentage_error,
        "APE SD": absolute_percentage_error_std,
    }

    if isinstance(identifier, str):
        return metric_identifier[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret metric function identifier:", identifier)
