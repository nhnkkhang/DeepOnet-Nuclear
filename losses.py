from . import backend as bkd
from . import config
from .backend import tf


def mean_absolute_error(y_true, y_pred):
    return bkd.reduce_mean(bkd.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)


def mean_squared_error(y_true, y_pred):
    # Warning:
    # - Do not use ``tf.losses.mean_squared_error``, which casts `y_true` and `y_pred` to ``float32``.
    # - Do not use ``tf.keras.losses.MSE``, which computes the mean value over the last dimension.
    # - Do not use ``tf.keras.losses.MeanSquaredError()``, which casts loss to ``float32``
    #     when calling ``compute_weighted_loss()`` calling ``scale_losses_by_sample_weight()``,
    #     although it finally casts loss back to the original type.
    return bkd.reduce_mean(bkd.square(y_true - y_pred))


def mean_l2_relative_error(y_true, y_pred):
    return bkd.reduce_mean(bkd.norm(y_true - y_pred, axis=1) / bkd.norm(y_true, axis=1))


def softmax_cross_entropy(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)


def zero(*_):
    # TODO: pytorch
    return tf.constant(0, dtype=config.real(tf))

def mod_mse_max(y_true, y_pred):
    
    # Define helper functions (or move them outside if preferred)
    def masked_mse_loss(y_t, y_p):
        mask = tf.cast(tf.not_equal(y_t, 0), bkd.float32)
        squared_error = bkd.square(y_t - y_p)
        masked_squared_error = squared_error * mask
        # Add epsilon for numerical stability
        loss = bkd.reduce_sum(masked_squared_error) / (bkd.reduce_sum(mask) + 1e-7)
        return loss

    def peak_loss(y_t, y_p):
        mask = tf.cast(tf.not_equal(y_t, 0), bkd.float32)
        max_true = tf.reduce_max(y_t, axis=1)
        max_pred = tf.reduce_max(y_p * mask, axis=1)
        p_loss = tf.reduce_mean(bkd.square(max_true - max_pred))
        return p_loss

    # --- Main Logic ---
    alpha = 0.1
    mse = masked_mse_loss(y_true, y_pred)
    peak = peak_loss(y_true, y_pred)

    final_loss = (1.0 - alpha) * mse + alpha * peak
    
    return final_loss

def mod_mae_max(y_true, y_pred):
    
    def masked_mae_loss(y_t, y_p):
        mask = tf.cast(tf.not_equal(y_t, 0), tf.float32)
        
        absolute_error = tf.abs(y_t - y_p)
        
        masked_absolute_error = absolute_error * mask

        loss = tf.reduce_sum(masked_absolute_error) / (tf.reduce_sum(mask) + 1e-7)
        return loss

    def peak_loss(y_t, y_p):
        mask = tf.cast(tf.not_equal(y_t, 0), tf.float32)
        
        max_true = tf.reduce_max(y_t, axis=1)
        max_pred = tf.reduce_max(y_p * mask, axis=1)
        
        p_loss = tf.reduce_mean(tf.abs(max_true - max_pred))
        return p_loss

    alpha = 0.1
    
    mae = masked_mae_loss(y_true, y_pred)
    peak = peak_loss(y_true, y_pred)
    
    final_loss = (1.0 - alpha) * mae + alpha * peak
    
    return final_loss

LOSS_DICT = {
    "mean absolute error": mean_absolute_error,
    "customMSE": mod_mse_max,
    "customMAE": mod_mae_max,
    "MAE": mean_absolute_error,
    "mae": mean_absolute_error,
    "mean squared error": mean_squared_error,
    "MSE": mean_squared_error,
    "mse": mean_squared_error,
    "mean absolute percentage error": mean_absolute_percentage_error,
    "MAPE": mean_absolute_percentage_error,
    "mape": mean_absolute_percentage_error,
    "mean l2 relative error": mean_l2_relative_error,
    "softmax cross entropy": softmax_cross_entropy,
    "zero": zero,
}


def get(identifier):
    """Retrieves a loss function.

    Args:
        identifier: A loss identifier. String name of a loss function, or a loss function.

    Returns:
        A loss function.
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    if isinstance(identifier, str):
        return LOSS_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)
