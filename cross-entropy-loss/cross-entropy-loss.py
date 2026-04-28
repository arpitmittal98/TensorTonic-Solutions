import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    N = len(y_true)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_corrected = y_pred[np.arange(len(y_pred)), y_true]
    ce_loss = -1/N*np.sum(np.log(y_pred_corrected))

    return ce_loss