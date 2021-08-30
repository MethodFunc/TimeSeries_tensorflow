import numpy as np


def accuracy(y_true, y_pred):
    """
    calc acc
    """
    y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
    y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    error_rate = abs((y_pred - y_true)) / y_true * 100
    acc = [100 - err for err in error_rate]
    acc = np.array(acc)
    acc_mean = np.mean(acc)

    print(f"Acc Mean: {acc_mean:.4f}%")

    return acc_mean
