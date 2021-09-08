__all__ = ["min_to_hour", "evalution", "check_acc"]

from .headers import *

"""
metrics
auth: Methodfunc - Kwak Piljong
date: 2021.08.03
modify_date: 2021.08.24
version: 0.3
"""


def min_to_hour(y_true, y_pred):
    hourly_true = []
    hourly_pred = []
    hourly_true_sum = 0
    hourly_pred_sum = 0

    for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
        hourly_true_sum += true_val
        hourly_pred_sum += pred_val
        if i % 6 == 5:
            hourly_true = np.append(hourly_true, hourly_true_sum, axis=0)
            hourly_pred = np.append(hourly_pred, hourly_pred_sum, axis=0)
            hourly_true_sum = 0
            hourly_pred_sum = 0

    return hourly_true, hourly_pred


def evalution(y_true, y_pred):
    """
    calc mae, mse, mape
    """
    y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
    y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae = abs(y_true - y_pred).mean()
    mse = ((y_true - y_pred) ** 2).mean()
    mape = (abs((y_true - y_pred) / y_true * 100)).mean()

    print(f"Mean absolute error: {mae:.4f}")
    print(f"Mean squared error: {mse:.4f}")
    print(f"Mean absolute percentage error: {mape:.4f}")


def check_acc(y_true, y_pred):
    """
    calc acc
    """
    y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
    y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    error_rate = abs((y_pred - y_true)) / y_true * 100
    calc_acc = np.array([100 - i for i in error_rate if i != np.inf])

    acc = np.zeros(calc_acc.shape)

    for i in range(len(calc_acc)):
        if calc_acc[i] < 0:
            acc[i] = 0
        else:
            acc[i] = calc_acc[i]

    acc_mean = np.mean(acc)

    print(f"Acc Mean: {acc_mean:.4f}%")
    print(f"Acc Min : {np.min(acc):.4f}%")
    print(f"Acc Max : {np.max(acc):.4f}%")
