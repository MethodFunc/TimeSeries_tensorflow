__all__ = ['min_to_hour', 'evalution', 'check_acc', 'forecasting']

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
    acc = []
    for i in error_rate:
        rate = 100 - i
        if rate < 0:
            continue
        else:
            acc.append(rate)

    acc = np.array(acc)
    print(f"Acc Mean: {acc.mean():.4f}%")
    print(f"Acc Max: {acc.max():.4f}%")
    print(f"Acc Min: {acc.min():.4f}%")


def forecasting(x, model, scale=None, inverse=False):
    true = []
    predicted = []
    for n, (X, y) in enumerate(x):
        true.append(y.numpy())
        temp = model.predict(X.numpy())
        predicted.append(temp)

    true = np.array(true)
    predicted = np.array(predicted)

    if inverse:
        if scale is None or not scale:
            raise 'scale is not define'
        true = scale.inverse_transform(true.reshape(-1, 1))
        predicted = scale.inverse_transform(predicted.reshape(-1, 1))

    return true, predicted
