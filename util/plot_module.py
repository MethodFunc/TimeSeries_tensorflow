__all__ = ["train_plot", "predict_plot"]

from .headers import *

"""
plot_module
auth: Methodfunc - Kwak Piljong
date: 2021.08.03
version: 0.1
"""


def train_plot(train_loss, val_loss):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(train_loss, label="train_loss")
    ax.plot(val_loss, label="val_loss")
    ax.legend()
    plt.show()


def predict_plot(y_true, y_pred):
    y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
    y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred

    plt.style.use("ggplot")
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    plt.rcParams["figure.figsize"] = (20, 7)
    plt.plot(y_true, label="actual")
    plt.plot(y_pred, label="predict")
    plt.legend()
    plt.show()
