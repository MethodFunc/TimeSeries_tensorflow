__all__ = ['CustomHistory', 'RMSE']

from .headers import *

"""
tensorflow custom callback
auth: Methodfunc - Kwak Piljong
date: 2021.08.20
modify date: 2021.08.24
version: 0.2
"""


class CustomHistory(tf.keras.callbacks.Callback):
    """
    log loss history
    """

    def __init__(self):
        super(CustomHistory, self).__init__()
        self.total_train = []
        self.total_val = []
        self.get_logs = {}

        self.ts = None
        self.timeit = None
        self.df = pd.DataFrame()

    def on_epoch_begin(self, batch, ogs=None):
        self.ts = time.time()

    def on_epoch_end(self, batch, logs=[]):
        self.total_train.append(logs.get('loss'))
        self.total_val.append(logs.get('val_loss'))
        self.timeit = time.time() - self.ts
        self.get_logs['time'] = self.timeit
        for key, values in logs.items():
            self.get_logs[key] = values

        self.update_df()

    def get_df(self):
        return self.df

    def update_df(self):
        self.df = self.df.append(self.get_logs, ignore_index=True)


class RMSE(Loss):
    def __int__(self):
        super(RMSE, self).__int__()

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        sqr_error = K.square(error)
        mean_sqr_error = K.mean(sqr_error)
        sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)

        return sqrt_mean_sqr_error
