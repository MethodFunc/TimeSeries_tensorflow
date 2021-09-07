__all__ = ["data_split", "single_window_dataset", "multivariable_window_dataset"]

import tensorflow as tf

"""
datamaker
auth: Methodfunc - Kwak Piljong
date: 2021.08.03
modify_date: 2021.08.13
version: 0.2
"""


def data_split(feature, label, split_size=None):
    if split_size > 1.0:
        raise BaseException("Data split length over...")

    data_size = len(feature)
    train = int(round(data_size * split_size, 0))

    x_train, y_train = feature[:train], label[:train]
    x_test, y_test = feature[train:], label[train:]

    return x_train, y_train, x_test, y_test


def single_window_dataset(series, window_size: int, batch_size: int, shuffle=False):
    """
    Tensorflow Datamaker
    Only Single variable step maker
    """
    if not type(shuffle) is bool:
        raise "Shuffle is must be bool type"

    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(lambda window: (window[:-1], window[1:]))

    return dataset.batch(batch_size, drop_remainder=True).prefetch(1)


def multivariable_window_dataset(
    x, y, window_size: int, batch_size: int, shuffle=False
):
    """
    Tensorflow multivariable datamaker for timeseries
    x = feature
    y = target(label)
    """
    ds_x = tf.data.Dataset.from_tensor_slices(x)
    ds_x = ds_x.window(window_size, shift=1, drop_remainder=True)
    ds_x = ds_x.flat_map(lambda window: window.batch(window_size))

    ds_y = tf.data.Dataset.from_tensor_slices(y[window_size:])

    ds = tf.data.Dataset.zip((ds_x, ds_y))

    if shuffle:
        ds = ds.shuffle(10000)

    return ds.batch(batch_size, drop_remainder=True).prefetch(1)
