# import module
from util.all import *
from setting import *


def preprocess_data(args):
    raw_data = LoadDataframe(args.main_path).get_df()
    clean_df = cleanup_df(raw_data)
    pre_df = processing_data(clean_df, shift=None)
    pre_df = pre_df[
        ["Direction_mean", "WindSpeed", "RotorP", "Average_PB", "GenertorTR", "ActiveP"]
    ]

    feature = pre_df.drop(["ActiveP"], axis=1).values
    target = pre_df["ActiveP"].values.reshape(-1, 1)
    args.feature_size = feature.shape[-1]

    return feature, target, pre_df


def make_dataset(feature, label, args):
    x_train, y_train, x_val, y_val = data_split(
        feature, label, split_size=args.train_size
    )
    x_val, y_val, x_test, y_test = data_split(x_val, y_val, split_size=args.val_size)

    train = multivariable_window_dataset(
        x_train,
        y_train,
        window_size=args.window_size,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val = multivariable_window_dataset(
        x_val, y_val, window_size=args.window_size, batch_size=args.batch_size
    )
    test = multivariable_window_dataset(
        x_test, y_test, window_size=args.window_size, batch_size=args.batch_size
    )

    return train, val, test


if __name__ == "__main__":
    args = define_parser()
    features, targets, data = preprocess_data(args)

    # scaling support: minmax, standard, robust
    fs, fs_data = scaling(features, args.method)
    ts, ts_data = scaling(targets, args.method)

    # data setting
    train_set, val_set, test_set = make_dataset(fs_data, ts_data, args)

    # tensorflow setting
    optimizer = optim(args)
    loss = RMSE()
    train_metric = tf.keras.metrics.MeanAbsoluteError()
    val_metric = tf.keras.metrics.MeanAbsoluteError()

    model = lstm_model(args)

    train(train_set, val_set, model, loss, optimizer, train_metric, val_metric, args)

    y_true, y_predict = forecasting(test_set, model=model, scale=ts, inverse=True)

    check_acc(y_true, y_predict)
    evalution(y_true, y_predict)
