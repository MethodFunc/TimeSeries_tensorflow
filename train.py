# import module
from util.all import *
from setting import *


def preprocess_data(args):
    raw_data = LoadDataframe(args.main_path).get_df()
    rotorspeed_df = LoadDataframe(args.sub_path).get_df()
    rotorspeed_df.drop('//DateTime', axis=1, inplace=True)
    merge_df = pd.merge(raw_data, rotorspeed_df, left_index=True, right_index=True)
    clean_df = cleanup_df(merge_df)
    pre_df = processing_data(clean_df, shift=None)
    pre_df = pre_df[['Direction_mean', 'WindSpeed', 'RotorP', 'RotorSpeed', 'Average_PB', 'GenertorTR', 'ActiveP']]

    feature = pre_df.drop(["ActiveP"], axis=1).values
    target = pre_df["ActiveP"].values.reshape(-1, 1)
    args.feature_size = feature.shape[-1]

    return feature, target, pre_df


def make_dataset(feature, label, args):
    x_train, y_train, x_val, y_val = data_split(feature, label, split_size=args.train_size)
    x_val, y_val, x_test, y_test = data_split(x_val, y_val, split_size=args.val_size)

    train = multivariable_window_dataset(x_train, y_train, window_size=args.window_size, batch_size=args.batch_size,
                                         shuffle=True)
    val = multivariable_window_dataset(x_val, y_val, window_size=args.window_size, batch_size=args.batch_size)
    test = multivariable_window_dataset(x_test, y_test, window_size=args.window_size, batch_size=args.batch_size)

    return train, val, test


if __name__ == '__main__':
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
    # loss = loss_fn(args)
    cus = CustomHistory()

    model = cus_model(args)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

    history = model.fit(train_set, epochs=args.epoch, validation_data=val_set, callbacks=[cus], verbose=2)

    train_plot(history)

    y_true, y_predict = forecasting(test_set, model=model, scale=ts, inverse=True)

    predict_plot(y_true[-144:], y_predict[-144:])

    check_acc(y_true, y_predict)
    evalution(y_true, y_predict)

    # min -> hour
    hour_true, hour_pred = min_to_hour(y_true, y_predict)
    predict_plot(hour_true, hour_pred)

    check_acc(hour_true, hour_pred)
    evalution(hour_true, hour_pred)

    extract_result(y_true, y_predict, idx=None)
