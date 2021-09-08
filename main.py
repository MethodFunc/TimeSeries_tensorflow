# import module
from util.all import *
from setting import *


if __name__ == "__main__":
    args = define_parser()
    raw_data = LoadDataframe(args.main_path).get_df()
    features, targets, data = preprocess_data(raw_data, args)

    # scaling support: minmax, standard, robust
    fs, fs_data = scaling(features, args.method)
    ts, ts_data = scaling(targets, args.method)

    # data setting
    train_set, val_set, test_set = make_dataset(fs_data, ts_data, args)

    # tensorflow setting
    optimizer = optim(args)
    # loss = RMSE()
    loss = loss_fn(args)
    train_metric = tf.keras.metrics.MeanAbsoluteError()
    val_metric = tf.keras.metrics.MeanAbsoluteError()

    model = conv_lstm(args)

    train(train_set, val_set, model, loss, optimizer, train_metric, val_metric, args)

    y_true = []
    y_predict = []
    for x, y in test_set:
        y_pred = predict_step(x, model)
        y_predict.append(y_pred.numpy())
        y_true.append(y.numpy())

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    # y_true, y_predict = forecasting(test_set, tensor_model=tensor_model, scale=ts, inverse=True)

    check_acc(y_true, y_predict)
    # evalution(y_true, y_predict)
    # predict_plot(y_true, y_predict)
