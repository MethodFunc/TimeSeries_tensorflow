from util.headers import *


# Default setting
def define_parser():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    # set load_data path
    args.main_path = "./main_data"

    # set datasets
    args.train_size = 0.8
    args.val_size = 0.5
    args.window_size = 72
    args.batch_size = 128

    # scaling support: minmax, standard, robust
    args.method = "minmax"

    # set tensorflow
    # optim list = adam, sgd, rmsprop
    args.optim = "adam"

    # loss_fn list = huber, mse, mae, binary, category
    args.loss_fn = "mse"
    args.lr = 1e-3
    args.epochs = 100

    return args
