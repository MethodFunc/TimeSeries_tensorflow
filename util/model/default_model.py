from ..headers import *


def default_model(args):
    model = Sequential()
    model.add(LSTM(128, dropout=0.24, input_shape=(args.window_size, args.feature_size), return_sequences=True))
    model.add(LSTM(256, dropout=0.32))
    model.add(Dense(1))

    return model
