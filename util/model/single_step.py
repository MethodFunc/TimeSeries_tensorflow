from ..headers import *


def single_step_model(args):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, padding='causal', activation='relu',
                     input_shape=(args.window_size, args.feature_size)))
    model.add(Conv1D(64, kernel_size=3, padding='causal', activation='relu'))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dense(1))

    return model
