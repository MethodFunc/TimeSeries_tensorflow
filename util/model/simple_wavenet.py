from ..headers import *


def simple_wavenet(args):
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(args.window_size, args.feature_size)))
    for rate in (1, 2, 4, 8) * 2:
        # use dilated convolution
        model.add(Conv1D(filters=20, kernel_size=2, padding="causal",
                         activation="relu", dilation_rate=rate))
    # output layer (1*1 convolution)
    model.add(Conv1D(filters=10, kernel_size=1))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    return model
