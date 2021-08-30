from ..headers import *


def simgle_many(args):
    model = Sequential([
        Conv1D(filters=256, kernel_size=3,
               strides=1, padding="causal",
               activation="relu",
               input_shape=(args.window_size, args.feature_size)),
        Conv1D(filters=512, kernel_size=3,
               strides=1, padding="causal",
               activation="relu"),
        # Bidirectional(LSTM(512, dropout=0.12, return_sequences=True)),
        (LSTM(512, dropout=0.12, return_sequences=True)),
        Bidirectional(LSTM(1024, dropout=0.16, return_sequences=True)),

        TimeDistributed(Dense(1))
    ])

    return model


def many_to_many_func(args):
    K.clear_session()

    input_layer = Input(shape=(args.window_size, args.feature_size))
    x = Conv1D(128, kernel_size=3, padding='causal')(input_layer)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)
    x = Conv1D(256, kernel_size=3, padding='causal')(x)
    x = Activation(tf.nn.relu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)
    x = (LSTM(256, dropout=0.14, return_sequences=True))(x)
    x = Bidirectional(LSTM(512, dropout=0.18, return_sequences=True))(x)
    output = TimeDistributed(Dense(1))(x)

    model = Model(inputs=input_layer, outputs=output)

    return model
