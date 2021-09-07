from ..headers import *


def lstm_model(args):
    input_layer = Input(shape=(args.window_size, args.feature_size), name="input_layer")
    x = (LSTM(128, return_sequences=True))(input_layer)
    x = (LSTM(64, dropout=0.3))(x)
    x = BatchNormalization(batch_size=128)(x)
    x = Dense(64)(x)
    x = Activation(tf.nn.relu())(x)
    x = Dense(32)(x)
    x = Activation(tf.nn.relu())(x)
    output = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output)

    return model
