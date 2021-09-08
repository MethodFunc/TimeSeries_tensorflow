from ..headers import *


def conv_lstm(args):
    K.clear_session()

    input_layer = Input(shape=(args.window_size, args.feature_size))
    x = Conv1D(128, kernel_size=3, padding="causal")(input_layer)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(256, kernel_size=3, padding="causal")(x)
    x = Activation(tf.nn.relu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(256, dropout=0.3, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, dropout=0.4, return_sequences=False))(x)
    output = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output)

    return model
