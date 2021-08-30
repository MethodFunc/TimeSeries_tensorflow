from ..headers import *


def case3_model(args):
    model = Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=(args.window_size, args.feature_size)),
        LSTM(128, dropout=0.18, return_sequences=True),
        LSTM(128, dropout=0.24),

        Dense(1)
    ])

    return model


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


def model_home(args):
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


def many_to_many_model(args):
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


def cus_model(args):
    K.clear_session()
    inp = Input(batch_shape=(args.batch_size, args.window_size, args.feature_size))
    x = Conv1D(64, kernel_regularizer='l2', kernel_size=3)(inp)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = Conv1D(128, kernel_regularizer='l2', kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(256, kernel_regularizer='l2', kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = Conv1D(512, kernel_regularizer='l2', kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = MaxPooling1D(2)(x)
    encode_out = Dropout(0.3)(x)

    # Decode

    x = UpSampling1D(2)(encode_out)
    x = Conv1DTranspose(512, kernel_regularizer='l2', kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = Conv1DTranspose(256, kernel_regularizer='l2', kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.28)(x)

    x = UpSampling1D(2)(x)
    x = Conv1DTranspose(128, kernel_regularizer='l2', kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.relu)(x)
    x = Conv1DTranspose(64, kernel_regularizer='l2', kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.sigmoid)(x)
    decode_out = Dropout(0.28)(x)

    # stateful lstm layers
    x = LSTM(128, return_sequences=True, dropout=0.2)(decode_out)
    x = LSTM(64, return_sequences=True, dropout=0.2)(x)
    x = LSTM(32, return_sequences=False, dropout=0.19)(x)

    # fully connect network layers
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.18)(x)
    outp = Dense(1)(x)

    # create model
    model = Model(inputs=inp, outputs=outp)

    return model
