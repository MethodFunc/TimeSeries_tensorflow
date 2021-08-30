from ..headers import *


def autoencoder(args):
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
