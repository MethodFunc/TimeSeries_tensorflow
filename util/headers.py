import numpy as np
import pandas as pd
import time

import argparse
import warnings
import matplotlib.pyplot as plt

from copy import deepcopy
import os
import datetime

warnings.filterwarnings("ignore")

# DeepLearning modules
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Conv1D,
    MaxPooling1D,
    TimeDistributed,
    Bidirectional,
    BatchNormalization,
    Activation,
    UpSampling1D,
    Conv1DTranspose,
)
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import Loss

from tensorflow.keras.losses import (
    Huber,
    mse,
    mae,
    binary_crossentropy,
    categorical_crossentropy,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
