import tensorflow as tf
import numpy as np
from tensorflow import keras

def positional_encoding(max_length, d_model):
    position = np.arange(max_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_enc = np.zeros((max_length, d_model))

    # Calculate the positional encoding
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)

    return tf.convert_to_tensor(pos_enc, dtype=tf.float32)
