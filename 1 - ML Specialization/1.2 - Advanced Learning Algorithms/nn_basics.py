import numpy as np
import tensorflow as tf


def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    a_out = g(np.matmul(a_in, W) + b)
    return a_out


model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(400,)),  # specify input size
        tf.keras.layers.Dense(units=25, activation='sigmoid'),
        tf.keras.layers.Dense(units=15, activation='sigmoid'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ], name="my_model"
)

