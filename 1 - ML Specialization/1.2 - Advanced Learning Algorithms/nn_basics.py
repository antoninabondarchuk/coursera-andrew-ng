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


def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    a = np.exp(z) / np.sum(np.exp(z))
    return a


model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(400,)),  # specify input size
        tf.keras.layers.Dense(units=25, activation='sigmoid'),
        tf.keras.layers.Dense(units=15, activation='sigmoid'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ], name="my_model"
)

softmax_model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(400,)),
        tf.keras.layers.Dense(units=25, activation='relu'),
        tf.keras.layers.Dense(units=15, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='linear')
    ], name="my_softmax_model"
)
softmax_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
