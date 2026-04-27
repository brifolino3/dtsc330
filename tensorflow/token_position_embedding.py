"""
This module defines positional encoding for the Transformer.
"""

import keras
import tensorflow as tf
import numpy as np


class TokenPositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = self._build_encoding(max_len, d_model)

    def _build_encoding(self, max_len, d_model):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]

        angles = pos / np.power(10000, (2 * (i // 2)) / d_model)

        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        return tf.cast(angles[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]