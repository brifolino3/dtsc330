"""
The decoder checks back on the misspelled input to fix it.
"""

import tensorflow as tf


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super().__init__()

        self.self_att = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = d_model)

        self.cross_att = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = d_model)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation = "relu"),
            tf.keras.layers.Dense(d_model)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()

    def call(self, x, enc_output):
        # decoder self-attention
        attn1 = self.self_att(x, x)
        x = self.norm1(x + attn1)

        # cross-attention
        attn2 = self.cross_att(x, enc_output)
        x = self.norm2(x + attn2)

        # feed forward
        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)